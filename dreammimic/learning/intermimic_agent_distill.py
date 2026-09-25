# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from isaacgym.torch_utils import *

import numpy as np
import torch 
from torch import nn

import learning.intermimic_agent as intermimic_agent


class InterMimicAgentDistill(intermimic_agent.InterMimicAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.expert_loss_coef = config['expert_loss_coef']
        self.entropy_coef = config['entropy_coef']
        self.ev_ma            = 0.0   # running avg explained‑variance
        self.critic_win_streak = 0    # consecutive windows EV ≥ threshold
        self.actor_update_num = 0
        
        # Distillation mode:
        # - dagger_only: teacher actions collect rollouts, supervise student.
        # - dagger_studentact: student actions collect rollouts, supervise student.
        # - dagger_rl: anneal teacher rollout actions to student rollout actions.
        # - dagger_rl_studentact: DAgger with a fixed teacher rollout fraction,
        #   then RL with student-only rollout actions.
        # - pcg_studentact: student rollouts with performance-conditioned
        #   teacher action blending and supervised guidance.
        # - rl_only: student actions, no teacher supervision loss.
        self.distillation_mode = config.get('distillation_mode', 'dagger_rl')
        if self.distillation_mode not in ['dagger_rl', 'dagger_rl_studentact', 'dagger_only', 'dagger_studentact', 'pcg_studentact', 'rl_only']:
            raise ValueError(f"Unknown distillation_mode: {self.distillation_mode}. Options: 'dagger_rl', 'dagger_rl_studentact', 'dagger_only', 'dagger_studentact', 'pcg_studentact', 'rl_only'")
        self.dagger_studentact_teacher_ratio = float(config.get('dagger_studentact_teacher_ratio', 0.25))
        self.dagger_studentact_rl_start_epoch = int(config.get('dagger_studentact_rl_start_epoch', 5000))
        self.dagger_studentact_teacher_ratio = min(max(self.dagger_studentact_teacher_ratio, 0.0), 1.0)
        self._dagger_rollout_reward_stats = None
        self._dagger_teacher_episode_rewards = None
        self._dagger_student_episode_rewards = None
        self._dagger_env_is_teacher = None
        self._dagger_last_teacher_ratio = None
        self._init_pcg_config(config)
        self._pcg_teacher_episode_rewards = None
        self._pcg_student_episode_rewards = None
        
        print(f"\n{'='*80}")
        print(f"Distillation Mode: {self.distillation_mode.upper()}")
        if self.distillation_mode == 'dagger_rl':
            print("  Using DAgger + RL progressive training")
        elif self.distillation_mode == 'dagger_rl_studentact':
            print("  Using DAgger + RL with student-majority rollouts")
            print(f"  DAgger teacher action ratio: {self.dagger_studentact_teacher_ratio:.3f}")
            print(f"  RL student-only rollout starts at epoch: {self.dagger_studentact_rl_start_epoch}")
        elif self.distillation_mode == 'dagger_only':
            print("  Using DAgger only (supervised learning)")
        elif self.distillation_mode == 'dagger_studentact':
            print("  Using DAgger with student rollout actions")
        elif self.distillation_mode == 'pcg_studentact':
            print("  Using performance-conditioned guidance with student-majority rollouts")
        elif self.distillation_mode == 'rl_only':
            print("  Using RL only (no teacher supervision)")
        print(f"{'='*80}\n")
        
        return

    def _get_dagger_studentact_teacher_ratio(self):
        if self.distillation_mode != 'dagger_rl_studentact':
            return None
        if int(getattr(self, 'epoch_num', 0)) >= self.dagger_studentact_rl_start_epoch:
            return 0.0
        return self.dagger_studentact_teacher_ratio

    def _dagger_init_episode_reward_meters(self):
        if self._dagger_teacher_episode_rewards is None:
            self._dagger_teacher_episode_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        if self._dagger_student_episode_rewards is None:
            self._dagger_student_episode_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)

    def _dagger_make_role_mask(self, count):
        count = int(count)
        mask = torch.zeros(count, dtype=torch.bool, device=self.ppo_device)
        if count <= 0:
            return mask
        ratio = self._get_dagger_studentact_teacher_ratio()
        ratio = 0.0 if ratio is None else float(np.clip(ratio, 0.0, 1.0))
        teacher_count = int(round(count * ratio))
        teacher_count = min(max(teacher_count, 0), count)
        if teacher_count > 0:
            mask[torch.randperm(count, device=self.ppo_device)[:teacher_count]] = True
        return mask

    def _dagger_init_env_roles(self):
        if self.distillation_mode != 'dagger_rl_studentact':
            return
        num_envs = self.vec_env.env.task.num_envs
        ratio = self._get_dagger_studentact_teacher_ratio()
        ratio = 0.0 if ratio is None else float(np.clip(ratio, 0.0, 1.0))
        needs_init = self._dagger_env_is_teacher is None or self._dagger_env_is_teacher.shape[0] != num_envs
        ratio_changed = self._dagger_last_teacher_ratio is None or abs(float(self._dagger_last_teacher_ratio) - ratio) > 1e-6
        if needs_init or ratio_changed:
            self._dagger_env_is_teacher = self._dagger_make_role_mask(num_envs)
            self._dagger_last_teacher_ratio = ratio

    def _dagger_resample_env_roles(self, done_indices):
        if self.distillation_mode != 'dagger_rl_studentact':
            return
        self._dagger_init_env_roles()
        if done_indices is None:
            return
        if isinstance(done_indices, torch.Tensor):
            if done_indices.numel() == 0:
                return
            env_ids = done_indices.view(-1).to(self.ppo_device).long()
        else:
            env_ids = to_torch(done_indices, device=self.ppo_device, dtype=torch.long).view(-1)
            if env_ids.numel() == 0:
                return
        self._dagger_env_is_teacher[env_ids] = self._dagger_make_role_mask(env_ids.numel())

    def _dagger_update_episode_reward_meters(self, done_indices):
        if self.distillation_mode != 'dagger_rl_studentact':
            return
        if self._dagger_env_is_teacher is None or done_indices is None:
            return
        self._dagger_init_episode_reward_meters()
        if isinstance(done_indices, torch.Tensor):
            if done_indices.numel() == 0:
                return
            env_ids = done_indices.view(-1).to(self.ppo_device).long()
        else:
            env_ids = to_torch(done_indices, device=self.ppo_device, dtype=torch.long).view(-1)
            if env_ids.numel() == 0:
                return
        done_rewards = self.current_rewards[env_ids]
        done_is_teacher = self._dagger_env_is_teacher[env_ids]
        if torch.any(done_is_teacher):
            self._dagger_teacher_episode_rewards.update(done_rewards[done_is_teacher])
        if torch.any(~done_is_teacher):
            self._dagger_student_episode_rewards.update(done_rewards[~done_is_teacher])

    def _reset_rollout_reward_stats(self):
        self._dagger_rollout_reward_stats = {
            'mixed_sum': 0.0,
            'mixed_count': 0.0,
            'teacher_sum': 0.0,
            'teacher_count': 0.0,
            'student_sum': 0.0,
            'student_count': 0.0,
            'teacher_ratio_sum': 0.0,
            'teacher_ratio_count': 0.0,
        }

    def _update_rollout_reward_stats(self, rewards, expert_mask):
        if self.distillation_mode != 'dagger_rl_studentact':
            return
        if self._dagger_rollout_reward_stats is None:
            self._reset_rollout_reward_stats()

        reward_values = rewards.detach()
        if reward_values.dim() > 1:
            reward_values = reward_values.mean(dim=-1)
        reward_values = reward_values.view(-1)
        teacher_mask = expert_mask.detach().to(reward_values.device).float().view(-1)
        if teacher_mask.numel() != reward_values.numel():
            teacher_mask = teacher_mask[:reward_values.numel()]
        student_mask = 1.0 - teacher_mask

        stats = self._dagger_rollout_reward_stats
        stats['mixed_sum'] += reward_values.sum().item()
        stats['mixed_count'] += float(reward_values.numel())
        teacher_count = teacher_mask.sum().item()
        student_count = student_mask.sum().item()
        if teacher_count > 0:
            stats['teacher_sum'] += (reward_values * teacher_mask).sum().item()
            stats['teacher_count'] += teacher_count
        if student_count > 0:
            stats['student_sum'] += (reward_values * student_mask).sum().item()
            stats['student_count'] += student_count
        stats['teacher_ratio_sum'] += teacher_mask.mean().item()
        stats['teacher_ratio_count'] += 1.0

    def _get_rollout_reward_means(self):
        stats = self._dagger_rollout_reward_stats
        if not stats:
            return {}

        def safe_mean(sum_key, count_key):
            count = stats[count_key]
            if count <= 0:
                return None
            return stats[sum_key] / count

        result = {
            'mixed': safe_mean('mixed_sum', 'mixed_count'),
            'teacher_step': safe_mean('teacher_sum', 'teacher_count'),
            'student_step': safe_mean('student_sum', 'student_count'),
            'teacher_ratio': safe_mean('teacher_ratio_sum', 'teacher_ratio_count'),
            'student_ratio': None,
        }
        if result['teacher_ratio'] is not None:
            result['student_ratio'] = 1.0 - result['teacher_ratio']
        teacher_reward = self._pcg_get_meter_mean(self._dagger_teacher_episode_rewards)
        student_reward = self._pcg_get_meter_mean(self._dagger_student_episode_rewards)
        if teacher_reward is not None:
            result['teacher_reward'] = teacher_reward
        if student_reward is not None:
            result['student_reward'] = student_reward
        return {k: v for k, v in result.items() if v is not None}

    def _format_reward_breakdown(self):
        base = super()._format_reward_breakdown()
        if self.distillation_mode == 'pcg_studentact':
            parts = [
                f"teacher_ratio:{float(self._pcg_teacher_env_ratio):.3f}",
                f"student_ratio:{1.0 - float(self._pcg_teacher_env_ratio):.3f}",
            ]
            teacher_reward = self._pcg_get_meter_mean(self._pcg_teacher_episode_rewards)
            student_reward = self._pcg_get_meter_mean(self._pcg_student_episode_rewards)
            if teacher_reward is not None:
                parts.append(f"teacher_reward:{teacher_reward:.3f}")
            if student_reward is not None:
                parts.append(f"student_reward:{student_reward:.3f}")
            if self._pcg_teacher_rew_ema is not None:
                parts.append(f"teacher_ema:{float(self._pcg_teacher_rew_ema):.3f}")
            if self._pcg_student_rew_ema is not None:
                parts.append(f"student_ema:{float(self._pcg_student_rew_ema):.3f}")
            parts.append(f"rl_coef:{self._pcg_rl_coef():.3f}")
            parts.append(f"sup_coef:{float(self._pcg_sup_coef):.3f}")
            pcg_line = "pcg " + ", ".join(parts)
            if base:
                return f"{base} | {pcg_line}"
            return pcg_line
        if self.distillation_mode != 'dagger_rl_studentact':
            return base
        rollout = self._get_rollout_reward_means()
        if not rollout:
            return base
        parts = []
        for key in ['mixed', 'teacher_reward', 'student_reward', 'teacher_step', 'student_step', 'teacher_ratio', 'student_ratio']:
            value = rollout.get(key, None)
            if value is not None and np.isfinite(value):
                parts.append(f"{key}:{value:.3f}")
        rollout_line = "rollout_reward " + ", ".join(parts)
        if base:
            return f"{base} | {rollout_line}"
        return rollout_line

    def _init_pcg_config(self, config):
        self.pcg_low_perf_threshold = float(config.get('pcg_low_perf_threshold', 8.0))
        self.pcg_high_perf_threshold = float(config.get('pcg_high_perf_threshold', 35.0))
        self.pcg_perf_ema = float(config.get('pcg_perf_ema', 0.05))
        self.pcg_blend_alpha_max = float(config.get('pcg_blend_alpha_max', 0.7))
        self.pcg_blend_alpha_min = float(config.get('pcg_blend_alpha_min', 0.0))
        self.pcg_sup_min_coef = float(config.get('pcg_sup_min_coef', 0.1))
        self.pcg_sup_max_coef = float(config.get('pcg_sup_max_coef', 1.0))
        self.pcg_rl_warmup_epochs = int(config.get('pcg_rl_warmup_epochs', 500))
        self.pcg_rl_ramp_epochs = int(config.get('pcg_rl_ramp_epochs', 3000))
        self.pcg_rl_coef_final = float(config.get('pcg_rl_coef_final', 1.0))
        self.pcg_anneal_sup = bool(config.get('pcg_anneal_sup', True))
        self._pcg_perf = None
        self.pcg_use_relative_reward = bool(config.get('pcg_use_relative_reward', False))
        self.pcg_reward_ema = float(config.get('pcg_reward_ema', self.pcg_perf_ema))
        self.pcg_ratio_ema = float(config.get('pcg_ratio_ema', 0.05))
        self.pcg_perf_eps = float(config.get('pcg_perf_eps', 1e-6))
        self.pcg_perf_target = float(config.get('pcg_perf_target', 1.0))
        self.pcg_teacher_ratio_min = float(config.get('pcg_teacher_ratio_min', 0.15))
        self.pcg_teacher_ratio_max = float(config.get('pcg_teacher_ratio_max', 0.65))
        self.pcg_student_ratio_min = float(config.get('pcg_student_ratio_min', 1.0 - self.pcg_teacher_ratio_max))
        self.pcg_student_ratio_max = float(config.get('pcg_student_ratio_max', 1.0 - self.pcg_teacher_ratio_min))
        self.pcg_teacher_ratio_min = float(np.clip(self.pcg_teacher_ratio_min, 0.0, 1.0))
        self.pcg_teacher_ratio_max = float(np.clip(self.pcg_teacher_ratio_max, 0.0, 1.0))
        self.pcg_student_ratio_min = float(np.clip(self.pcg_student_ratio_min, 0.0, 1.0))
        self.pcg_student_ratio_max = float(np.clip(self.pcg_student_ratio_max, 0.0, 1.0))
        teacher_min_from_student = 1.0 - self.pcg_student_ratio_max
        teacher_max_from_student = 1.0 - self.pcg_student_ratio_min
        self.pcg_teacher_ratio_min = max(self.pcg_teacher_ratio_min, teacher_min_from_student)
        self.pcg_teacher_ratio_max = min(self.pcg_teacher_ratio_max, teacher_max_from_student)
        if self.pcg_teacher_ratio_min > self.pcg_teacher_ratio_max:
            mid = float(np.clip(0.5 * (self.pcg_teacher_ratio_min + self.pcg_teacher_ratio_max), 0.0, 1.0))
            self.pcg_teacher_ratio_min = mid
            self.pcg_teacher_ratio_max = mid
        self._pcg_teacher_env_ratio = self.pcg_teacher_ratio_max
        self._pcg_sup_coef = self.pcg_sup_max_coef
        self._pcg_teacher_rew_ema = None
        self._pcg_student_rew_ema = None
        self._pcg_env_is_teacher = None

    def _pcg_init_state(self):
        num_envs = self.vec_env.env.task.num_envs
        if self._pcg_perf is None or self._pcg_perf.shape[0] != num_envs:
            init_perf = 0.5 * (self.pcg_low_perf_threshold + self.pcg_high_perf_threshold)
            self._pcg_perf = torch.ones(num_envs, device=self.ppo_device, dtype=torch.float32) * init_perf
        if self.pcg_use_relative_reward:
            self._pcg_init_env_roles()

    def _pcg_init_env_roles(self):
        num_envs = self.vec_env.env.task.num_envs
        if self._pcg_env_is_teacher is None or self._pcg_env_is_teacher.shape[0] != num_envs:
            self._pcg_env_is_teacher = self._pcg_make_role_mask(num_envs)

    def _pcg_make_role_mask(self, count):
        count = int(count)
        mask = torch.zeros(count, dtype=torch.bool, device=self.ppo_device)
        if count <= 0:
            return mask
        ratio = float(np.clip(self._pcg_teacher_env_ratio, self.pcg_teacher_ratio_min, self.pcg_teacher_ratio_max))
        teacher_count = int(round(count * ratio))
        teacher_count = min(max(teacher_count, 0), count)
        if teacher_count > 0:
            mask[torch.randperm(count, device=self.ppo_device)[:teacher_count]] = True
        return mask

    def _pcg_resample_env_roles(self, done_indices):
        if not self.pcg_use_relative_reward:
            return
        self._pcg_init_env_roles()
        if done_indices is None:
            return
        if isinstance(done_indices, torch.Tensor):
            if done_indices.numel() == 0:
                return
            env_ids = done_indices.view(-1).to(self.ppo_device).long()
        else:
            env_ids = to_torch(done_indices, device=self.ppo_device, dtype=torch.long).view(-1)
            if env_ids.numel() == 0:
                return
        self._pcg_env_is_teacher[env_ids] = self._pcg_make_role_mask(env_ids.numel())

    def _pcg_normalized_perf(self):
        self._pcg_init_state()
        denom = max(self.pcg_high_perf_threshold - self.pcg_low_perf_threshold, 1e-6)
        return torch.clamp((self._pcg_perf - self.pcg_low_perf_threshold) / denom, 0.0, 1.0)

    def _pcg_studentact_guidance(self, res_dict, expert_actions):
        if self.pcg_use_relative_reward:
            self._pcg_init_env_roles()
            teacher_mask = self._pcg_env_is_teacher.detach()
            student_mask = ~teacher_mask
            res_dict['actions'][teacher_mask] = expert_actions[teacher_mask]
            res_dict['rand_action_mask'] = student_mask.float()
            sup_coef = torch.full(
                (teacher_mask.shape[0],),
                float(self._pcg_sup_coef),
                dtype=torch.float32,
                device=self.ppo_device,
            )
            teacher_mask_f = teacher_mask.float()
            return {
                'expert_mask': teacher_mask_f,
                'pcg_blend_alpha': teacher_mask_f,
                'pcg_sup_coef': sup_coef.detach(),
            }
        perf_norm = self._pcg_normalized_perf()
        low_perf = 1.0 - perf_norm
        alpha = self.pcg_blend_alpha_min + low_perf * (self.pcg_blend_alpha_max - self.pcg_blend_alpha_min)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        env_actions = (1.0 - alpha.unsqueeze(-1)) * res_dict['actions'] + alpha.unsqueeze(-1) * expert_actions
        sup_coef = self.pcg_sup_min_coef + low_perf * (self.pcg_sup_max_coef - self.pcg_sup_min_coef)
        sup_coef = torch.clamp(sup_coef, min=0.0)
        res_dict['actions'] = torch.clamp(env_actions, -1.0, 1.0)
        return {
            'expert_mask': alpha.detach(),
            'pcg_blend_alpha': alpha.detach(),
            'pcg_sup_coef': sup_coef.detach(),
        }

    def _pcg_update_done_performance(self, done_indices):
        if self.pcg_use_relative_reward:
            return
        self._pcg_init_state()
        if done_indices is None:
            return
        if isinstance(done_indices, torch.Tensor):
            if done_indices.numel() == 0:
                return
            env_ids = done_indices.view(-1).to(self.ppo_device).long()
        else:
            env_ids = to_torch(done_indices, device=self.ppo_device, dtype=torch.long).view(-1)
            if env_ids.numel() == 0:
                return
        done_rewards = self.current_rewards[env_ids]
        if done_rewards.dim() > 1:
            done_rewards = done_rewards.mean(dim=-1)
        old = self._pcg_perf[env_ids]
        self._pcg_perf[env_ids] = (1.0 - self.pcg_perf_ema) * old + self.pcg_perf_ema * done_rewards.detach()

    def _pcg_update_relative_performance(self, shaped_rewards, teacher_mask):
        if not self.pcg_use_relative_reward:
            return
        reward_values = shaped_rewards.detach()
        if reward_values.dim() > 1:
            reward_values = reward_values.mean(dim=-1)
        reward_values = reward_values.view(-1)
        teacher_mask = teacher_mask.detach().to(reward_values.device).float().view(-1)
        if teacher_mask.numel() != reward_values.numel():
            teacher_mask = teacher_mask[:reward_values.numel()]
        student_mask = 1.0 - teacher_mask
        teacher_count = teacher_mask.sum()
        student_count = student_mask.sum()
        if teacher_count > 0:
            teacher_mean = (reward_values * teacher_mask).sum() / (teacher_count + self.pcg_perf_eps)
            teacher_mean = float(teacher_mean.item())
            if self._pcg_teacher_rew_ema is None:
                self._pcg_teacher_rew_ema = teacher_mean
            else:
                self._pcg_teacher_rew_ema = (1.0 - self.pcg_reward_ema) * self._pcg_teacher_rew_ema + self.pcg_reward_ema * teacher_mean
        if student_count > 0:
            student_mean = (reward_values * student_mask).sum() / (student_count + self.pcg_perf_eps)
            student_mean = float(student_mean.item())
            if self._pcg_student_rew_ema is None:
                self._pcg_student_rew_ema = student_mean
            else:
                self._pcg_student_rew_ema = (1.0 - self.pcg_reward_ema) * self._pcg_student_rew_ema + self.pcg_reward_ema * student_mean
        self._pcg_update_relative_ratios()

    def _pcg_update_relative_ratios(self):
        if self._pcg_teacher_rew_ema is None or self._pcg_student_rew_ema is None:
            return
        teacher_reward = float(self._pcg_teacher_rew_ema)
        student_reward = float(self._pcg_student_rew_ema)
        relative_perf = student_reward / (teacher_reward + self.pcg_perf_eps)
        progress = float(np.clip(relative_perf / max(self.pcg_perf_target, self.pcg_perf_eps), 0.0, 1.0))
        target_teacher_ratio = self.pcg_teacher_ratio_max - progress * (self.pcg_teacher_ratio_max - self.pcg_teacher_ratio_min)
        self._pcg_teacher_env_ratio = (
            (1.0 - self.pcg_ratio_ema) * float(self._pcg_teacher_env_ratio)
            + self.pcg_ratio_ema * float(target_teacher_ratio)
        )
        self._pcg_teacher_env_ratio = float(np.clip(
            self._pcg_teacher_env_ratio,
            self.pcg_teacher_ratio_min,
            self.pcg_teacher_ratio_max,
        ))
        if self.pcg_anneal_sup:
            target_sup = self.pcg_sup_max_coef - progress * (self.pcg_sup_max_coef - self.pcg_sup_min_coef)
            self._pcg_sup_coef = (
                (1.0 - self.pcg_ratio_ema) * float(self._pcg_sup_coef)
                + self.pcg_ratio_ema * float(target_sup)
            )
            self._pcg_sup_coef = float(np.clip(self._pcg_sup_coef, self.pcg_sup_min_coef, self.pcg_sup_max_coef))
        else:
            self._pcg_sup_coef = float(self.pcg_sup_max_coef)

    def _pcg_rl_coef(self):
        if self.distillation_mode != 'pcg_studentact':
            return 1.0
        epoch = int(getattr(self, 'epoch_num', 0))
        if epoch <= self.pcg_rl_warmup_epochs:
            return 0.0
        denom = max(self.pcg_rl_ramp_epochs, 1)
        frac = min(max((epoch - self.pcg_rl_warmup_epochs) / denom, 0.0), 1.0)
        return self.pcg_rl_coef_final * frac

    def _pcg_init_episode_reward_meters(self):
        if self._pcg_teacher_episode_rewards is None:
            self._pcg_teacher_episode_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        if self._pcg_student_episode_rewards is None:
            self._pcg_student_episode_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)

    def _pcg_update_episode_reward_meters(self, done_indices):
        if self.distillation_mode != 'pcg_studentact' or not self.pcg_use_relative_reward:
            return
        if self._pcg_env_is_teacher is None or done_indices is None:
            return
        self._pcg_init_episode_reward_meters()
        if isinstance(done_indices, torch.Tensor):
            if done_indices.numel() == 0:
                return
            env_ids = done_indices.view(-1).to(self.ppo_device).long()
        else:
            env_ids = to_torch(done_indices, device=self.ppo_device, dtype=torch.long).view(-1)
            if env_ids.numel() == 0:
                return
        done_rewards = self.current_rewards[env_ids]
        done_is_teacher = self._pcg_env_is_teacher[env_ids]
        if torch.any(done_is_teacher):
            self._pcg_teacher_episode_rewards.update(done_rewards[done_is_teacher])
        if torch.any(~done_is_teacher):
            self._pcg_student_episode_rewards.update(done_rewards[~done_is_teacher])

    def _pcg_get_meter_mean(self, meter):
        if meter is None or meter.current_size <= 0:
            return None
        return torch.as_tensor(meter.get_mean()).float().mean().item()


    def init_tensors(self):
        super().init_tensors()
        if self.distillation_mode == 'pcg_studentact':
            self._pcg_init_episode_reward_meters()
        elif self.distillation_mode == 'dagger_rl_studentact':
            self._dagger_init_episode_reward_meters()
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['expert_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['expert'] = torch.zeros((*batch_shape, self.actions_num), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['pcg_sup_coef'] = torch.ones(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['pcg_blend_alpha'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.tensor_list += ['amp_obs', 'rand_action_mask', 'expert', 'expert_mask', 'pcg_sup_coef', 'pcg_blend_alpha']
        self._init_extra_distill_tensors(batch_shape)
        return

    def _init_extra_distill_tensors(self, batch_shape):
        return


    def play_steps(self):
        self.set_eval()
        self._reset_rollout_reward_stats()

        epinfos = []
        update_list = self.update_list

        # Initialize DAgger beta coefficient based on distillation mode
        if self.distillation_mode == 'dagger_only':
            # Always use teacher actions (pure DAgger)
            beta_t = 1.0
        elif self.distillation_mode in ['dagger_studentact', 'pcg_studentact', 'rl_only']:
            # Never replace rollout actions with teacher actions.
            # dagger_studentact still uses expert supervision in the update.
            beta_t = 0.0
        elif self.distillation_mode == 'dagger_rl_studentact':
            beta_t = self._get_dagger_studentact_teacher_ratio()
        else:  # 'dagger_rl' - progressive DAgger to RL
            beta_t = max(1 - max((self.epoch_num - 500) / 5000, 0), 0)

        for n in range(self.horizon_length):

            self.obs, self.expert = self.env_reset(self.done_indices)
            if self.distillation_mode == 'pcg_studentact':
                self._pcg_resample_env_roles(self.done_indices)
            elif self.distillation_mode == 'dagger_rl_studentact':
                self._dagger_resample_env_roles(self.done_indices)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self._record_extra_distill_step(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            elif self.distillation_mode == 'pcg_studentact':
                res_dict = super().get_action_values(self.obs, self._rand_action_probs)
                pcg_info = self._pcg_studentact_guidance(res_dict, self.expert['actions'].to(self.ppo_device))
                res_dict.update(pcg_info)
                self._pcg_set_task_student_rollout_mask(res_dict['expert_mask'])
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs, beta_t, self.expert['actions'].to(self.ppo_device))
            self.experience_buffer.update_data('expert', n, self.expert['mus'].to(self.ppo_device))
            if self.distillation_mode != 'pcg_studentact':
                res_dict['pcg_sup_coef'] = torch.ones_like(res_dict['rand_action_mask'])
                res_dict['pcg_blend_alpha'] = torch.zeros_like(res_dict['rand_action_mask'])

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos, self.expert = self.env_step(res_dict['actions'])
            self._update_rollout_reward_stats(rewards, res_dict['expert_mask'])
            shaped_rewards = self.rewards_shaper(rewards)
            if self.distillation_mode == 'pcg_studentact':
                self._pcg_update_relative_performance(shaped_rewards, res_dict['expert_mask'])
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])
            self.experience_buffer.update_data('pcg_sup_coef', n, res_dict['pcg_sup_coef'])
            self.experience_buffer.update_data('pcg_blend_alpha', n, res_dict['pcg_blend_alpha'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            self.done_indices = all_done_indices[::self.num_agents]
            if self.distillation_mode == 'pcg_studentact':
                self._pcg_update_episode_reward_meters(self.done_indices)
                self._pcg_update_done_performance(self.done_indices)
            elif self.distillation_mode == 'dagger_rl_studentact':
                self._dagger_update_episode_reward_meters(self.done_indices)
  
            self.game_rewards.update(self.current_rewards[self.done_indices])
            self.game_lengths.update(self.current_lengths[self.done_indices])
            self.algo_observer.process_infos(infos, self.done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)
                
            self.done_indices = self.done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict


    def get_action_values(self, obs_dict, rand_action_probs, use_experts=0.0, expert=None):
        res_dict = super().get_action_values(obs_dict, rand_action_probs)
        num_envs = self.vec_env.env.task.num_envs
        if self.distillation_mode == 'dagger_rl_studentact':
            self._dagger_init_env_roles()
            expert_action_probs = self._dagger_env_is_teacher.to(self.ppo_device).float()
        else:
            expert_action_probs = self._make_expert_action_mask(num_envs, use_experts)
        det_action_mask = expert_action_probs == 1.0
        res_dict['actions'][det_action_mask] = expert[det_action_mask]
        res_dict['expert_mask'] = expert_action_probs

        return res_dict

    def _pcg_set_task_student_rollout_mask(self, expert_mask):
        try:
            task = self.vec_env.env.task
        except AttributeError:
            return
        student_mask = (expert_mask.detach().view(-1).to(self.ppo_device) < 0.5)
        task._curr_is_student_rollout = student_mask.to(task.device)

    def _make_expert_action_mask(self, num_envs, use_experts):
        use_experts = float(use_experts)
        if self.distillation_mode == 'dagger_rl_studentact' and 0.0 < use_experts < 1.0:
            expert_count = int(round(num_envs * use_experts))
            expert_count = min(max(expert_count, 0), num_envs)
            mask = torch.zeros(num_envs, dtype=torch.float32, device=self.ppo_device)
            if expert_count > 0:
                expert_ids = torch.randperm(num_envs, device=self.ppo_device)[:expert_count]
                mask[expert_ids] = 1.0
            return mask
        expert_action_probs = to_torch([use_experts for _ in range(num_envs)], dtype=torch.float32, device=self.ppo_device)
        return torch.bernoulli(expert_action_probs)
    

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        expert = batch_dict['expert']
        expert_mask = batch_dict['expert_mask']
        self.dataset.values_dict['expert'] = expert
        self.dataset.values_dict['expert_mask'] = expert_mask
        self.dataset.values_dict['pcg_sup_coef'] = batch_dict.get('pcg_sup_coef', torch.ones_like(expert_mask))
        self.dataset.values_dict['pcg_blend_alpha'] = batch_dict.get('pcg_blend_alpha', torch.zeros_like(expert_mask))
        self._prepare_extra_distill_dataset(batch_dict)
        return

    def _prepare_extra_distill_dataset(self, batch_dict):
        return


    def _supervise_loss(self, student, teacher):
        e_loss = (student - teacher)**2

        info = {
            'expert_loss': e_loss.sum(dim=-1)
        }
        return info
    
    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, expert = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos, expert #.to(self.ppo_device)
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos, expert #.to(self.ppo_device)
        
    def env_reset(self, env_ids=None):
        obs, expert = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs, expert


    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        expert_mus = input_dict['expert']
        pcg_sup_coef = input_dict.get('pcg_sup_coef', None)
        pcg_blend_alpha = input_dict.get('pcg_blend_alpha', None)
        if pcg_sup_coef is None:
            pcg_sup_coef = torch.ones(obs_batch.shape[0], device=self.ppo_device, dtype=torch.float32)
        else:
            pcg_sup_coef = pcg_sup_coef.view(-1).to(self.ppo_device)
        if pcg_blend_alpha is None:
            pcg_blend_alpha = torch.zeros_like(pcg_sup_coef)
        else:
            pcg_blend_alpha = pcg_blend_alpha.view(-1).to(self.ppo_device)
        obs_batch = self._preproc_obs(obs_batch)

        expert_mask = (input_dict['expert_mask'] > -1).float()
        expert_sum = torch.sum(expert_mask)

        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.amp.autocast('cuda', enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            if rand_action_sum > 0:
                a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
                a_loss = a_info['actor_loss']
                a_clipped = a_info['actor_clipped'].float()
                c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                c_loss = c_info['critic_loss']

                if self.epoch_num > 7000:
                    returns_var = return_batch.var(unbiased=False) + 1e-8  # avoid divide‑by‑0
                    errors_var = (return_batch - values).var(unbiased=False)
                    ev = 1.0 - errors_var / returns_var
                    self.ev_ma = 0.99 * self.ev_ma + 0.01 * ev.item()
                    if self.ev_ma >= 0.6:
                        self.critic_win_streak += 1
                    else:
                        self.critic_win_streak = 0
                        
                b_loss = self.bound_loss(mu)
                
                c_loss = torch.mean(c_loss)
                e_info = self._supervise_loss(mu, expert_mus)
                e_loss_raw = e_info['expert_loss']
                a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
                entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
                b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
                a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum
                e_loss = torch.mean(pcg_sup_coef * e_loss_raw)
                latent_loss = self._compute_extra_distill_loss(mu, expert_mus, input_dict)
                
                # Compute loss based on distillation mode
                if self.distillation_mode == 'rl_only':
                    # Pure RL: always use actor and critic loss
                    if self.epoch_num > 6000 and self.critic_win_streak >= 3:
                        loss = a_loss * min((self.actor_update_num / 4000), 1) + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss
                        self.actor_update_num += 1
                    elif self.epoch_num > 5000:
                        loss = min(((self.epoch_num - 5000) / 1000), 1) * self.critic_coef * c_loss
                    else:
                        # Early stage: still use critic loss
                        loss = min(((self.epoch_num - 5000) / 1000), 1) * self.critic_coef * c_loss if self.epoch_num > 5000 else self.critic_coef * c_loss
                elif self.distillation_mode == 'dagger_only':
                    # Pure DAgger: only use expert loss (even if there are student actions)
                    loss = self.expert_loss_coef * e_loss
                elif self.distillation_mode == 'pcg_studentact':
                    rl_coef = self._pcg_rl_coef()
                    rl_loss = a_loss + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss
                    loss = self.expert_loss_coef * e_loss + rl_coef * rl_loss
                    e_info['pcg_rl_coef'] = torch.tensor(rl_coef, device=self.ppo_device)
                    e_info['pcg_sup_coef'] = torch.mean(pcg_sup_coef)
                    e_info['pcg_blend_alpha'] = torch.mean(pcg_blend_alpha)
                    e_info['pcg_teacher_env_ratio'] = torch.tensor(float(self._pcg_teacher_env_ratio), device=self.ppo_device)
                    e_info['pcg_student_env_ratio'] = torch.tensor(1.0 - float(self._pcg_teacher_env_ratio), device=self.ppo_device)
                    if self._pcg_teacher_rew_ema is not None:
                        e_info['pcg_teacher_rew_ema'] = torch.tensor(float(self._pcg_teacher_rew_ema), device=self.ppo_device)
                    if self._pcg_student_rew_ema is not None:
                        e_info['pcg_student_rew_ema'] = torch.tensor(float(self._pcg_student_rew_ema), device=self.ppo_device)
                elif self.distillation_mode == 'dagger_rl_studentact':
                    if self.epoch_num >= self.dagger_studentact_rl_start_epoch:
                        if self.epoch_num > 6000 and self.critic_win_streak >= 3:
                            loss = a_loss * min((self.actor_update_num / 4000), 1) + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss
                            self.actor_update_num += 1
                        elif self.epoch_num > 5000:
                            loss = min(((self.epoch_num - 5000) / 1000), 1) * self.critic_coef * c_loss
                        else:
                            loss = self.critic_coef * c_loss
                    else:
                        loss = self.expert_loss_coef * e_loss
                else:  # 'dagger_rl' - progressive training
                    if self.epoch_num > 6000 and self.critic_win_streak >= 3:
                        loss = a_loss * min((self.actor_update_num / 4000), 1) + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss + self.expert_loss_coef * e_loss * max(1 - (self.actor_update_num / 4000), 0.1)
                        self.actor_update_num += 1
                    elif self.epoch_num > 5000:
                        loss = min(((self.epoch_num - 5000) / 1000), 1) * self.critic_coef * c_loss + self.expert_loss_coef * e_loss
                    else:
                        loss = self.expert_loss_coef * e_loss
                if latent_loss is not None:
                    loss = loss + latent_loss
                    e_info['latent_loss'] = latent_loss.detach()
                
            else:
                # All actions are teacher actions (or deterministic student actions)
                if self.distillation_mode == 'rl_only':
                    # In RL-only mode, this shouldn't happen, but handle gracefully
                    loss = torch.tensor(0.0, device=self.ppo_device, requires_grad=True)
                    e_info = {'expert_loss': torch.tensor(0.0, device=self.ppo_device)}
                else:
                    e_info = self._supervise_loss(mu, expert_mus)
                    e_loss = e_info['expert_loss']
                    e_loss = torch.mean(e_loss)
                    loss = self.expert_loss_coef * e_loss
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(e_info)
        return

    def _record_extra_distill_step(self, step_idx):
        return

    def _compute_extra_distill_loss(self, student_actions, expert_actions, input_dict):
        return None

    def _get_best_reward_metric(self, mean_rewards):
        if self.distillation_mode == 'pcg_studentact':
            student_reward = self._pcg_get_meter_mean(self._pcg_student_episode_rewards)
            return student_reward
        if self.distillation_mode == 'dagger_rl_studentact':
            student_reward = self._pcg_get_meter_mean(self._dagger_student_episode_rewards)
            if student_reward is not None:
                return student_reward
            rollout = self._get_rollout_reward_means()
            return rollout.get('student_step', None)
        return super()._get_best_reward_metric(mean_rewards)

    def _get_best_reward_metric_name(self):
        if self.distillation_mode == 'pcg_studentact':
            return 'student_reward'
        if self.distillation_mode == 'dagger_rl_studentact':
            return 'student_reward'
        return super()._get_best_reward_metric_name()

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        self.writer.add_scalar('losses/e_loss', torch_ext.mean_list(train_info['expert_loss']).item(), frame)
        rollout = self._get_rollout_reward_means()
        for key, value in rollout.items():
            self.writer.add_scalar(f'dagger_rl_studentact/{key}', value, frame)
        if 'pcg_rl_coef' in train_info:
            self.writer.add_scalar('pcg/rl_coef', torch_ext.mean_list(train_info['pcg_rl_coef']).item(), frame)
        if 'pcg_sup_coef' in train_info:
            self.writer.add_scalar('pcg/sup_coef', torch_ext.mean_list(train_info['pcg_sup_coef']).item(), frame)
        if 'pcg_blend_alpha' in train_info:
            self.writer.add_scalar('pcg/blend_alpha', torch_ext.mean_list(train_info['pcg_blend_alpha']).item(), frame)
        if 'pcg_teacher_env_ratio' in train_info:
            self.writer.add_scalar('pcg/teacher_env_ratio', torch_ext.mean_list(train_info['pcg_teacher_env_ratio']).item(), frame)
        if 'pcg_student_env_ratio' in train_info:
            self.writer.add_scalar('pcg/student_env_ratio', torch_ext.mean_list(train_info['pcg_student_env_ratio']).item(), frame)
        if 'pcg_teacher_rew_ema' in train_info:
            self.writer.add_scalar('pcg/teacher_rew_ema', torch_ext.mean_list(train_info['pcg_teacher_rew_ema']).item(), frame)
        if 'pcg_student_rew_ema' in train_info:
            self.writer.add_scalar('pcg/student_rew_ema', torch_ext.mean_list(train_info['pcg_student_rew_ema']).item(), frame)
        teacher_reward = self._pcg_get_meter_mean(self._pcg_teacher_episode_rewards)
        student_reward = self._pcg_get_meter_mean(self._pcg_student_episode_rewards)
        if teacher_reward is not None:
            self.writer.add_scalar('pcg/teacher_episode_reward', teacher_reward, frame)
        if student_reward is not None:
            self.writer.add_scalar('pcg/student_episode_reward', student_reward, frame)
        if 'latent_loss' in train_info:
            self.writer.add_scalar('losses/latent_loss', torch_ext.mean_list(train_info['latent_loss']).item(), frame)

        return