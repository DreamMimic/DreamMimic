import torch
import numpy as np

from env.tasks.intermimic_all import InterMimic_All
from env.tasks.intermimic import InterMimic


class InterMimic_All_NoRef(InterMimic_All):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Keep both teacher and student observation sizes distinct
        student_obs_size = cfg["env"]["numObs"]
        teacher_obs_size = cfg["env"].get("teacherObsSize", student_obs_size)

        # Temporarily switch to teacher obs size to build and load teacher models in parent init
        cfg["env"]["numObs"] = teacher_obs_size
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        # Restore student obs size for RL policy (student network)
        self.cfg["env"]["numObs"] = student_obs_size
        self._num_obs = student_obs_size

        # Allocate distinct buffers: teacher (full obs) and student (no-ref)
        self.teacher_obs_size = teacher_obs_size
        self.obs_buf_teacher = torch.zeros((self.num_envs, self.teacher_obs_size), device=self.device, dtype=torch.float)
        self.student_obs_buf = torch.zeros((self.num_envs, student_obs_size), device=self.device, dtype=torch.float)

        # Expose student buffer as the main obs buffer for RL
        self.obs_buf = self.student_obs_buf

    def _compute_observations_iter_noref(self, env_ids=None, delta_t=1):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        ts = self.progress_buf[env_ids].clone()
        next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]] - 1)

        # Build a zero reference observation to remove reference-frame information
        ref_obs_zeros = torch.zeros_like(self.hoi_data[self.data_id[env_ids], next_ts])

        # Reuse the same humanoid/task obs builders but with zero ref
        obs = self._compute_humanoid_obs(env_ids, ref_obs_zeros, next_ts)
        task_obs = self._compute_task_obs(env_ids, ref_obs_zeros)
        obs = torch.cat([obs, task_obs], dim=-1)

        # IG terms: build with zero ref to eliminate ref contribution
        # ig_all uses current obs only; ref_ig - ig becomes -ig when ref is zero
        ig_all, ig, ref_ig = self._compute_ig_obs(env_ids, ref_obs_zeros)
        return torch.cat((obs, ig_all, ref_ig - ig), dim=-1)

    def _compute_observations(self, env_ids=None):
        # First compute full obs (teacher) via parent implementation into teacher buffer
        # Temporarily swap buffers so parent writes into teacher buffer
        original_obs_buf = self.obs_buf
        self.obs_buf = self.obs_buf_teacher
        super()._compute_observations(env_ids)
        # After parent computation, teacher obs are in self.obs_buf_teacher
        # Restore student buffer as active obs buffer
        self.obs_buf = original_obs_buf

        # Now compute no-ref student obs: use current-only observation (remove all reference terms)
        # We directly use the current HOI observation built in post_physics_step
        if env_ids is None:
            # Ensure current observation is up to date
            student_obs = self._curr_obs.clone()
            self.student_obs_buf[:] = student_obs
            # Replace obs_buf with student obs (no-ref) for student policy
            self.obs_buf = self.student_obs_buf
        else:
            student_obs = self._curr_obs[env_ids].clone()
            self.student_obs_buf[env_ids] = student_obs
            # Replace obs_buf with student obs (no-ref) for student policy
            self.obs_buf[env_ids] = self.student_obs_buf[env_ids]
        return

    def step(self, weights):
        # Same as parent, but feed teacher models with full-observation buffer
        self.pre_physics_step(weights)
        self._physics_step()
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        self.post_physics_step()

        with torch.no_grad():
            batched_forward = self.vmap(self.single_model_forward, in_dims=(0, 0, 0, 0)) if hasattr(self, 'vmap') else None
            if batched_forward is None:
                from torch.func import vmap
                batched_forward = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            # Teacher uses full observations (obs_buf_teacher) - 3198 dim
            mus_all, sigma_all = batched_forward(
                self.stacked_params,
                self.obs_buf_teacher.unsqueeze(0).repeat(self.running_means_all.shape[0], 1, 1),
                self.running_means_all,
                self.running_vars_all,
            )
            distr = torch.distributions.Normal(mus_all, sigma_all)
            selected_action = distr.sample()
            teacher_actions_all = torch.clamp(selected_action, min=-1.0, max=1.0)
            self.action_buf = teacher_actions_all[self.model_indices, self.sample_indices]
            self.mu_buf = mus_all[self.model_indices, self.sample_indices]

        if self.dr_randomizations.get('observations', None):
            # Apply noise to student observations (obs_buf) - 1211 dim
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def _compute_observations_retarget(self, env_ids=None):
        # Override parent to fix the env_ids=None indexing bug
        is_empty = False
        if env_ids is not None:
            try:
                is_empty = (env_ids.numel() == 0)
            except Exception:
                try:
                    is_empty = (len(env_ids) == 0)
                except Exception:
                    is_empty = False

        if (env_ids is None) or is_empty:
            # Fix: when env_ids is None, use slice notation instead of [None]
            self._curr_ref_obs[:] = self.hoi_data_retarget[self.data_id, self.progress_buf].clone()
            self.obs_buf_retarget[:] = torch.cat((self._compute_observations_iter(self.hoi_data_retarget, None, 1), self._compute_observations_iter(self.hoi_data_retarget, None, 16)), dim=-1)
        else:
            self._curr_ref_obs[env_ids] = self.hoi_data_retarget[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf_retarget[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data_retarget, env_ids, 1), self._compute_observations_iter(self.hoi_data_retarget, env_ids, 16)), dim=-1)
        return

    def reset(self, env_ids=None):
        # Bypass InterMimic_All.reset (which uses self.obs_buf) to avoid dim mismatch.
        # Call InterMimic.reset -> Humanoid.reset, which will invoke our _compute_observations
        # and fill both teacher and student buffers correctly.
        InterMimic.reset(self, env_ids=env_ids)
        # Align teacher model indices with dataset ids (same as parent, but we compute actions using teacher obs)
        id_to_index = {subid: i for i, subid in enumerate(self.models_subid)}
        self.model_indices = torch.tensor([id_to_index[subid.item()] for subid in self.dataset_id], 
                             dtype=torch.long, device=self.device)
        N = self.dataset_id.shape[0]
        self.sample_indices = torch.arange(N, device=self.device)
        # Ensure retarget obs is available for DAgger wrappers
        self._compute_observations_retarget(env_ids)
        # Compute teacher actions using full-observation buffer
        with torch.no_grad():
            from torch.func import vmap
            batched_forward = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            mus_all, sigma_all = batched_forward(
                self.stacked_params,
                self.obs_buf_teacher.unsqueeze(0).repeat(self.running_means_all.shape[0], 1, 1),
                self.running_means_all,
                self.running_vars_all,
            )
            distr = torch.distributions.Normal(mus_all, sigma_all)
            selected_action = distr.sample()
            teacher_actions_all = torch.clamp(selected_action, min=-1.0, max=1.0)
            self.action_buf = teacher_actions_all[self.model_indices, self.sample_indices]
            self.mu_buf = mus_all[self.model_indices, self.sample_indices]
        return


    def _compute_hoi_observations(self, env_ids=None):
        object_points = self.object_points[self.object_id[self.data_id]]
        # 下采样到 256 并确保连续，降低 compute_sdf 显存压力并避免 .view 报错
        object_points = object_points[:, :256, :].contiguous()
        self._curr_obs[:] = self.build_hoi_observations(
            self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
            self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:, 0, :],
            self._dof_pos, self._dof_vel, self._rigid_body_pos,
            self._local_root_obs, self._root_height_obs, self._dof_obs_size,
            self._target_states, self._tar_contact_forces, self._contact_forces,
            object_points, self._rigid_body_rot, self._rigid_body_vel, self._rigid_body_ang_vel
        )