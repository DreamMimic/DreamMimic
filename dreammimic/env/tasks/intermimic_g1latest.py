import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid_g1 import Humanoid_G1
from env.tasks.intermimic import InterMimic, compute_sdf


class InterMimicG1(Humanoid_G1, InterMimic):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.hoi_data = self._load_motion(self.motion_file, startk=1, initk=15)
        self.scaling = cfg['env']['scaling']
        self.init_root_height = cfg['env']['initRootHeight']
        
        # G1需要重新设置数据组件索引，因为观测空间布局不同
        self._setup_g1_data_components()
        self.init_dof = torch.cat([to_torch([-0.1, 0, 0.0, 0.3, -0.2, 0, -0.1, 0, 0.0, 0.3, -0.2, 0, 0, 0, 0, 
                                             0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float),
                                   to_torch([0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float)])
        # 重新对齐 G1 的状态缓存维度（SMPLX 默认是 332，G1 需按自身 DoF 调整）
        try:
            root_dim = self._humanoid_root_states.shape[1]  # 通常 13
            dof_dim = self._dof_pos.shape[1]               # G1: 43
            target_dim = self._target_states.shape[1] if hasattr(self, '_target_states') else 13
            g1_state_dim = root_dim + dof_dim + dof_dim + target_dim
            self._curr_state = torch.zeros((self.num_envs, self.rollout_length, g1_state_dim), device=self.device, dtype=torch.float)
        except Exception:
            pass
        return
    
    def _setup_g1_data_components(self):
        """设置G1特定的数据组件索引，对应build_hoi_observations的输出格式"""
        # G1的观测空间布局（来自build_hoi_observations）:
        # root_pos(3) + root_rot(4) + dof_pos_new(153) + dof_vel_new(153) + 
        # body_pos_new(52*3=156) + body_rot_new(52*4=208) + 
        # body_vel_new(52*3=156) + body_rot_vel_new(52*3=156) +
        # target_states(13) + ig_new(52*3=156) + contact_new(52) + target_contact(1)
        
        component_sizes = {
            'root_pos': 3,
            'root_rot': 4, 
            'dof_pos': 153,
            'dof_vel': 153,
            'body_pos': 156,  # 52*3
            'body_rot': 208,  # 52*4
            'body_pos_vel': 156,  # 52*3 (body_vel_new)
            'body_rot_vel': 156,  # 52*3 (body_rot_vel_new)
            'obj_pos': 3,     # 从target_states前3个
            'obj_rot': 4,     # 从target_states 3:7
            'obj_pos_vel': 3, # 从target_states 7:10
            'obj_rot_vel': 3, # 从target_states 10:13
            'ig': 156,        # 52*3 (ig_new)
            'contact_human': 52,  # contact_new
            'contact_obj': 1  # target_contact
        }
        
        # 重新计算累积索引
        self.data_component_order = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot', 'body_pos_vel', 'body_rot_vel',
            'obj_pos', 'obj_rot', 'obj_pos_vel', 'obj_rot_vel', 'ig', 'contact_human', 'contact_obj'
        ]
        
        # 计算每个组件在观测中的起始位置
        cumulative_size = 0
        self.data_component_index = [0]
        
        for component in self.data_component_order:
            if component in ['obj_pos', 'obj_rot', 'obj_pos_vel', 'obj_rot_vel']:
                # 这些组件来自target_states，需要特殊处理
                continue
            cumulative_size += component_sizes[component]
            self.data_component_index.append(cumulative_size)
            
        # 特殊处理target_states相关组件的索引
        target_start = 3 + 4 + 153 + 153 + 156 + 208 + 156 + 156  # 前面所有组件的总大小
        self.target_states_start = target_start
        
    def extract_data_component(self, var_name, ref=False, data_id=None, t=None, obs=None):
        """G1特定的数据组件提取，处理特殊的观测布局"""
        if var_name in ['obj_pos', 'obj_rot', 'obj_pos_vel', 'obj_rot_vel']:
            # 这些组件来自target_states
            if obs is not None:
                target_states = obs[..., self.target_states_start:self.target_states_start+13]
                if var_name == 'obj_pos':
                    return target_states[..., 0:3]
                elif var_name == 'obj_rot':
                    return target_states[..., 3:7]
                elif var_name == 'obj_pos_vel':
                    return target_states[..., 7:10]
                elif var_name == 'obj_rot_vel':
                    return target_states[..., 10:13]
        else:
            # 使用父类的方法处理其他组件
            return super().extract_data_component(var_name, ref, data_id, t, obs)

    def _compute_reward(self, actions):
        """
        Complete reward computation for G1.
        Adapted from InterMimic._compute_reward with G1-specific parameters.
        """
        rb, human_reset, key_pos, ref_key_pos = self.compute_humanoid_reward_g1(self.reward_weights)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward_g1(self.reward_weights)
        rig, ig_reset = self.compute_ig_reward_g1(self.reward_weights, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset = self.compute_cg_reward_g1(self.reward_weights)
        
        # Debug: 每100步打印一次reward组件
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 0
            
        if self._debug_step_count % 100 == 0:
            print(f"[DEBUG] Step {self._debug_step_count}")
            print(f"[DEBUG] rb (humanoid): mean={rb.mean():.6f}, max={rb.max():.6f}, min={rb.min():.6f}")
            print(f"[DEBUG] ro (object):   mean={ro.mean():.6f}, max={ro.max():.6f}, min={ro.min():.6f}")
            print(f"[DEBUG] rig (ig):      mean={rig.mean():.6f}, max={rig.max():.6f}, min={rig.min():.6f}")
            print(f"[DEBUG] rcg (contact): mean={rcg.mean():.6f}, max={rcg.max():.6f}, min={rcg.min():.6f}")
            final_reward = rb * ro * rig * rcg
            print(f"[DEBUG] Final reward:  mean={final_reward.mean():.6f}, max={final_reward.max():.6f}, min={final_reward.min():.6f}")
            print("---")
        
        # 安全检查：限制各个组件的范围，防止异常值
        rb = torch.clamp(rb, min=1e-8, max=10.0)
        ro = torch.clamp(ro, min=1e-8, max=10.0)  
        rig = torch.clamp(rig, min=1e-8, max=10.0)
        rcg = torch.clamp(rcg, min=1e-8, max=10.0)
        
        self.rew_buf[:] = rb * ro * rig * rcg
        kinematic_reset = torch.logical_or(human_reset, object_reset)
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        
        index = torch.arange(self._curr_reward.shape[0])
        self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
        self._sum_reward[index] += self.rew_buf
        self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
            self._humanoid_root_states,
            self._dof_pos,
            self._dof_vel,
            self._target_states,
        ], dim=1)
        return

    def _compute_reset(self):
        super()._compute_reset()


    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return


    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids, key_body_ids_gt, contact_body_ids_gt):
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        len_keypos = len(key_body_ids)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])
        
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids_gt, :]
        _body_pos = body_pos[:, key_body_ids, :]

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos[:, key_body_ids, :] - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = body_rot[:, key_body_ids, :].reshape(body_rot.shape[0] * len_keypos, body_rot.shape[2])
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], len_keypos * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs).view(ref_obs.shape[0], -1, 4)
        ref_body_rot_no_hand = ref_body_rot[:, key_body_ids_gt, :]
        body_rot_no_hand = body_rot[:, key_body_ids]

        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids_gt, :]
        _body_vel = body_vel[:, key_body_ids, :]
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 52, 3)[:, key_body_ids_gt]
        body_ang_vel_no_hand = body_ang_vel[:, key_body_ids]
        diff_global_ang_vel = ref_body_ang_vel_no_hand - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, len_keypos * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel[:, key_body_ids, :].reshape(body_vel.shape[0] * len_keypos, body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], len_keypos * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel[:, key_body_ids, :].reshape(body_ang_vel.shape[0] * len_keypos, body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], len_keypos * body_ang_vel.shape[2])

        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids_gt]
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

        obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
        return obs

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return   

    def _reset_target(self, env_ids):
        super()._reset_target(env_ids)
        self._target_states[env_ids, 0:2] = self._target_states[env_ids, 0:2] * self.scaling
        return


    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

        return    

    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos * self.scaling
        self._humanoid_root_states[env_ids, 2:3] = self.init_root_height
        self._humanoid_root_states[env_ids, 3:5] = 0
        self._humanoid_root_states[env_ids, 5:6] = 1
        self._humanoid_root_states[env_ids, 6:7] = -1
        self._humanoid_root_states[env_ids, 7:10] = 0
        self._humanoid_root_states[env_ids, 10:13] = 0
        
        self._dof_pos[env_ids] = self.init_dof
        self._dof_vel[env_ids] = 0
        return
    
    def _compute_ig_obs(self, env_ids, ref_obs):
        ig = self.ig[env_ids]
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
        ig_all = ig_all.view(env_ids.shape[0], -1)    
        ref_ig = self.extract_data_component('ig', obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids_gt, :]
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(env_ids.shape[0], -1)
        return ig_all, ig, ref_ig
    
    def _compute_observations(self, env_ids=None):
        # Ensure HOI observations (including self.ig) are computed before using _compute_ig_obs
        # During reset, upstream may call _compute_observations without calling _compute_hoi_observations first.
        # This guarantees self.ig is available for _compute_ig_obs.
        self._compute_hoi_observations()
        if (env_ids is None):
            self._curr_ref_obs[:] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(self.hoi_data, None, 1), self._compute_observations_iter(self.hoi_data, None, 16), (self.progress_buf >= 5).float().unsqueeze(1)), dim=-1)

        else:
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data, env_ids, 1), self._compute_observations_iter(self.hoi_data, env_ids, 16), (self.progress_buf[env_ids] >= 5).float().unsqueeze(1)), dim=-1)
            
        return
    
    def _compute_hoi_observations(self, env_ids=None):
        self._curr_obs[:] = self.build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                        self._rigid_body_rot[:, 0, :],
                                                        self._rigid_body_vel[:, 0, :],
                                                        self._rigid_body_ang_vel[:, 0, :],
                                                        self._dof_pos, self._dof_vel, self._rigid_body_pos,
                                                        self._local_root_obs, self._root_height_obs, 
                                                        self._dof_obs_size, self._target_states,
                                                        self._tar_contact_forces,
                                                        self._contact_forces,
                                                        self.object_points[self.object_id[self.data_id]],
                                                        self._rigid_body_rot,
                                                        self._rigid_body_vel,
                                                        self._rigid_body_ang_vel,
                                                        self._key_body_ids,
                                                        self._contact_body_ids,
                                                        self._key_body_ids_gt,
                                                        self._contact_body_ids_gt,
                                                        )
        return

    
    def build_hoi_observations(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, 
                               local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel, _key_body_ids, _contact_body_ids, _key_body_ids_gt, _contact_body_ids_gt):

        contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
        target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

        tar_pos = target_states[:, 0:3]
        tar_rot = target_states[:, 3:7]
        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        ig = compute_sdf(body_pos, obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1, 3)    
        self.ig = ig
        dof_pos_new = torch.zeros((root_pos.shape[0], 153), device=root_pos.device)
        dof_vel_new = torch.zeros((root_pos.shape[0], 153), device=root_pos.device)    
        dof_pos_new[:, :dof_pos.shape[1]] = dof_pos        
        dof_vel_new[:, :dof_vel.shape[1]] = dof_vel    
        contact_new = torch.zeros((root_pos.shape[0], 52), device=root_pos.device) 
        contact_new[:, _contact_body_ids_gt] = contact[:, _contact_body_ids]
        body_pos_new = torch.zeros((root_pos.shape[0], 52, 3), device=root_pos.device) 
        body_vel_new = torch.zeros((root_pos.shape[0], 52, 3), device=root_pos.device)
        ig_new = torch.zeros((root_pos.shape[0], 52, 3), device=root_pos.device)
        body_pos_new[:, _key_body_ids_gt] = body_pos[:, _key_body_ids]
        body_vel_new[:, _key_body_ids_gt] = body_vel[:, _key_body_ids]
        ig_new[:, _key_body_ids_gt] = ig[:, _key_body_ids]
        body_rot_new = torch.zeros((root_pos.shape[0], 52, 4), device=root_pos.device) 
        body_rot_vel_new = torch.zeros((root_pos.shape[0], 52, 3), device=root_pos.device) 
        body_rot_new[:, _key_body_ids_gt] = body_rot[:, _key_body_ids]
        body_rot_vel_new[:, _key_body_ids_gt] = body_rot_vel[:, _key_body_ids]
        obs = torch.cat((root_pos, root_rot, dof_pos_new, dof_vel_new, 
                         body_pos_new.view(body_pos_new.shape[0],-1), 
                         body_rot_new.view(body_rot_new.shape[0],-1), 
                         body_vel_new.view(body_vel_new.shape[0],-1), 
                         body_rot_vel_new.view(body_rot_vel_new.shape[0],-1),
                         target_states, ig_new.view(ig_new.shape[0],-1), contact_new, target_contact), dim=-1)        
        return obs
    
    def play_dataset_step(self, time):
        return
    
    # ============ G1-specific Reward Functions ============
    
    def compute_humanoid_reward_g1(self, w):
        """G1-specific humanoid reward computation."""
        len_keypos = len(self._key_body_ids)
        # 关键理解：build_hoi_observations将G1数据映射到52维GT布局
        # _curr_obs中的body_pos是52维的，其中_key_body_ids_gt位置存放了G1的关键身体数据
        # 但我们要提取的是G1实际的关键身体，所以用_key_body_ids_gt作为索引
        key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._key_body_ids_gt]
        ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._key_body_ids_gt]
        
        # Interaction graph weighting
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1)
        weight_h = (-5 * ref_ig_norm).exp()
        weight_hp = weight_h.clone().detach()
        
        # Higher weight for ankle/toe (important for locomotion)
        # 在GT布局中找到对应G1 ankle_roll的位置
        ankle_roll_gt_ids = []
        for i, body_name in enumerate(self.key_bodies):
            if 'ankle_roll' in body_name.lower():
                # 找到这个body在GT布局中的位置
                gt_idx = self._key_body_ids_gt[i].item()
                ankle_roll_gt_ids.append(gt_idx)
        
        if len(ankle_roll_gt_ids) > 0:
            weight_hp[:, ankle_roll_gt_ids] = 1.0
        
        # Body position reward
        # weight_hp的维度是52（GT布局），我们需要提取对应G1关键身体的权重
        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, self._key_body_ids_gt], dim=-1)
        rp = torch.exp(-ep * w['p'])
        
        # Body rotation reward
        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        
        er = torch.mean(diff[:, :] * weight_hr, dim=-1)
        rr = torch.exp(-er * w['r'])
        
        # Body velocity rewards
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2, dim=-1)
        rpv = torch.exp(-epv * w['pv'])
        
        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2, dim=-1)
        rrv = torch.exp(-erv * w['rv'])
        
        # Energy penalty - 按SMPLX方式计算，但使用G1的DOF数量
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel) * self.fps_data
        # G1有43个DOF，按SMPLX的方式reshape并计算
        dof_diffacc = (local_vel * (self.progress_buf - self.start_times > 2).float().unsqueeze(dim=-1)).clone()
        energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()
        
        rb = rp * rr * rpv * rrv * energy
        human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        
        return rb, human_reset, key_pos, ref_key_pos
    
    def compute_obj_reward_g1(self, w):
        """G1-specific object reward computation."""
        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        root_rot = self.extract_data_component('root_rot', obs=self._curr_obs)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        
        obj_pos = self.extract_data_component('obj_pos', obs=self._curr_obs)
        obj_rot = self.extract_data_component('obj_rot', obs=self._curr_obs)
        local_obj_pos = obj_pos - root_pos
        local_obj_pos[..., -1] = obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)
        local_obj_rot = quat_mul(heading_rot, obj_rot)
        
        object_points = self.object_points[self.object_id[self.data_id]]
        obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)
        
        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_root_rot = self.extract_data_component('root_rot', obs=self._curr_ref_obs)
        ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)
        
        ref_obj_pos = self.extract_data_component('obj_pos', obs=self._curr_ref_obs)
        ref_obj_rot = self.extract_data_component('obj_rot', obs=self._curr_ref_obs)
        ref_local_obj_pos = ref_obj_pos - ref_root_pos
        ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
        ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)
        ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)
        
        ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)
        
        # Object position reward
        eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2), dim=-1)
        rop = torch.exp(-eop * w['op'])
        
        # Object rotation reward
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 1)
        eor = torch.mean(diff, dim=-1)
        ror = torch.exp(-eor * w['or'])
        
        # Object velocity rewards
        obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_obs)
        ref_obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_ref_obs)
        eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2, dim=-1)
        ropv = torch.exp(-eopv * w['opv'])
        
        obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_obs)
        ref_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_ref_obs)
        eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2, dim=-1)
        rorv = torch.exp(-eorv * w['orv'])
        
        # Object energy penalty
        hist_obj_vel = self.extract_data_component('obj_pos_vel', obs=self._hist_obs)
        obj_diffacc = (self.extract_data_component('obj_pos_vel', obs=self._curr_obs) - hist_obj_vel) * self.fps_data
        obj_diffacc = obj_diffacc * (self.progress_buf - self.start_times > 2).float().unsqueeze(dim=-1)
        
        hist_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('obj_rot_vel', obs=self._curr_obs) - hist_obj_rot_vel) * self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3) * (self.progress_buf - self.start_times > 2).float().unsqueeze(dim=-1)
        
        obj_energy = (obj_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (obj_rot_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp())
        ro = rop * ror * ropv * rorv * obj_energy
        object_reset = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1) > 0.5
        
        return ro, object_reset, obj_points, ref_obj_points
    
    def compute_ig_reward_g1(self, w, key_pos, ref_key_pos, obj_points, ref_obj_points):
        """G1-specific interaction graph reward computation."""
        len_keypos = len(self._key_body_ids)
        ig = key_pos.view(-1, len_keypos, 3).unsqueeze(2) - obj_points.unsqueeze(1)
        ref_ig = ref_key_pos.view(-1, len_keypos, 3).unsqueeze(2) - ref_obj_points.unsqueeze(1)
        
        # Interaction graph reward
        weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.01))
        weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.01))
        weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        
        eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)
        rig = torch.exp(-w['ig'] * (eig.sum(dim=-1).sum(dim=-1) * 0.5))
        
        reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
        
        return rig, reset_ig
    
    def compute_cg_reward_g1(self, w):
        """G1-specific contact graph reward computation.
        返回形状需与 InterMimic.compute_cg_reward 一致：
        - rcg: (N,)
        - contact_reset: (N, 2)  # 左/右手的 contact reset 指示
        """
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)

        # 采用与 SMPLX GT 布局一致的左右手区间（当前观测为 52 槽位）
        left_contact_hand_ids = list(range(17, 33))
        right_contact_hand_ids = list(range(36, 52))

        # 左手 any-contact
        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, left_contact_hand_ids]
        left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        # 右手 any-contact
        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, right_contact_hand_ids]
        right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        # contact_reset 形状对齐为 (N, 2)
        contact_reset_left = torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(-1)
        contact_reset_right = torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1)
        contact_reset = torch.cat([contact_reset_left, contact_reset_right], dim=-1)

        # 左手接触奖励 - 与SMPLX完全一致
        ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left * w.get('cg_hand', 5.0))) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)

        # 右手接触奖励 - 与SMPLX完全一致
        ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right * w.get('cg_hand', 5.0))) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)

        rcg_hand = rcg_left * rcg_right

        # 其他部位接触奖励 - 与SMPLX一致
        other_ids = [i for i in range(len(self.contact_bodies)) if i not in left_contact_hand_ids and i not in right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other * w.get('cg_other', 5.0))

        # 全局接触奖励 - 与SMPLX一致
        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all * w.get('cg_all', 3.0))

        # 接触能量惩罚 - 与SMPLX一致
        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

        rcg = rcg_hand * rcg_other * rcg_all * contact_energy
        return rcg, contact_reset