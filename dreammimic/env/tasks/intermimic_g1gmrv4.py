import torch
import numpy as np
import time
try:
    import cv2
except Exception:
    cv2 = None

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid_g1 import Humanoid_G1
from env.tasks.intermimic import InterMimic, compute_sdf


class InterMimicG1(Humanoid_G1, InterMimic):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        sensor_cfg = cfg.get("sensor", {})
        self.enable_camera = bool(sensor_cfg.get("enableCamera", False))
        self.visual_input_type = sensor_cfg.get("visual_input_type", "depth_seg")
        self.depth_clip_lower = float(sensor_cfg.get("depth_clip_lower", 0.15))
        self.camera_handles = {}
        self.camera_sensor_dict = {}
        self.enable_realtime_vis = bool(sensor_cfg.get("enable_realtime_vis", False))
        self.vis_env_id = int(sensor_cfg.get("vis_env_id", cfg.get("env", {}).get("visualizeEnvId", 0)))
        self.vis_fps = float(sensor_cfg.get("vis_fps", 5.0))
        self.vis_scale = int(sensor_cfg.get("vis_scale", 4))
        self.last_vis_time = 0.0
        self._camera_window_name = f"G1 Camera Observations (Env {self.vis_env_id})"

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.hoi_data = self._load_motion(self.motion_file, startk=1, initk=15)
        self.scaling = cfg['env']['scaling']
        self.init_root_height = cfg['env']['initRootHeight']
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
        
        # 可视化相关设置
        self.enable_link_vis = cfg['env'].get('enableLinkVisualization', False)  # 默认关闭，测试时开启
        self.vis_env_id = int(sensor_cfg.get('vis_env_id', cfg['env'].get('visualizeEnvId', self.vis_env_id)))  # 要可视化的环境ID
        self.link_vis_scale = cfg['env'].get('linkVisScale', 0.05)  # 可视化marker的大小
        if self.enable_camera:
            self._setup_g1_camera_tensors()
            if self.enable_realtime_vis and cv2 is None:
                print("[G1 Camera] OpenCV is unavailable; realtime camera visualization is disabled.")
            else:
                print(f"[G1 Camera] enabled for env {self.vis_env_id}, type={self.visual_input_type}")
        
        return

    def _safe_reward_factor(self, reward):
        """Keep multiplicative reward terms finite and probability-like."""
        return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0).clamp_(min=0.0, max=1.0)

    def _compute_reward(self, actions):
        """
        Complete reward computation for G1.
        Adapted from InterMimic._compute_reward with G1-specific parameters.
        """
        rb, human_reset, key_pos, ref_key_pos = self.compute_humanoid_reward_g1(self.reward_weights)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward_g1(self.reward_weights)
        rig, ig_reset = self.compute_ig_reward_g1(self.reward_weights, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset = self.compute_cg_reward_g1(self.reward_weights)
        
        self.rew_buf[:] = self._safe_reward_factor(rb * ro * rig * rcg)
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
        self._update_reward_stats(rb, ro, rig, rcg, self.rew_buf)
        
        # 可视化link映射关系（仅在测试时且启用时）
        if self.enable_link_vis and self.viewer:
            self._visualize_link_mapping(key_pos, ref_key_pos)
        
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
        if self.enable_camera and env_id == self.vis_env_id:
            camera_handle = self._create_g1_onboard_camera(env_ptr, self.humanoid_handles[env_id])
            if camera_handle is not None:
                self.camera_handles[env_id] = camera_handle
        return   

    def _create_g1_onboard_camera(self, env_handle, actor_handle):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_cfg = self.cfg["sensor"]["onboard_camera"]
        camera_props.height = int(camera_cfg["resolution"][1])
        camera_props.width = int(camera_cfg["resolution"][0])
        if camera_cfg.get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = float(camera_cfg["horizontal_fov"])

        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        if camera_handle is None:
            return None

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*camera_cfg["position"])
        local_transform.r = gymapi.Quat.from_euler_zyx(*camera_cfg["rotation"])

        body_handle = -1
        for body_name in ("head_link", "mid360_link", "torso_link", "pelvis"):
            body_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, body_name)
            if body_handle != -1:
                break
        if body_handle == -1:
            print("[G1 Camera] Could not find a body for camera attachment.")
            return None

        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            body_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        return camera_handle

    def _setup_g1_camera_tensors(self):
        if not self.enable_camera or self.vis_env_id not in self.camera_handles:
            return
        env_handle = self.envs[self.vis_env_id]
        camera_handle = self.camera_handles[self.vis_env_id]
        try:
            depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH
            )
            seg_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, camera_handle, gymapi.IMAGE_SEGMENTATION
            )
        except Exception as exc:
            print(f"[G1 Camera] Failed to get camera tensors: {exc}")
            return
        self.camera_sensor_dict = {
            "depth": gymtorch.wrap_tensor(depth_tensor),
            "seg": gymtorch.wrap_tensor(seg_tensor),
        }

    def _update_g1_camera_vis(self):
        if not self.enable_camera or not self.enable_realtime_vis or cv2 is None:
            return
        if "depth" not in self.camera_sensor_dict or "seg" not in self.camera_sensor_dict:
            return

        now = time.time()
        if now - self.last_vis_time < 1.0 / max(self.vis_fps, 1e-6):
            return
        self.last_vis_time = now

        try:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            depth = self.camera_sensor_dict["depth"].clone()
            seg = self.camera_sensor_dict["seg"].clone()
            self.gym.end_access_image_tensors(self.sim)
        except Exception:
            try:
                self.gym.end_access_image_tensors(self.sim)
            except Exception:
                pass
            return

        camera_cfg = self.cfg["sensor"]["onboard_camera"]
        height = int(camera_cfg["resolution"][1])
        width = int(camera_cfg["resolution"][0])
        depth = depth.reshape(height, width)
        seg = seg.reshape(height, width)

        depth = depth.float()
        depth[depth < -2] = -2
        depth[depth > -self.depth_clip_lower] = 0
        depth = torch.clamp(-depth / 2.0, 0.0, 1.0)
        mask = (seg >= int(self.cfg.get("sensor", {}).get("segmentation_id", 2))).float()
        seg_depth = depth * mask

        depth_img = (depth.detach().cpu().numpy() * 255).astype(np.uint8)
        mask_img = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
        seg_depth_img = (seg_depth.detach().cpu().numpy() * 255).astype(np.uint8)

        display_size = (width * self.vis_scale, height * self.vis_scale)
        depth_display = cv2.resize(depth_img, display_size, interpolation=cv2.INTER_NEAREST)
        mask_display = cv2.resize(mask_img, display_size, interpolation=cv2.INTER_NEAREST)
        seg_depth_display = cv2.resize(seg_depth_img, display_size, interpolation=cv2.INTER_NEAREST)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
        seg_depth_colored = cv2.applyColorMap(seg_depth_display, cv2.COLORMAP_JET)
        combined = np.hstack([depth_colored, mask_colored, seg_depth_colored])
        cv2.imshow(self._camera_window_name, combined)
        cv2.waitKey(1)

    def post_physics_step(self):
        super().post_physics_step()
        self._update_g1_camera_vis()

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
    
    # ============ GMR-style Retarget (Limb-Length Aware) ============
    
    def _init_retarget_mapping(self):
        """构建 key_bodies 名称到局部索引的映射，并定义简单的四肢链，用于GMR风格的肢段缩放。"""
        if hasattr(self, "_gt_slot_by_name"):
            return
        self._gt_slot_by_name = {}  # name -> local idx in key_pos/ref_key_pos (0..K-1)
        try:
            key_body_names = self.key_bodies  # from cfg['env']['keyBodies'], length K
            for local_idx, name in enumerate(key_body_names):
                self._gt_slot_by_name[name] = local_idx
        except Exception:
            self._gt_slot_by_name = {}
        # 定义简单的肢段链（父 -> 子），与G1的link名称对应
        self._retarget_chains = [
            # Left leg
            ("left_hip_yaw_link", "left_knee_link"),
            ("left_knee_link", "left_ankle_roll_link"),
            # Right leg
            ("right_hip_yaw_link", "right_knee_link"),
            ("right_knee_link", "right_ankle_roll_link"),
            # Left arm
            ("left_shoulder_yaw_link", "left_elbow_link"),
            ("left_elbow_link", "left_wrist_yaw_link"),
            # Right arm
            ("right_shoulder_yaw_link", "right_elbow_link"),
            ("right_elbow_link", "right_wrist_yaw_link"),
        ]
        return

    def _retarget_ref_keypos(self, ref_body_pos_52, curr_body_pos_52):
        """
        GMR风格的retarget：
        1. 对四肢链进行长度缩放（保持当前实现）
        2. 对torso和shoulders等非四肢链进行整体缩放（类似GMR的human_scale_table）
        3. 减去ground offset，使得最低的foot点对齐到地面
        
        - ref_body_pos_52: (N, 52, 3) SMPLX参考关键点（GT布局）
        - curr_body_pos_52: (N, 52, 3) 当前G1关键点（GT布局）
        
        返回retarget后的SMPLX位置（已缩放并减去ground offset）
        """
        self._init_retarget_mapping()
        N = ref_body_pos_52.shape[0]
        eps = 1e-6
        
        # Step 1: 找到root（pelvis）位置，通常是52槽位的索引0
        root_idx_52 = 0  # SMPLX的root/pelvis在52槽位中的索引
        ref_root_pos = ref_body_pos_52[:, root_idx_52, :]  # (N, 3)
        
        # Step 2: 对四肢链进行长度缩放（保持原有逻辑）
        ret = ref_body_pos_52.clone()
        for parent_name, child_name in self._retarget_chains:
            p_slot_local = self._gt_slot_by_name.get(parent_name, None)
            c_slot_local = self._gt_slot_by_name.get(child_name, None)
            if p_slot_local is None or c_slot_local is None:
                continue
            p_slot_52 = self._key_body_ids_gt[p_slot_local]
            c_slot_52 = self._key_body_ids_gt[c_slot_local]
            
            # SMPLX 肢段向量
            vh = ref_body_pos_52[:, c_slot_52, :] - ref_body_pos_52[:, p_slot_52, :]
            # G1 当前肢段向量
            vr = curr_body_pos_52[:, c_slot_52, :] - curr_body_pos_52[:, p_slot_52, :]
            # 长度比例
            s = (vr.norm(dim=-1, keepdim=True) + eps) / (vh.norm(dim=-1, keepdim=True) + eps)
            # 沿着SMPLX方向缩放到G1长度
            ret[:, c_slot_52, :] = ret[:, p_slot_52, :] + s * vh
        
        # Step 3: 对torso和shoulders等非四肢链进行缩放
        # 计算legs和arms的平均缩放比例
        leg_scales = []
        arm_scales = []
        for parent_name, child_name in self._retarget_chains:
            p_slot_local = self._gt_slot_by_name.get(parent_name, None)
            c_slot_local = self._gt_slot_by_name.get(child_name, None)
            if p_slot_local is None or c_slot_local is None:
                continue
            p_slot_52 = self._key_body_ids_gt[p_slot_local]
            c_slot_52 = self._key_body_ids_gt[c_slot_local]
            
            vh = ref_body_pos_52[:, c_slot_52, :] - ref_body_pos_52[:, p_slot_52, :]
            vr = curr_body_pos_52[:, c_slot_52, :] - curr_body_pos_52[:, p_slot_52, :]
            s = (vr.norm(dim=-1, keepdim=True) + eps) / (vh.norm(dim=-1, keepdim=True) + eps)
            
            # 区分legs和arms
            if 'hip' in parent_name.lower() or 'knee' in parent_name.lower():
                leg_scales.append(s)
            elif 'shoulder' in parent_name.lower() or 'elbow' in parent_name.lower():
                arm_scales.append(s)
        
        # 计算平均缩放比例（per-batch）
        # 默认值（类似GMR的human_scale_table）
        default_torso_scale = 0.9
        default_shoulder_scale = 0.8
        
        if len(leg_scales) > 0:
            leg_avg_scale = torch.stack(leg_scales, dim=0).mean(dim=0)  # (N, 1)
            torso_scale = leg_avg_scale  # 使用leg的缩放比例作为torso的参考
        else:
            torso_scale = torch.full((N, 1), default_torso_scale, device=ref_body_pos_52.device, dtype=ref_body_pos_52.dtype)
        
        if len(arm_scales) > 0:
            arm_avg_scale = torch.stack(arm_scales, dim=0).mean(dim=0)  # (N, 1)
            shoulder_scale = arm_avg_scale  # 使用arm的缩放比例作为shoulder的参考
        else:
            shoulder_scale = torch.full((N, 1), default_shoulder_scale, device=ref_body_pos_52.device, dtype=ref_body_pos_52.dtype)
        
        # 对torso和shoulders进行缩放（在local frame中）
        torso_slot_local = self._gt_slot_by_name.get('torso_link', None)
        left_shoulder_slot_local = self._gt_slot_by_name.get('left_shoulder_yaw_link', None)
        right_shoulder_slot_local = self._gt_slot_by_name.get('right_shoulder_yaw_link', None)
        
        if torso_slot_local is not None:
            torso_idx_52 = self._key_body_ids_gt[torso_slot_local]
            local_torso = ref_body_pos_52[:, torso_idx_52, :] - ref_root_pos  # (N, 3)
            ret[:, torso_idx_52, :] = ref_root_pos + local_torso * torso_scale  # (N, 3) * (N, 1)
        
        if left_shoulder_slot_local is not None:
            left_shoulder_idx_52 = self._key_body_ids_gt[left_shoulder_slot_local]
            local_left_shoulder = ref_body_pos_52[:, left_shoulder_idx_52, :] - ref_root_pos  # (N, 3)
            ret[:, left_shoulder_idx_52, :] = ref_root_pos + local_left_shoulder * shoulder_scale  # (N, 3) * (N, 1)
        
        if right_shoulder_slot_local is not None:
            right_shoulder_idx_52 = self._key_body_ids_gt[right_shoulder_slot_local]
            local_right_shoulder = ref_body_pos_52[:, right_shoulder_idx_52, :] - ref_root_pos  # (N, 3)
            ret[:, right_shoulder_idx_52, :] = ref_root_pos + local_right_shoulder * shoulder_scale  # (N, 3) * (N, 1)
        
        # Step 4: 减去ground offset（找到最低的ankle/foot点，然后减去offset）
        # 找到ankle的索引
        left_ankle_slot_local = self._gt_slot_by_name.get('left_ankle_roll_link', None)
        right_ankle_slot_local = self._gt_slot_by_name.get('right_ankle_roll_link', None)
        
        lowest_heights = []
        if left_ankle_slot_local is not None:
            left_ankle_idx_52 = self._key_body_ids_gt[left_ankle_slot_local]
            lowest_heights.append(ret[:, left_ankle_idx_52, 2])  # z坐标
        if right_ankle_slot_local is not None:
            right_ankle_idx_52 = self._key_body_ids_gt[right_ankle_slot_local]
            lowest_heights.append(ret[:, right_ankle_idx_52, 2])
        
        if len(lowest_heights) > 0:
            # 找到最低的foot高度
            lowest_height = torch.stack(lowest_heights, dim=1).min(dim=1)[0]  # (N,)
            # Ground offset：减去最低点的高度，使得foot对齐到地面（z=0）
            # 注意：这里我们减去最低点的高度，使得foot刚好在地面上
            height_offset = lowest_height  # (N,)
            ret[:, :, 2] = ret[:, :, 2] - height_offset.unsqueeze(-1)  # (N, 52) - (N, 1)
        
        return ret
    
    # ============ G1-specific Reward Functions ============
    
    def compute_humanoid_reward_g1(self, w):
        """G1-specific humanoid reward computation (GMR风格retarget + link位置reward)."""
        len_keypos = len(self._key_body_ids_gt)
        # 注意：在 build_hoi_observations 中，当前观测已被映射到 52 维 GT 布局
        # 因此这里应当统一使用 52 槽位，再通过 _key_body_ids_gt 取子集
        body_pos_52 = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)     # (N, 52, 3)
        ref_body_pos_52 = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)  # (N, 52, 3)

        # 当前G1关键点（K个keyBodies在52槽位的子集）
        key_pos = body_pos_52[:, self._key_body_ids_gt, :]  # (N, K, 3)
        # 原始SMPLX关键点（未retarget）
        ref_key_pos_smplx = ref_body_pos_52[:, self._key_body_ids_gt, :]  # (N, K, 3)

        # GMR风格retarget：在52槽位上做肢段缩放，再投影回关键点集合
        ref_body_pos_gmr_52 = self._retarget_ref_keypos(ref_body_pos_52, body_pos_52)
        ref_key_pos_gmr = ref_body_pos_gmr_52[:, self._key_body_ids_gt, :]  # (N, K, 3)

        # 保存三套关键点用于可视化：
        # - 原始SMPLX: 绿色
        # - GMR目标:   红色
        # - 当前G1:    蓝色
        self._vis_ref_key_pos_smplx = ref_key_pos_smplx.detach()
        self._vis_ref_key_pos_gmr = ref_key_pos_gmr.detach()
        self._vis_key_pos_curr = key_pos.detach()
        
        # 打印一次link名称和顺序（仅在第一个环境的第一帧打印）
        # if not hasattr(self, '_vis_link_names_printed'):
        #     print("\n" + "="*80)
        #     print("Link Visualization Mapping (for debugging):")
        #     print("="*80)
        #     print(f"Total key bodies: {len(self.key_bodies)}")
        #     print(f"_key_body_ids_gt (52-slot indices): {self._key_body_ids_gt.cpu().tolist()}")
        #     print("\nLink order for each colored sphere:")
        #     print("-" * 80)
        #     for i, (link_name, gt_idx) in enumerate(zip(self.key_bodies, self._key_body_ids_gt.cpu().tolist())):
        #         print(f"  Index {i:2d}: {link_name:30s} -> 52-slot index {gt_idx:2d}")
        #         print(f"    - Green sphere (SMPLX):   ref_key_pos_smplx[:, {i}, :]")
        #         print(f"    - Red sphere (GMR target): ref_key_pos_gmr[:, {i}, :]")
        #         print(f"    - Blue sphere (G1 curr):   key_pos[:, {i}, :]")
        #     print("="*80 + "\n")
        #     self._vis_link_names_printed = True

        # 后续reward使用GMR目标ref_key_pos_gmr作为参考
        ref_key_pos = ref_key_pos_gmr
        
        # Interaction graph weighting（仍然使用SMPLX参考的IG）
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1)
        weight_h = (-5 * ref_ig_norm).exp()
        weight_hp = weight_h.clone().detach()
        
        # Higher weight for ankle/toe (important for locomotion)
        ankle_roll_ids = [i for i in range(len_keypos) if 'ankle_roll' in self.key_bodies[i].lower()]
        if len(ankle_roll_ids) > 0:
            weight_hp[:, ankle_roll_ids] = 1.0
        
        # Body position reward
        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, self._key_body_ids_gt], dim=-1)
        rp = self._safe_reward_factor(torch.exp(-ep * w['p']))
        
        # Body rotation reward
        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        
        er = torch.mean(diff[:, :] * weight_hr, dim=-1)
        rr = self._safe_reward_factor(torch.exp(-er * w['r']))
        
        # Body velocity rewards
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2, dim=-1)
        rpv = self._safe_reward_factor(torch.exp(-epv * w['pv']))
        
        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2, dim=-1)
        rrv = self._safe_reward_factor(torch.exp(-erv * w['rv']))
        
        # Energy penalty (G1: 以自身 DoF 维度计算，不再使用 SMPLX 的 51*3 形状)
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel) * self.fps_data
        dof_diffacc = local_vel * (self.progress_buf - self.start_times > 2).float().unsqueeze(dim=-1)
        energy = self._safe_reward_factor(dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp())
        
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
        rop = self._safe_reward_factor(torch.exp(-eop * w['op']))
        
        # Object rotation reward
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 1)
        eor = torch.mean(diff, dim=-1)
        ror = self._safe_reward_factor(torch.exp(-eor * w['or']))
        
        # Object velocity rewards
        obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_obs)
        ref_obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_ref_obs)
        eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2, dim=-1)
        ropv = self._safe_reward_factor(torch.exp(-eopv * w['opv']))
        
        obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_obs)
        ref_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_ref_obs)
        eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2, dim=-1)
        rorv = self._safe_reward_factor(torch.exp(-eorv * w['orv']))
        
        # Object energy penalty
        hist_obj_vel = self.extract_data_component('obj_pos_vel', obs=self._hist_obs)
        obj_diffacc = (self.extract_data_component('obj_pos_vel', obs=self._curr_obs) - hist_obj_vel) * self.fps_data
        obj_diffacc = obj_diffacc * (self.progress_buf - self.start_times > 2).float().unsqueeze(dim=-1)
        
        hist_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('obj_rot_vel', obs=self._curr_obs) - hist_obj_rot_vel) * self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3) * (self.progress_buf - self.start_times > 2).float().unsqueeze(dim=-1)
        
        obj_energy = self._safe_reward_factor(
            obj_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()
            * obj_rot_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()
        )
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
        rig = self._safe_reward_factor(torch.exp(-w['ig'] * (eig.sum(dim=-1).sum(dim=-1) * 0.5)))
        
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

        # 简化的接触奖励：整体接触分布与参考贴合（使用 cg1）
        ecg_all = ((ref_human_contact - human_contact).abs()).mean(dim=-1)
        rcg_all = self._safe_reward_factor(torch.exp(-ecg_all * w['cg1']))

        # 接触能量项（与原版一致）
        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = self._safe_reward_factor(contact_all.pow(2).mul(-w['eg3']).exp())

        rcg = self._safe_reward_factor(rcg_all * contact_energy)
        return rcg, contact_reset
    
    def _visualize_link_mapping(self, key_pos, ref_key_pos):
        """
        可视化三套link位置：
        - 绿色球体：原始SMPLX各个link位置（当与GMR不重合时）
        - 红色球体：GMR风格retarget后的G1目标link位置（当与SMPLX不重合时）
        - 黄色球体：当SMPLX和GMR位置重合时（距离<0.01m），显示为黄色，避免颜色覆盖
        - 蓝色球体：当前G1各个link位置
        
        额外：从GMR目标（红）到当前G1（蓝）的连接线，颜色表示误差大小：
           - 红色=大误差（>0.3m）
           - 黄色=中等误差（0.1-0.3m）
           - 绿色=小误差（<0.1m）
        """
        # 三套关键点必须都存在
        if (not hasattr(self, "_vis_ref_key_pos_smplx") or
            not hasattr(self, "_vis_ref_key_pos_gmr") or
            not hasattr(self, "_vis_key_pos_curr")):
            return
        
        # 只可视化指定的环境
        env_id = self.vis_env_id
        if env_id >= self.num_envs:
            return
        
        # 获取当前环境三套关键点
        smplx_pos = self._vis_ref_key_pos_smplx[env_id].cpu().numpy()  # 绿色（K, 3）
        gmr_pos = self._vis_ref_key_pos_gmr[env_id].cpu().numpy()      # 红色（K, 3）
        curr_pos = self._vis_key_pos_curr[env_id].cpu().numpy()        # 蓝色（K, 3）
        
        # 获取环境指针
        env_ptr = self.envs[env_id]
        
        # 清除之前的线条
        self.gym.clear_lines(self.viewer)
        
        # 准备线条数据（用于连接线）
        num_links = curr_pos.shape[0]
        lines = []
        colors = []
        
        # 球体半径（从配置读取，默认0.05）
        sphere_radius = self.link_vis_scale if hasattr(self, 'link_vis_scale') else 0.05
        
        # 打印一次位置信息用于调试（仅在第一个环境的第一帧打印）
        if not hasattr(self, '_vis_positions_printed'):
            print("\n" + "="*80)
            print(f"Link Positions for Visualization (Env {env_id}, first frame):")
            print("="*80)
            for i in range(min(num_links, len(self.key_bodies))):
                link_name = self.key_bodies[i] if i < len(self.key_bodies) else f"Unknown_{i}"
                smplx_p = smplx_pos[i]
                gmr_p = gmr_pos[i]
                curr_p = curr_pos[i]
                error_gmr_curr = np.linalg.norm(gmr_p - curr_p)
                error_smplx_curr = np.linalg.norm(smplx_p - curr_p)
                print(f"\n  Link {i:2d}: {link_name}")
                print(f"    Green (SMPLX):   [{smplx_p[0]:7.3f}, {smplx_p[1]:7.3f}, {smplx_p[2]:7.3f}]")
                print(f"    Red (GMR):       [{gmr_p[0]:7.3f}, {gmr_p[1]:7.3f}, {gmr_p[2]:7.3f}]")
                print(f"    Blue (G1 curr):   [{curr_p[0]:7.3f}, {curr_p[1]:7.3f}, {curr_p[2]:7.3f}]")
                print(f"    Error (GMR->G1):  {error_gmr_curr:.4f}m")
                print(f"    Error (SMPLX->G1): {error_smplx_curr:.4f}m")
            print("="*80 + "\n")
            self._vis_positions_printed = True
        
        # 重合阈值（米）
        overlap_threshold = 0.01
        
        for i in range(num_links):
            smplx_p = smplx_pos[i]  # 原始SMPLX (3,)
            gmr_p = gmr_pos[i]      # GMR目标 (3,)
            curr_p = curr_pos[i]    # 当前G1 (3,)
            
            # 检查SMPLX和GMR是否重合
            smplx_gmr_distance = np.linalg.norm(smplx_p - gmr_p)
            is_overlap = smplx_gmr_distance < overlap_threshold
            
            # 计算 GMR目标 -> 当前G1 的误差
            error = np.linalg.norm(gmr_p - curr_p)
            
            # 连接线的颜色：根据误差大小设置
            if error > 0.3:
                # 大误差：红色
                line_color = np.array([1.0, 0.0, 0.0])
            elif error > 0.1:
                # 中等误差：黄色（红+绿）
                line_color = np.array([1.0, 1.0, 0.0])
            else:
                # 小误差：绿色
                line_color = np.array([0.0, 1.0, 0.0])
            
            # 添加从GMR目标位置(红)到当前G1位置(蓝)的连接线（重复绘制多次使其更粗）
            for _ in range(3):
                lines.append([gmr_p[0], gmr_p[1], gmr_p[2],
                             curr_p[0], curr_p[1], curr_p[2]])
                colors.append(line_color)
            
            # 绘制SMPLX和GMR位置
            if is_overlap:
                # 如果SMPLX和GMR重合，画一个黄色球体
                sphere_geom_overlap = gymutil.WireframeSphereGeometry(
                    sphere_radius, 20, 20, None, color=(1.0, 1.0, 0.0))
                # 使用SMPLX位置（或GMR位置，因为它们重合）
                sphere_pose_overlap = gymapi.Transform(
                    gymapi.Vec3(smplx_p[0], smplx_p[1], smplx_p[2]), r=None)
                gymutil.draw_lines(sphere_geom_overlap, self.gym, self.viewer, env_ptr, sphere_pose_overlap)
            else:
                # 如果不重合，分别画绿色和红色球体
                # 1) 原始SMPLX位置：绿色球体
                sphere_geom_smplx = gymutil.WireframeSphereGeometry(
                    sphere_radius, 20, 20, None, color=(0.0, 1.0, 0.0))
                sphere_pose_smplx = gymapi.Transform(
                    gymapi.Vec3(smplx_p[0], smplx_p[1], smplx_p[2]), r=None)
                gymutil.draw_lines(sphere_geom_smplx, self.gym, self.viewer, env_ptr, sphere_pose_smplx)
                
                # 2) GMR目标位置：红色球体
                sphere_geom_gmr = gymutil.WireframeSphereGeometry(
                    sphere_radius, 20, 20, None, color=(1.0, 0.0, 0.0))
                sphere_pose_gmr = gymapi.Transform(
                    gymapi.Vec3(gmr_p[0], gmr_p[1], gmr_p[2]), r=None)
                gymutil.draw_lines(sphere_geom_gmr, self.gym, self.viewer, env_ptr, sphere_pose_gmr)
            
            # 3) 当前G1位置：蓝色球体
            sphere_geom_curr = gymutil.WireframeSphereGeometry(
                sphere_radius, 20, 20, None, color=(0.0, 0.0, 1.0))
            sphere_pose_curr = gymapi.Transform(
                gymapi.Vec3(curr_p[0], curr_p[1], curr_p[2]), r=None)
            gymutil.draw_lines(sphere_geom_curr, self.gym, self.viewer, env_ptr, sphere_pose_curr)
        
        # 添加连接线到viewer
        if len(lines) > 0:
            lines_array = np.array(lines, dtype=np.float32)  # (N, 6)
            colors_array = np.array(colors, dtype=np.float32)  # (N, 3)
            self.gym.add_lines(self.viewer, env_ptr, len(lines_array), 
                              lines_array, colors_array)
        
        return