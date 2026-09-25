import torch
import numpy as np
import os
import threading
import time
from collections import defaultdict
from queue import Queue
from torchvision import transforms
try:
    import cv2
except Exception:
    cv2 = None

from env.tasks.intermimic_all import InterMimic_All
from env.tasks.intermimic import InterMimic

from isaacgym import gymapi
from isaacgym import gymtorch


class InterMimic_All_NoRef_Vis(InterMimic_All):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Keep both teacher and student observation sizes distinct
        self.global_step_counter = 0
        student_obs_size = cfg["env"]["numObs"]
        teacher_obs_size = cfg["env"].get("teacherObsSize", student_obs_size)
        # Whether to drop privileged object info from student obs
        self.remove_privileged_obs = bool(cfg.get("env", {}).get("removePrivilegedObs", False))
        
        # Camera configuration - initialize early
        self.enable_camera = cfg.get("sensor", {}).get("enableCamera", False)
        self.camera_mode = "front_only"  # Simple front camera only for SMPLX
        self.depth_clip_lower = 0.15
        # Visual input type: 'rgb', 'depth', 'segmentation', 'depth_seg' (default)
        self.visual_input_type = cfg.get("sensor", {}).get("visual_input_type", "depth_seg")
        
        # Initialize camera components early, before super().__init__()
        self.camera_sensor_dict = defaultdict(list)
        self.camera_handles = []
        # Visualization flags
        sensor_cfg = cfg.get("sensor", {})
        self.cam_visualize = bool(sensor_cfg.get("visualize", True))
        self.cam_visualize_stride = int(sensor_cfg.get("visualize_stride", 5))
        self.cam_vis_env = int(sensor_cfg.get("visualize_env", 0))
        self.cam_output_dir = sensor_cfg.get("output_dir", "camera_vis")
        
        # Real-time visualization settings
        self.enable_realtime_vis = bool(sensor_cfg.get("enable_realtime_vis", False))
        self.vis_fps = float(sensor_cfg.get("vis_fps", 5.0))  # Visualization frame rate
        self.vis_env_id = int(sensor_cfg.get("vis_env_id", 0))  # Environment to visualize
        self.vis_scale = int(sensor_cfg.get("vis_scale", 4))  # Display scale factor
        
        # Threading for visualization
        self.vis_queue = None
        self.vis_thread = None
        self.vis_stop_event = None
        self.last_vis_time = 0.0
        
        if self.cam_visualize:
            try:
                os.makedirs(self.cam_output_dir, exist_ok=True)
            except Exception:
                pass
        
        # Initialize real-time visualization if enabled
        if self.enable_realtime_vis and cv2 is not None:
            self.vis_queue = Queue(maxsize=2)  # Small queue to avoid lag
            self.vis_stop_event = threading.Event()
            self.vis_thread = threading.Thread(target=self._vis_thread_worker, daemon=True)
            self.vis_thread.start()
            print(f"[Camera Vis] Real-time visualization enabled: {self.vis_fps} FPS, env_id={self.vis_env_id}, scale={self.vis_scale}x")
        
        if self.enable_camera:
            self.resize_transform = transforms.Resize((cfg["sensor"]["resized_resolution"][1], cfg["sensor"]["resized_resolution"][0]))

        # Temporarily switch to teacher obs size to build and load teacher models in parent init
        cfg["env"]["numObs"] = teacher_obs_size
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        # Restore student obs size for RL policy (student network)
        self.cfg["env"]["numObs"] = student_obs_size
        self._num_obs = student_obs_size

        # Allocate distinct buffers: teacher (full obs) and student (no-ref)
        self.teacher_obs_size = teacher_obs_size
        self.obs_buf_teacher = torch.zeros((self.num_envs, self.teacher_obs_size), device=self.device, dtype=torch.float)
        
        # Calculate student obs size based on whether camera is enabled and privileged removal
        # Derive base current-observation length from kinematic sizes
        num_bodies = int(self._rigid_body_pos.shape[1])
        dof_dim = int(self._dof_pos.shape[1])
        ts_dim = int(self._target_states.shape[1]) if hasattr(self, '_target_states') else 13
        base_full_len = 8 + 2 * dof_dim + 17 * num_bodies + ts_dim  # see build_hoi_observations layout
        base_no_priv = 7 + 2 * dof_dim + 14 * num_bodies            # drop ts_dim + 3*num_bodies + 1
        base_len = base_no_priv if self.remove_privileged_obs else base_full_len

        if self.enable_camera:
            # Initialize camera history buffer
            self.camera_history_len = 3
            img_size = self.cfg["sensor"]["resized_resolution"][0] * self.cfg["sensor"]["resized_resolution"][1]
            num_channels = 2  # depth + segmentation
            # Determine number of channels based on visual_input_type
            if self.visual_input_type == 'rgb':
                num_channels = 3
            elif self.visual_input_type == 'depth' or self.visual_input_type == 'segmentation':
                num_channels = 1
            elif self.visual_input_type == 'depth_seg':
                num_channels = 2
            else:
                num_channels = 2  # default
            
            visual_obs_size = img_size * num_channels * self.camera_history_len
            actual_student_obs_size = base_len + visual_obs_size
            
            self.camera_history_buf = torch.zeros(
                self.num_envs, self.camera_history_len, img_size * num_channels, 
                device=self.device, dtype=torch.float
            )
        else:
            actual_student_obs_size = base_len
            
        self.student_obs_buf = torch.zeros((self.num_envs, actual_student_obs_size), device=self.device, dtype=torch.float)

        # Expose student buffer as the main obs buffer for RL
        self.obs_buf = self.student_obs_buf
        # Update reported observation size to match computed
        self._num_obs = actual_student_obs_size
        try:
            self.cfg["env"]["numObservations"] = actual_student_obs_size
        except Exception:
            pass

    def _strip_privileged_from_obs(self, obs_tensor):
        """Remove privileged components (object state, IG terms, target contact) from obs.
        Keeps proprioception-only terms. Works for full batch or indexed batch.
        """
        if obs_tensor is None or obs_tensor.numel() == 0:
            return obs_tensor
        try:
            num_bodies = int(self._rigid_body_pos.shape[1])
            dof_dim = int(self._dof_pos.shape[1])
            ts_dim = int(self._target_states.shape[1]) if hasattr(self, '_target_states') else 13
            contact_len = num_bodies
            ig_len = num_bodies * 3

            # Segment boundaries per build_hoi_observations concatenation order
            idx = 0
            idx += 3                           # root_pos
            idx += 4                           # root_rot
            idx += dof_dim                     # dof_pos
            idx += dof_dim                     # dof_vel
            idx += num_bodies * 3              # body_pos
            idx += num_bodies * 4              # body_rot
            idx += num_bodies * 3              # body_vel
            idx += num_bodies * 3              # body_rot_vel
            ts_start = idx
            ig_start = ts_start + ts_dim
            contact_start = ig_start + ig_len
            tar_contact_start = contact_start + contact_len
            tar_contact_len = 1

            total_dim = obs_tensor.shape[1]
            mask = torch.ones(total_dim, dtype=torch.bool, device=obs_tensor.device)
            # Drop target_states
            mask[ts_start:ts_start + ts_dim] = False
            # Drop IG terms
            mask[ig_start:ig_start + ig_len] = False
            # Drop target_contact
            end_tc = min(tar_contact_start + tar_contact_len, total_dim)
            if tar_contact_start < total_dim:
                mask[tar_contact_start:end_tc] = False

            return obs_tensor[:, mask]
        except Exception:
            # Fallback: if any shape assumption fails, return original
            return obs_tensor

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
            # Optionally remove privileged info
            if self.remove_privileged_obs:
                student_obs = self._strip_privileged_from_obs(student_obs)
            
            # Add visual information if camera is enabled
            if self.enable_camera and hasattr(self, 'camera_history_buf'):
                # Flatten camera history and concatenate with student obs
                visual_obs = self.camera_history_buf.view(self.num_envs, -1)
                student_obs = torch.cat([student_obs, visual_obs], dim=-1)
            
            # Ensure the student obs buffer has the right size
            if student_obs.shape[1] != self.student_obs_buf.shape[1]:
                print(f"Warning: student obs size mismatch. Expected {self.student_obs_buf.shape[1]}, got {student_obs.shape[1]}")
                # Resize buffer if needed
                self.student_obs_buf = torch.zeros((self.num_envs, student_obs.shape[1]), device=self.device, dtype=torch.float)
                
            self.student_obs_buf[:] = student_obs
            # Replace obs_buf with student obs (no-ref + visual) for student policy
            self.obs_buf = self.student_obs_buf
        else:
            student_obs = self._curr_obs[env_ids].clone()
            # Optionally remove privileged info
            if self.remove_privileged_obs:
                student_obs = self._strip_privileged_from_obs(student_obs)
            
            # Add visual information if camera is enabled
            if self.enable_camera and hasattr(self, 'camera_history_buf'):
                # Flatten camera history and concatenate with student obs
                visual_obs = self.camera_history_buf[env_ids].view(len(env_ids), -1)
                student_obs = torch.cat([student_obs, visual_obs], dim=-1)
            
            # Ensure the student obs buffer has the right size
            if student_obs.shape[1] != self.student_obs_buf.shape[1]:
                print(f"Warning: student obs size mismatch. Expected {self.student_obs_buf.shape[1]}, got {student_obs.shape[1]}")
                # Resize buffer if needed
                self.student_obs_buf = torch.zeros((self.num_envs, student_obs.shape[1]), device=self.device, dtype=torch.float)
                
            self.student_obs_buf[env_ids] = student_obs
            # Replace obs_buf with student obs (no-ref + visual) for student policy
            self.obs_buf[env_ids] = self.student_obs_buf[env_ids]
        return

    def step(self, weights):
        # Same as parent, but feed teacher models with full-observation buffer
        self.pre_physics_step(weights)
        self.global_step_counter+=1
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

        
    def _create_onboard_camera(self, env_handle, actor_handle, env_idx):
        """Create onboard camera for SMPLX humanoid"""
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = self.cfg["sensor"]["onboard_camera"]["resolution"][1]
        camera_props.width = self.cfg["sensor"]["onboard_camera"]["resolution"][0]
        
        if self.cfg["sensor"]["onboard_camera"].get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = self.cfg["sensor"]["onboard_camera"]["horizontal_fov"]
            
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        if camera_handle is None:
            return None
        
        # Set camera position and orientation relative to head
        local_transform = gymapi.Transform()
        local_pos = self.cfg["sensor"]["onboard_camera"]["position"].copy()
        # Add small randomization
        local_pos[0] += np.random.uniform(-0.01, 0.01)
        local_pos[1] += np.random.uniform(-0.01, 0.01)
        local_pos[2] += np.random.uniform(-0.01, 0.01)
        local_transform.p = gymapi.Vec3(*local_pos)
        
        local_rot = self.cfg["sensor"]["onboard_camera"]["rotation"].copy()
        local_rot[1] += np.random.uniform(-0.087, 0.087)  # Small pitch variation
        local_transform.r = gymapi.Quat.from_euler_zyx(*local_rot)
        
        # Attach camera to head body
        head_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "Pelvis")
        if head_handle == -1:
            # Fallback to torso if head not found
            head_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "Torso")
        # import ipdb; ipdb.set_trace()
        # Validate handles before attaching
        try:
            if camera_handle is not None and head_handle != -1:
                self.gym.attach_camera_to_body(
                    camera_handle, env_handle, head_handle, local_transform, gymapi.FOLLOW_TRANSFORM
                )
            else:
                return None
        except Exception:
            return None
        
        return camera_handle
        
    def _setup_camera_tensors(self):
        """Setup camera image tensors after all environments are created"""
        if not self.enable_camera:
            return
            
        for env_i, env_handle in enumerate(self.envs):
            if env_i >= len(self.camera_handles):
                self.camera_sensor_dict["forward_depth"].append(None)
                self.camera_sensor_dict["forward_seg"].append(None)
                self.camera_sensor_dict["forward_color"].append(None)
                continue
            if self.camera_handles[env_i] is None:
                self.camera_sensor_dict["forward_depth"].append(None)
                self.camera_sensor_dict["forward_seg"].append(None)
                self.camera_sensor_dict["forward_color"].append(None)
                continue
            # Get depth image tensor
            try:
                depth_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.camera_handles[env_i], gymapi.IMAGE_DEPTH
                )
                self.camera_sensor_dict["forward_depth"].append(gymtorch.wrap_tensor(depth_tensor))
            except Exception:
                self.camera_sensor_dict["forward_depth"].append(None)
            
            # Get segmentation image tensor  
            try:
                seg_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.camera_handles[env_i], gymapi.IMAGE_SEGMENTATION
                )
                self.camera_sensor_dict["forward_seg"].append(gymtorch.wrap_tensor(seg_tensor))
            except Exception:
                self.camera_sensor_dict["forward_seg"].append(None)

            # Get RGB image tensor
            try:
                color_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.camera_handles[env_i], gymapi.IMAGE_COLOR
                )
                self.camera_sensor_dict["forward_color"].append(gymtorch.wrap_tensor(color_tensor))
            except Exception:
                self.camera_sensor_dict["forward_color"].append(None)
    
    def _get_seg_id(self):
        """Get segmentation ID for objects of interest"""
        # Use config if provided, otherwise default to object IDs (2+)
        try:
            return int(self.cfg.get("sensor", {}).get("segmentation_id", 2))
        except Exception:
            return 2
        
    def _get_front_camera_obs(self):
        """Get front camera observations based on visual_input_type
        
        Returns:
            tuple: (visual_obs, depth_obs, seg_obs) where visual_obs is the formatted observation
                   based on visual_input_type, and depth_obs/seg_obs are for compatibility
        """
        if not self.enable_camera:
            return None, None, None
        
        # Get RGB image if needed
        rgb_image = None
        if self.visual_input_type == 'rgb':
            if len(self.camera_sensor_dict.get("forward_color", [])) > 0:
                color_tensors = [t for t in self.camera_sensor_dict["forward_color"] if t is not None]
                if len(color_tensors) > 0:
                    # Stack tensors: each tensor is (H, W, C) or (H*W*C,)
                    rgb_image = torch.stack(color_tensors).to(self.device).clone()
                    
                    # Handle different tensor shapes
                    if rgb_image.dim() == 1:
                        # Flattened: (H*W*C,) -> reshape to (H, W, C)
                        H, W = self.cfg["sensor"]["onboard_camera"]["resolution"][1], self.cfg["sensor"]["onboard_camera"]["resolution"][0]
                        rgb_image = rgb_image.view(H, W, -1)
                    elif rgb_image.dim() == 2:
                        # (H, W) -> add channel dim
                        rgb_image = rgb_image.unsqueeze(-1)
                        if rgb_image.shape[-1] == 1:
                            # Grayscale to RGB
                            rgb_image = rgb_image.repeat(1, 1, 3)
                    
                    # Handle channel dimension
                    if rgb_image.dim() == 3:
                        if rgb_image.shape[-1] == 4:
                            # RGBA to RGB
                            rgb_image = rgb_image[:, :, :3]
                        elif rgb_image.shape[-1] == 1:
                            # Grayscale to RGB
                            rgb_image = rgb_image.repeat(1, 1, 3)
                        elif rgb_image.shape[-1] != 3:
                            # Unexpected channel count, pad or slice
                            if rgb_image.shape[-1] < 3:
                                padding = torch.zeros(rgb_image.shape[0], rgb_image.shape[1], 3 - rgb_image.shape[-1], 
                                                     device=rgb_image.device, dtype=rgb_image.dtype)
                                rgb_image = torch.cat([rgb_image, padding], dim=-1)
                            else:
                                rgb_image = rgb_image[:, :, :3]
                    
                    # Normalize to [0, 1] if needed (assuming uint8 input)
                    if rgb_image.max() > 1.0:
                        rgb_image = rgb_image / 255.0
        
        # Get segmentation mask if needed
        seg_image = None
        forward_mask_image = None
        if self.visual_input_type in ['segmentation', 'depth_seg']:
            if len(self.camera_sensor_dict.get("forward_seg", [])) == 0 or any(t is None for t in self.camera_sensor_dict["forward_seg"]):
                return None, None, None
            seg_id = self._get_seg_id()
            seg_tensors = [t for t in self.camera_sensor_dict["forward_seg"] if t is not None]
            seg_image = torch.stack(seg_tensors).to(self.device).clone()
            
            # Create mask for objects (seg_id >= 2, excluding ground=0 and humanoid=1)
            forward_mask = (seg_image >= seg_id)
            forward_mask_image = forward_mask.float()
            forward_mask_image[forward_mask_image > 0.5] = 1.
            forward_mask_image[forward_mask_image <= 0.5] = 0.
            
            # Ensure consistent shape: seg_image from torch.stack is (B, H, W) or (B, H*W)
            # forward_mask_image will be (B, H, W) if seg_image is (B, H, W)
            # We need to handle batch dimension properly
            if forward_mask_image.dim() == 3:
                # Could be (B, H, W) or (H, W, C) - check by shape
                # If first dim is small (likely batch), treat as (B, H, W)
                # Otherwise treat as (H, W, C)
                if forward_mask_image.shape[0] < forward_mask_image.shape[-1] and forward_mask_image.shape[-1] not in [1, 2, 3]:
                    # Likely (B, H, W) - keep as is, will be handled in resize
                    pass
                elif forward_mask_image.shape[-1] == 1:
                    # (H, W, 1) -> squeeze to (H, W)
                    forward_mask_image = forward_mask_image.squeeze(-1)
            elif forward_mask_image.dim() == 4:
                # (B, C, H, W) or (B, H, W, C) - unlikely but handle it
                if forward_mask_image.shape[0] == 1:
                    forward_mask_image = forward_mask_image.squeeze(0)
        
        # Get depth image if needed
        normalized_depth = None
        if self.visual_input_type in ['depth', 'depth_seg']:
            if len(self.camera_sensor_dict.get("forward_depth", [])) == 0 or any(t is None for t in self.camera_sensor_dict["forward_depth"]):
                return None, None, None
            depth_tensors = [t for t in self.camera_sensor_dict["forward_depth"] if t is not None]
            depth_image = torch.stack(depth_tensors).to(self.device).clone()
            depth_image[depth_image < -2] = -2
            depth_image[depth_image > -self.depth_clip_lower] = 0
            depth_image *= -1
            normalized_depth = (depth_image - 0.) / (2. - 0.)
        
        # Store original resolution images for visualization
        if normalized_depth is not None:
            self.depth_image_original = normalized_depth.clone()
        if forward_mask_image is not None:
            self.forward_mask_original = forward_mask_image.clone()
        
        # Build visual observation based on visual_input_type
        if self.visual_input_type == 'rgb':
            visual_obs = rgb_image
            if visual_obs is None:
                return None, None, None
        elif self.visual_input_type == 'depth':
            visual_obs = normalized_depth
            if visual_obs is None:
                return None, None, None
        elif self.visual_input_type == 'segmentation':
            visual_obs = forward_mask_image
            if visual_obs is None:
                return None, None, None
            # forward_mask_image is likely (B, H, W) from torch.stack
            # For single env, squeeze batch dim; for multi-env, keep batch
            if visual_obs.dim() == 3:
                # Check if it's (B, H, W) or (H, W, C)
                # If last dim is not in [1,2,3] and first dim < last dim, it's likely (B, H, W)
                if visual_obs.shape[-1] not in [1, 2, 3] and visual_obs.shape[0] < visual_obs.shape[-1]:
                    # Likely (B, H, W) - keep as is for batch processing
                    pass
                elif visual_obs.shape[0] == 1 and visual_obs.shape[-1] not in [1, 2, 3]:
                    # (1, H, W) -> squeeze to (H, W)
                    visual_obs = visual_obs.squeeze(0)
        elif self.visual_input_type == 'depth_seg':
            # Combine depth and segmentation
            if normalized_depth is None or forward_mask_image is None:
                return None, None, None
            visual_obs = torch.stack([normalized_depth.squeeze(-1), forward_mask_image.squeeze(-1)], dim=-1)
            self.forward_seg_depth_original = (normalized_depth * forward_mask_image).clone()
        else:
            raise ValueError(f"Unknown visual_input_type: {self.visual_input_type}")
        
        # Resize for network input
        if self.cfg["sensor"].get("resized_resolution", None):
            # Handle different input dimensions
            if visual_obs.dim() == 2:  # (H, W) - single channel image
                # Add channel dimension -> resize
                visual_obs = visual_obs.unsqueeze(-1)  # (H, W, 1)
                visual_obs = visual_obs.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
                visual_obs = self.resize_transform(visual_obs).squeeze(0).permute(1, 2, 0)  # (H_new, W_new, 1)
            elif visual_obs.dim() == 3:  # (H, W, C) or (B, H, W)
                if visual_obs.shape[-1] in [1, 2, 3]:  # Has channel dimension (H, W, C)
                    # (H, W, C) -> (C, H, W) -> resize -> (C, H_new, W_new) -> (H_new, W_new, C)
                    C = visual_obs.shape[-1]
                    visual_obs = visual_obs.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                    visual_obs_resized = self.resize_transform(visual_obs)  # (1, C, H_new, W_new)
                    visual_obs = visual_obs_resized.squeeze(0).permute(1, 2, 0)  # (H_new, W_new, C)
                else:
                    # (B, H, W) - batch of single channel images
                    B, H, W = visual_obs.shape
                    visual_obs = visual_obs.unsqueeze(1)  # (B, 1, H, W)
                    visual_obs_resized = self.resize_transform(visual_obs)  # (B, 1, H_new, W_new)
                    visual_obs = visual_obs_resized.squeeze(1)  # (B, H_new, W_new)
                    # Add channel dimension for consistency: (B, H_new, W_new) -> (B, H_new, W_new, 1)
                    visual_obs = visual_obs.unsqueeze(-1)  # (B, H_new, W_new, 1)
            elif visual_obs.dim() == 4:  # (B, H, W, C) or (B, C, H, W)
                if visual_obs.shape[1] in [1, 2, 3] and visual_obs.shape[-1] not in [1, 2, 3]:
                    # (B, C, H, W) format
                    visual_obs_resized = self.resize_transform(visual_obs)  # (B, C, H_new, W_new)
                    visual_obs = visual_obs_resized.permute(0, 2, 3, 1)  # (B, H_new, W_new, C)
                else:
                    # (B, H, W, C) format
                    B, H, W, C = visual_obs.shape
                    visual_obs_reshaped = visual_obs.view(B * C, H, W).unsqueeze(1)  # (B*C, 1, H, W)
                    visual_obs_resized = self.resize_transform(visual_obs_reshaped)  # (B*C, 1, H_new, W_new)
                    H_new, W_new = visual_obs_resized.shape[2], visual_obs_resized.shape[3]
                    visual_obs = visual_obs_resized.view(B, C, H_new, W_new).permute(0, 2, 3, 1)  # (B, H_new, W_new, C)
        
        # Ensure visual_obs has batch dimension
        if visual_obs.dim() == 3:  # (H, W, C) -> add batch dim
            visual_obs = visual_obs.unsqueeze(0)  # (1, H, W, C)
        
        # Flatten for concatenation with other observations
        visual_obs_flat = visual_obs.flatten(start_dim=1)
        
        # Return format: (visual_obs, depth_obs, seg_obs) for compatibility
        return (visual_obs_flat, 
                normalized_depth.flatten(start_dim=1) if normalized_depth is not None else None,
                forward_mask_image.flatten(start_dim=1) if forward_mask_image is not None else None)

    def _save_camera_images(self):
        """Save RGB, depth, and segmentation images for a chosen env id."""
        if not self.cam_visualize or cv2 is None:
            return
        env_i = min(max(self.cam_vis_env, 0), self.num_envs - 1)
        
        # Original resolution from config
        W_orig = self.cfg["sensor"]["onboard_camera"]["resolution"][0]
        H_orig = self.cfg["sensor"]["onboard_camera"]["resolution"][1]
        
        # Color image - original resolution
        try:
            if len(self.camera_sensor_dict.get("forward_color", [])) > env_i and self.camera_sensor_dict["forward_color"][env_i] is not None:
                color = self.camera_sensor_dict["forward_color"][env_i].clone().detach().cpu().numpy()
                color = color.reshape(H_orig, W_orig, 4)[:, :, :3]
                bgr = color[:, :, ::-1]
                save_path = os.path.join(self.cam_output_dir, f"color_step{int(self.global_step_counter)}.png")
                # cv2.imwrite(save_path, bgr)
                # print(f'✓ Saved color image: {save_path}')
        except Exception as e:
            print(f'✗ Failed to save color image: {e}')

        # Depth image - original resolution (使用未resize的版本)
        try:
            if hasattr(self, 'depth_image_original') and self.depth_image_original is not None:
                depth = self.depth_image_original[env_i].clone().detach().cpu().numpy().reshape(H_orig, W_orig)
                depth_img = (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
                save_path = os.path.join(self.cam_output_dir, f"depth_step{int(self.global_step_counter)}.png")
                # cv2.im(save_path, depth_img)
                # print(f'✓ Saved writedepth image: {save_path} (original resolution {W_orig}x{H_orig})')
        except Exception as e:
            print(f'✗ Failed to save depth image: {e}')

        # Segmentation image - original resolution (显示所有ID，不只是mask)
        try:
            if len(self.camera_sensor_dict.get("forward_seg", [])) > env_i and self.camera_sensor_dict["forward_seg"][env_i] is not None:
                seg_raw = self.camera_sensor_dict["forward_seg"][env_i].clone().detach().cpu().numpy().reshape(H_orig, W_orig)
                # Normalize segmentation IDs to 0-255 for visualization
                # Map: 0=ground(black), 1=humanoid(gray), 2+=objects(various colors)
                seg_vis = np.zeros_like(seg_raw, dtype=np.uint8)
                seg_vis[seg_raw == 0] = 0       # Ground = black
                seg_vis[seg_raw == 1] = 85      # Humanoid = dark gray
                seg_vis[seg_raw >= 2] = ((seg_raw[seg_raw >= 2] - 2) * 40 + 120) % 255  # Objects = various
                # save_path = os.path.join(self.cam_output_dir, f"seg_step{int(self.global_step_counter)}.png")
                # cv2.imwrite(save_path, seg_vis)
                # print(f'✓ Saved segmentation image: {save_path} (original resolution {W_orig}x{H_orig})')
                # print(f'  Seg IDs present: {np.unique(seg_raw)} (0=ground, 1=humanoid, 2+=objects)')
        except Exception as e:
            print(f'✗ Failed to save segmentation image: {e}')
        
        # Object mask - original resolution (binary mask for objects only)
        try:
            if hasattr(self, 'forward_mask_original') and self.forward_mask_original is not None:
                mask = self.forward_mask_original[env_i].clone().detach().cpu().numpy().reshape(H_orig, W_orig)
                mask_img = (mask > 0.5).astype(np.uint8) * 255
                save_path = os.path.join(self.cam_output_dir, f"obj_mask_step{int(self.global_step_counter)}.png")
                # cv2.imwrite(save_path, mask_img)
                # print(f'✓ Saved object mask: {save_path} (original resolution {W_orig}x{H_orig})')
        except Exception as e:
            print(f'✗ Failed to save object mask: {e}')
      
    def obtain_imgs(self):
        """Obtain camera images"""
        if not self.enable_camera:
            return
            
        # If no valid camera handles/tensors, skip
        if not hasattr(self, 'camera_handles') or len(self.camera_handles) == 0:
            return
        if len(self.camera_sensor_dict.get("forward_depth", [])) == 0 or len(self.camera_sensor_dict.get("forward_seg", [])) == 0:
            return

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        result = self._get_front_camera_obs()
        if result is not None:
            self.depth_image_flat, self.forward_mask, self.forward_seg_depth = result
        # import ipdb; ipdb.set_trace()
        self.gym.end_access_image_tensors(self.sim)
        # try:
        #     result = self._get_front_camera_obs()
        #     if result is not None:
        #         self.depth_image_flat, self.forward_mask, self.forward_seg_depth = result
        # finally:
        #     self.gym.end_access_image_tensors(self.sim)
        
    def make_img_obs(self):
        """Create image observations for the network based on visual_input_type"""
        if not self.enable_camera or not hasattr(self, 'camera_history_buf'):
            return
        
        # Get visual observation based on visual_input_type
        result = self._get_front_camera_obs()
        if result is None:
            return
        
        visual_obs, depth_obs, seg_obs = result
        
        if visual_obs is None:
            return
        
        # visual_obs is already flattened, shape: (num_envs, img_size * num_channels)
        tensor_obs = visual_obs
        
        # Update camera history buffer
        self.camera_history_buf = torch.where(
            (self.progress_buf <= 1)[:, None, None],
            torch.stack([tensor_obs] * self.camera_history_len, dim=1),
            self.camera_history_buf
        )
        
        self.camera_history_buf = torch.cat([
            self.camera_history_buf[:, 1:],
            tensor_obs.unsqueeze(1)
        ], dim=1)
        
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        """Override to add camera creation"""
        # Call parent's _build_env first
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        # Add camera if enabled
        if self.enable_camera:
            camera_handle = self._create_onboard_camera(env_ptr, self.humanoid_handles[env_id], env_id)
            self.camera_handles.append(camera_handle)
            
    def create_sim(self):
        """Override to setup camera tensors after sim creation"""
        # Ensure graphics are available for camera sensors in headless mode
        if self.enable_camera:
            # If BaseTask set graphics_device_id to -1 due to headless, override to enable camera rendering
            try:
                if getattr(self, 'graphics_device_id', None) == -1:
                    self.graphics_device_id = self.device_id
            except Exception:
                pass

        super().create_sim()
        
        # Setup camera tensors after all environments are created
        if self.enable_camera:
            self._setup_camera_tensors()
            
    def __del__(self):
        """Cleanup visualization thread on destruction"""
        self._cleanup_visualization()
    
    def _cleanup_visualization(self):
        """Clean up visualization resources"""
        if self.enable_realtime_vis and self.vis_thread is not None:
            if self.vis_stop_event is not None:
                self.vis_stop_event.set()
            if self.vis_queue is not None:
                try:
                    self.vis_queue.put_nowait(None)  # Shutdown signal
                except:
                    pass
            if self.vis_thread.is_alive():
                self.vis_thread.join(timeout=1.0)
            if cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            
    def _vis_thread_worker(self):
        """Worker thread for real-time visualization"""
        if cv2 is None:
            return
        
        window_name = "Camera Observations (Env {})".format(self.vis_env_id)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while not self.vis_stop_event.is_set():
            try:
                # Get images from queue with timeout
                images = self.vis_queue.get(timeout=0.1)
                if images is None:  # Shutdown signal
                    break
                
                depth_img, mask_img, seg_depth_img = images
                
                # Resize for display
                H, W = depth_img.shape[:2]
                display_H, display_W = H * self.vis_scale, W * self.vis_scale
                
                depth_display = cv2.resize(depth_img, (display_W, display_H), interpolation=cv2.INTER_NEAREST)
                mask_display = cv2.resize(mask_img, (display_W, display_H), interpolation=cv2.INTER_NEAREST)
                seg_depth_display = cv2.resize(seg_depth_img, (display_W, display_H), interpolation=cv2.INTER_NEAREST)
                
                # Convert to color for better visualization
                depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
                seg_depth_colored = cv2.applyColorMap(seg_depth_display, cv2.COLORMAP_JET)
                
                # Combine images horizontally
                combined = np.hstack([depth_colored, mask_colored, seg_depth_colored])
                
                # Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined, "Depth", (10, 30), font, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, "Mask", (display_W + 10, 30), font, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, "Seg+Depth", (display_W * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, combined)
                cv2.waitKey(1)  # Non-blocking wait
                
            except Exception as e:
                # Ignore timeout and other errors
                pass
        
        cv2.destroyWindow(window_name)
    
    def _update_realtime_vis(self):
        """Update real-time visualization with current camera images"""
        if not self.enable_realtime_vis or cv2 is None:
            return
        
        if not hasattr(self, 'depth_image_original') or self.depth_image_original is None:
            return
        
        # Check if enough time has passed (rate limiting)
        current_time = time.time()
        min_interval = 1.0 / self.vis_fps
        if current_time - self.last_vis_time < min_interval:
            return
        
        self.last_vis_time = current_time
        
        # Get images for the specified environment
        env_id = min(max(self.vis_env_id, 0), self.num_envs - 1)
        
        try:
            # Get original resolution images
            W_orig = self.cfg["sensor"]["onboard_camera"]["resolution"][0]
            H_orig = self.cfg["sensor"]["onboard_camera"]["resolution"][1]
            
            # Depth image
            depth = self.depth_image_original[env_id].clone().detach().cpu().numpy().reshape(H_orig, W_orig)
            depth_img = (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
            
            # Mask image
            if hasattr(self, 'forward_mask_original') and self.forward_mask_original is not None:
                mask = self.forward_mask_original[env_id].clone().detach().cpu().numpy().reshape(H_orig, W_orig)
                mask_img = (mask * 255).astype(np.uint8)
            else:
                mask_img = np.zeros((H_orig, W_orig), dtype=np.uint8)
            
            # Seg+Depth image
            if hasattr(self, 'forward_seg_depth_original') and self.forward_seg_depth_original is not None:
                seg_depth = self.forward_seg_depth_original[env_id].clone().detach().cpu().numpy().reshape(H_orig, W_orig)
                seg_depth_img = (np.clip(seg_depth, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                seg_depth_img = np.zeros((H_orig, W_orig), dtype=np.uint8)
            
            # Put images in queue (non-blocking, drop if queue is full)
            if self.vis_queue is not None:
                try:
                    self.vis_queue.put_nowait((depth_img, mask_img, seg_depth_img))
                except:
                    pass  # Queue full, skip this frame
                    
        except Exception as e:
            # Silently ignore errors to avoid disrupting simulation
            pass
    
    def post_physics_step(self):
        """Override post physics step to include camera processing"""
        super().post_physics_step()
        
        if self.enable_camera:
            # Only obtain images if we have camera handles
            if hasattr(self, 'camera_handles') and len(self.camera_handles) > 0:
                self.obtain_imgs()
            self.make_img_obs()
            # Optional visualization every N steps
            # import ipdb; ipdb.set_trace()
            if self.cam_visualize and (int(getattr(self, 'global_step_counter', 0)) % self.cam_visualize_stride == 0):
                self._save_camera_images()
            
            # Update real-time visualization
            if self.enable_realtime_vis:
                self._update_realtime_vis()
