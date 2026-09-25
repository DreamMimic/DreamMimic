# Copyright (c) 2018-2022, NVIDIA Corporation
# G1 No-reference visual distillation environment with World Model support
# G1-specific implementation to avoid affecting SMPLX training

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

from env.tasks.intermimic_g1 import InterMimicG1
from env.tasks.intermimic import InterMimic

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from rl_games.algos_torch import torch_ext
from learning import intermimic_network_builder, intermimic_models_teacher
from torch.func import vmap
from functorch import make_functional
import yaml
import os

def get_all_paths(dir_path):
    paths = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            paths.append(os.path.join(root, name))
    return paths


class InterMimicG1_NoRef_Vis_WM(InterMimicG1):
    """G1 No-reference visual distillation environment with World Model support"""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Keep both teacher and student observation sizes distinct
        self.global_step_counter = 0
        student_obs_size = cfg["env"]["numObs"]
        teacher_obs_size = cfg["env"].get("teacherObsSize", student_obs_size)
        # Whether to drop privileged object info from student obs
        self.remove_privileged_obs = bool(cfg.get("env", {}).get("removePrivilegedObs", False))
        
        # Camera configuration - initialize early
        self.enable_camera = cfg.get("sensor", {}).get("enableCamera", False)
        self.camera_mode = "front_only"  # Simple front camera only, same as SMPLX
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
        self.cam_output_dir = sensor_cfg.get("output_dir", "camera_vis_g1")
        
        # Real-time visualization settings
        self.enable_realtime_vis = bool(sensor_cfg.get("enable_realtime_vis", False))
        self.vis_fps = float(sensor_cfg.get("vis_fps", 5.0))
        self.vis_env_id = int(sensor_cfg.get("vis_env_id", 0))
        self.vis_scale = int(sensor_cfg.get("vis_scale", 4))
        
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
            self.vis_queue = Queue(maxsize=2)
            self.vis_stop_event = threading.Event()
            self.vis_thread = threading.Thread(target=self._vis_thread_worker, daemon=True)
            self.vis_thread.start()
        
        if self.enable_camera:
            self.resize_transform = transforms.Resize((cfg["sensor"]["resized_resolution"][1], cfg["sensor"]["resized_resolution"][0]))

        # Temporarily switch to teacher obs size to build and load teacher models in parent init
        cfg["env"]["numObs"] = teacher_obs_size
        cfg["env"]["numObservations"] = teacher_obs_size
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        # Allocate distinct buffers: teacher (full obs) and student (no-ref)
        self.teacher_obs_size = teacher_obs_size
        self.obs_buf_teacher = torch.zeros((self.num_envs, self.teacher_obs_size), device=self.device, dtype=torch.float)
        
        # Calculate student obs size based on whether camera is enabled and privileged removal
        # Derive base current-observation length from kinematic sizes (G1 specific)
        num_bodies = int(self._rigid_body_pos.shape[1])
        dof_dim = int(self._dof_pos.shape[1])  # G1: 43
        ts_dim = int(self._target_states.shape[1]) if hasattr(self, '_target_states') else 13
        # G1 build_hoi_observations layout: root(7) + dof_pos(43) + dof_vel(43) + body_pos(52*3) + body_rot(52*4) + 
        # body_vel(52*3) + body_rot_vel(52*3) + target_states(13) + ig(52*3) + contact(52) + target_contact(1)
        base_full_len = 7 + 2 * dof_dim + 17 * num_bodies + ts_dim
        # Remove privileged: target_states(13) + ig(3*num_bodies) + target_contact(1)
        base_no_priv = 7 + 2 * dof_dim + 14 * num_bodies
        base_len = base_no_priv if self.remove_privileged_obs else base_full_len

        if self.enable_camera:
            # Initialize camera history buffer
            self.camera_history_len = 3
            img_size = self.cfg["sensor"]["resized_resolution"][0] * self.cfg["sensor"]["resized_resolution"][1]
            num_channels = 2  # default depth_seg
            if self.visual_input_type == 'rgb':
                num_channels = 3
            elif self.visual_input_type == 'depth' or self.visual_input_type == 'segmentation':
                num_channels = 1
            elif self.visual_input_type == 'depth_seg':
                num_channels = 2
            
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
        self.num_obs = actual_student_obs_size
        self._num_obs = actual_student_obs_size
        try:
            self.cfg["env"]["numObservations"] = actual_student_obs_size
            self.cfg["env"]["numObs"] = actual_student_obs_size
        except Exception:
            pass
        
        # Initialize teacher models for distillation (similar to InterMimic_All)
        self.action_buf = torch.zeros((self.num_envs, self._num_actions), device=self.device, dtype=torch.float)
        self.mu_buf = torch.zeros((self.num_envs, self._num_actions), device=self.device, dtype=torch.float)
        self.obs_buf_retarget = torch.zeros((self.num_envs, cfg["env"]["numObsRetarget"]), device=self.device, dtype=torch.float)
        
        self.models = []
        self.functional_models = []
        self.params_list = []
        self.running_means = []
        self.running_vars = []
        self.models_subid = []
        
        obs_shape = teacher_obs_size
        config = {
            'actions_num': self._num_actions,  # G1: 43
            'input_shape': (obs_shape,),
            'num_seqs': cfg["env"]["numEnvs"] * 1,
            'value_size': 1,
        }
        network = intermimic_network_builder.InterMimicBuilder()
        with open(os.path.join(os.getcwd(), cfg["env"]["teacherPolicyCFG"]), 'r') as f:
            cfg_teacher = yaml.load(f, Loader=yaml.SafeLoader)
        network.load(cfg_teacher['params']['network'])
        network = intermimic_models_teacher.ModelInterMimicContinuous(network)
        teacher_policy = cfg["env"]["teacherPolicy"]
        models_path = get_all_paths(teacher_policy)
        
        for model_path in models_path:
            subid = int(model_path.split('/')[-1].split('.')[0][3:])
            ck = torch_ext.load_checkpoint(model_path)
            model = network.build(config)
            model.to(self.device)
            model.load_state_dict(ck['model'])
            self.models.append(model)
            f_model, params = make_functional(model)
            self.functional_models.append(f_model)
            self.params_list.append(params)
            running_mean, running_var = ck['running_mean_std']['running_mean'], ck['running_mean_std']['running_var']
            self.running_means.append(running_mean)
            self.running_vars.append(running_var)
            self.models_subid.append(subid)
        
        # Transpose list of parameter tuples into tuple of parameter lists
        self.params_zip = list(zip(*self.params_list))
        # Now stack along a new dimension (model dimension = 0)
        self.stacked_params = tuple(torch.stack(p_tensors, dim=0) for p_tensors in self.params_zip)
        self.running_means_all = torch.stack(self.running_means).float()
        self.running_vars_all = torch.stack(self.running_vars).float()
        
        # Initialize retarget data if needed
        if 'motion_file_retarget' in cfg['env']:
            self.motion_file_retarget = cfg['env']['motion_file_retarget']
            motion_file_retarget = os.listdir(self.motion_file_retarget)
            motion_file_retarget = sorted([data_path for data_path in motion_file_retarget if data_path.split('_')[0] in cfg['env']['dataSub']])
            self.motion_file_retarget = [os.path.join(self.motion_file_retarget, data_path) for data_path in motion_file_retarget]
            self.hoi_data_retarget = self._load_motion(self.motion_file_retarget, startk=1)
        
        return

    def _create_onboard_camera(self, env_handle, actor_handle, env_idx):
        """Create onboard camera for G1 robot - attach to d435_link"""
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = self.cfg["sensor"]["onboard_camera"]["resolution"][1]
        camera_props.width = self.cfg["sensor"]["onboard_camera"]["resolution"][0]
        
        if self.cfg["sensor"]["onboard_camera"].get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = self.cfg["sensor"]["onboard_camera"]["horizontal_fov"]
            
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        if camera_handle is None:
            return None
        
        # Set camera position and orientation relative to d435_link
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
        
        # G1: Attach camera to d435_link (as specified in config)
        attach_link_name = self.cfg["sensor"]["onboard_camera"].get("attach_link", "d435_link")
        d435_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, attach_link_name)
        
        # Fallback options if d435_link not found
        if d435_handle == -1:
            d435_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "mid360_link")
        if d435_handle == -1:
            d435_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "torso_link")
        if d435_handle == -1:
            d435_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "base_link")
        
        # Validate handles before attaching
        try:
            if camera_handle is not None and d435_handle != -1:
                self.gym.attach_camera_to_body(
                    camera_handle, env_handle, d435_handle, local_transform, gymapi.FOLLOW_TRANSFORM
                )
            else:
                return None
        except Exception as e:
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
        try:
            return int(self.cfg.get("sensor", {}).get("segmentation_id", 2))
        except Exception:
            return 2
        
    def _get_front_camera_obs(self):
        """Get front camera observations based on visual_input_type - G1 version
        Same implementation as SMPLX version to avoid blocking during reset
        """
        if not self.enable_camera:
            return None, None, None
        
        # Get RGB image if needed
        rgb_image = None
        if self.visual_input_type == 'rgb':
            if len(self.camera_sensor_dict.get("forward_color", [])) > 0:
                color_tensors = [t for t in self.camera_sensor_dict["forward_color"] if t is not None]
                if len(color_tensors) > 0:
                    rgb_image = torch.stack(color_tensors).to(self.device).clone()
                    if rgb_image.dim() == 1:
                        H, W = self.cfg["sensor"]["onboard_camera"]["resolution"][1], self.cfg["sensor"]["onboard_camera"]["resolution"][0]
                        rgb_image = rgb_image.view(H, W, -1)
                    elif rgb_image.dim() == 2:
                        rgb_image = rgb_image.unsqueeze(-1)
                        if rgb_image.shape[-1] == 1:
                            rgb_image = rgb_image.repeat(1, 1, 3)
                    if rgb_image.dim() == 3:
                        if rgb_image.shape[-1] == 4:
                            rgb_image = rgb_image[:, :, :3]
                        elif rgb_image.shape[-1] == 1:
                            rgb_image = rgb_image.repeat(1, 1, 3)
                        elif rgb_image.shape[-1] != 3:
                            if rgb_image.shape[-1] < 3:
                                padding = torch.zeros(rgb_image.shape[0], rgb_image.shape[1], 3 - rgb_image.shape[-1], 
                                                     device=rgb_image.device, dtype=rgb_image.dtype)
                                rgb_image = torch.cat([rgb_image, padding], dim=-1)
                            else:
                                rgb_image = rgb_image[:, :, :3]
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
            forward_mask = (seg_image >= seg_id)
            forward_mask_image = forward_mask.float()
            forward_mask_image[forward_mask_image > 0.5] = 1.
            forward_mask_image[forward_mask_image <= 0.5] = 0.
            if forward_mask_image.dim() == 3:
                if forward_mask_image.shape[0] < forward_mask_image.shape[-1] and forward_mask_image.shape[-1] not in [1, 2, 3]:
                    pass
                elif forward_mask_image.shape[-1] == 1:
                    forward_mask_image = forward_mask_image.squeeze(-1)
            elif forward_mask_image.dim() == 4:
                if forward_mask_image.shape[0] == 1:
                    forward_mask_image = forward_mask_image.squeeze(0)
        
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
            if visual_obs.dim() == 3:
                if visual_obs.shape[-1] not in [1, 2, 3] and visual_obs.shape[0] < visual_obs.shape[-1]:
                    pass
                elif visual_obs.shape[0] == 1 and visual_obs.shape[-1] not in [1, 2, 3]:
                    visual_obs = visual_obs.squeeze(0)
        elif self.visual_input_type == 'depth_seg':
            if normalized_depth is None or forward_mask_image is None:
                return None, None, None
            visual_obs = torch.stack([normalized_depth.squeeze(-1), forward_mask_image.squeeze(-1)], dim=-1)
            self.forward_seg_depth_original = (normalized_depth * forward_mask_image).clone()
        else:
            raise ValueError(f"Unknown visual_input_type: {self.visual_input_type}")
        
        # Resize for network input
        if self.cfg["sensor"].get("resized_resolution", None):
            if visual_obs.dim() == 2:
                visual_obs = visual_obs.unsqueeze(-1)
                visual_obs = visual_obs.permute(2, 0, 1).unsqueeze(0)
                visual_obs = self.resize_transform(visual_obs).squeeze(0).permute(1, 2, 0)
            elif visual_obs.dim() == 3:
                if visual_obs.shape[-1] in [1, 2, 3]:
                    C = visual_obs.shape[-1]
                    visual_obs = visual_obs.permute(2, 0, 1).unsqueeze(0)
                    visual_obs_resized = self.resize_transform(visual_obs)
                    visual_obs = visual_obs_resized.squeeze(0).permute(1, 2, 0)
                else:
                    B, H, W = visual_obs.shape
                    visual_obs = visual_obs.unsqueeze(1)
                    visual_obs_resized = self.resize_transform(visual_obs)
                    visual_obs = visual_obs_resized.squeeze(1)
                    visual_obs = visual_obs.unsqueeze(-1)
            elif visual_obs.dim() == 4:
                if visual_obs.shape[1] in [1, 2, 3] and visual_obs.shape[-1] not in [1, 2, 3]:
                    visual_obs_resized = self.resize_transform(visual_obs)
                    visual_obs = visual_obs_resized.permute(0, 2, 3, 1)
                else:
                    B, H, W, C = visual_obs.shape
                    visual_obs_reshaped = visual_obs.view(B * C, H, W).unsqueeze(1)
                    visual_obs_resized = self.resize_transform(visual_obs_reshaped)
                    H_new, W_new = visual_obs_resized.shape[2], visual_obs_resized.shape[3]
                    visual_obs = visual_obs_resized.view(B, C, H_new, W_new).permute(0, 2, 3, 1)
        
        # Ensure visual_obs has batch dimension
        if visual_obs.dim() == 3:
            visual_obs = visual_obs.unsqueeze(0)
        
        # Flatten for concatenation
        visual_obs_flat = visual_obs.flatten(start_dim=1)
        
        return (visual_obs_flat, 
                normalized_depth.flatten(start_dim=1) if normalized_depth is not None else None,
                forward_mask_image.flatten(start_dim=1) if forward_mask_image is not None else None)
    
    def _update_camera_history(self, visual_obs):
        """Update camera history buffer"""
        if not self.enable_camera or visual_obs is None:
            return
        
        # Shift history: move old frames forward
        self.camera_history_buf[:, :-1] = self.camera_history_buf[:, 1:].clone()
        # Add new frame at the end
        self.camera_history_buf[:, -1] = visual_obs
    
    def _compute_observations(self, env_ids=None):
        """Compute observations - G1 version with visual support"""
        # First compute full obs (teacher) via parent implementation into teacher buffer
        original_obs_buf = self.obs_buf
        self.obs_buf = self.obs_buf_teacher
        super()._compute_observations(env_ids)
        # After parent computation, teacher obs are in self.obs_buf_teacher
        self.obs_buf = original_obs_buf

        # Now compute no-ref student obs: use current-only observation (remove all reference terms)
        # We directly use the current HOI observation built in post_physics_step
        # NOTE: Camera images are updated in post_physics_step(), not here, to avoid blocking reset
        if env_ids is None:
            # Ensure current observation is up to date
            student_obs = self._curr_obs.clone()
            
            # Optionally remove privileged info
            if self.remove_privileged_obs:
                student_obs = self._strip_privileged_from_obs(student_obs)
            
            # Add visual information if camera is enabled
            # Use camera_history_buf directly (updated in post_physics_step), not _get_front_camera_obs()
            if self.enable_camera and hasattr(self, 'camera_history_buf'):
                # Flatten camera history and concatenate with student obs
                visual_obs_flat = self.camera_history_buf.view(self.num_envs, -1)
                student_obs = torch.cat([student_obs, visual_obs_flat], dim=-1)
            
            # Ensure the student obs buffer has the right size
            if student_obs.shape[1] != self.student_obs_buf.shape[1]:
                self.student_obs_buf = torch.zeros((self.num_envs, student_obs.shape[1]), device=self.device, dtype=torch.float)
                self.obs_buf = self.student_obs_buf
                
            self.student_obs_buf[:] = student_obs
            # Replace obs_buf with student obs (no-ref + visual) for student policy
            self.obs_buf = self.student_obs_buf
        else:
            student_obs = self._curr_obs[env_ids].clone()
            # Optionally remove privileged info
            if self.remove_privileged_obs:
                student_obs = self._strip_privileged_from_obs(student_obs)
            
            # Add visual information if camera is enabled
            # Use camera_history_buf directly (updated in post_physics_step)
            if self.enable_camera and hasattr(self, 'camera_history_buf'):
                # Flatten camera history and concatenate
                visual_obs_flat = self.camera_history_buf[env_ids].view(len(env_ids), -1)
                student_obs = torch.cat([student_obs, visual_obs_flat], dim=-1)
            
            # Ensure the student obs buffer has the right size
            if student_obs.shape[1] != self.student_obs_buf.shape[1]:
                self.student_obs_buf = torch.zeros((self.num_envs, student_obs.shape[1]), device=self.device, dtype=torch.float)
                self.obs_buf = self.student_obs_buf
                
            self.student_obs_buf[env_ids] = student_obs
            # Replace obs_buf with student obs (no-ref + visual) for student policy
            self.obs_buf[env_ids] = self.student_obs_buf[env_ids]
        return

    def _strip_privileged_from_obs(self, obs):
        """Strip privileged information from observations - G1 version"""
        # G1 privileged info: target_states, ig, target_contact
        # Calculate indices based on G1 observation structure
        num_bodies = int(self._rigid_body_pos.shape[1]) if hasattr(self, '_rigid_body_pos') else 52
        dof_dim = int(self._dof_pos.shape[1]) if hasattr(self, '_dof_pos') else 43
        ts_dim = 13
        ig_len = num_bodies * 3
        contact_len = num_bodies
        
        # Calculate indices
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
        
        # Check bounds
        total_dim = obs.shape[-1]
        if tar_contact_start + tar_contact_len > total_dim:
            return obs  # Return as-is if dimensions don't match
        
        # Remove privileged parts
        obs_stripped = torch.cat([
            obs[:, :ts_start],
            obs[:, tar_contact_start + tar_contact_len:]
        ], dim=-1)
        
        return obs_stripped
    
    def single_model_forward(self, params, obs, mean, var):
        """Single model forward pass for teacher models"""
        curr_obs = (obs - mean) / torch.sqrt(var + 1e-5)
        curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
        
        # Construct input_dict for the model
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': curr_obs,
            'rnn_states': None
        }
        
        # Run the functional model
        res_dict = self.functional_models[0](params, input_dict)
        mu = res_dict['mus']
        sigma = res_dict['log_std'].exp()
        return mu, sigma
    
    def _compute_observations_retarget(self, env_ids=None):
        """Compute retarget observations for DAgger wrappers"""
        if not hasattr(self, 'hoi_data_retarget'):
            return
        if env_ids is None:
            self._curr_ref_obs[:] = self.hoi_data_retarget[self.data_id, self.progress_buf].clone()
            obs_1 = self._compute_observations_iter(self.hoi_data_retarget, None, 1)
            obs_16 = self._compute_observations_iter(self.hoi_data_retarget, None, 16)
            self.obs_buf_retarget[:] = torch.cat((obs_1, obs_16), dim=-1)
        else:
            self._curr_ref_obs[env_ids] = self.hoi_data_retarget[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf_retarget[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data_retarget, env_ids, 1),
                                                         self._compute_observations_iter(self.hoi_data_retarget, env_ids, 16)), dim=-1)
        return

    def _compute_observations_retarget(self, env_ids=None):
        """Compute retarget observations for DAgger wrappers"""
        if not hasattr(self, 'hoi_data_retarget'):
            return
        if env_ids is None:
            self._curr_ref_obs[:] = self.hoi_data_retarget[self.data_id, self.progress_buf].clone()
            obs_1 = self._compute_observations_iter(self.hoi_data_retarget, None, 1)
            obs_16 = self._compute_observations_iter(self.hoi_data_retarget, None, 16)
            self.obs_buf_retarget[:] = torch.cat((obs_1, obs_16), dim=-1)
        else:
            self._curr_ref_obs[env_ids] = self.hoi_data_retarget[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf_retarget[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data_retarget, env_ids, 1), 
                                                         self._compute_observations_iter(self.hoi_data_retarget, env_ids, 16)), dim=-1)
        return
    
    def reset(self, env_ids=None):
        """Reset environments - G1 version"""
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
        try:
            self._compute_observations_retarget(env_ids)
        except Exception:
            pass
        
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
    
    def obtain_imgs(self):
        """Obtain camera images"""
        if not self.enable_camera:
            return
            
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
        self.gym.end_access_image_tensors(self.sim)
        
    def make_img_obs(self):
        """Create image observations for the network based on visual_input_type - G1 version"""
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
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if self.enable_camera:
            camera_handle = self._create_onboard_camera(env_ptr, self.humanoid_handles[env_id], env_id)
            self.camera_handles.append(camera_handle)
            
    def create_sim(self):
        """Override to setup camera tensors after sim creation"""
        if self.enable_camera:
            try:
                if getattr(self, 'graphics_device_id', None) == -1:
                    self.graphics_device_id = self.device_id
            except Exception:
                pass

        super().create_sim()
        
        if self.enable_camera:
            self._setup_camera_tensors()
    
    def post_physics_step(self):
        """Override post physics step to include camera processing - G1 version"""
        self.progress_buf += 1
        
        self._refresh_sim_tensors()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._update_hist_hoi_obs()
        self._compute_hoi_observations()
        self._compute_observations(env_ids)
        self._compute_observations_retarget(env_ids)
        self._compute_reward(self.actions)
        self._compute_reset()
        
        if self.enable_camera:
            # Only obtain images if we have camera handles
            if hasattr(self, 'camera_handles') and len(self.camera_handles) > 0:
                self.obtain_imgs()
            self.make_img_obs()
            # Optional visualization every N steps
            if self.cam_visualize and (int(getattr(self, 'global_step_counter', 0)) % self.cam_visualize_stride == 0):
                self._save_camera_images()
        
        self.extras["terminate"] = self._terminate_buf
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self._dof_vel[:]
        self.last_root_vel[:] = self._humanoid_root_states[:, 7:13]
        self.last_root_pos[:] = self._humanoid_root_states[:, 0:3]
        
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self._humanoid_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self._rigid_body_ang_vel[:, 11, :])
        self.projected_gravity[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self.gravity_vec)
        
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        
        return
    
    def _save_camera_images(self):
        """Save camera images for visualization"""
        if not self.cam_visualize or cv2 is None:
            return
        # Implementation similar to SMPLX version
        pass
