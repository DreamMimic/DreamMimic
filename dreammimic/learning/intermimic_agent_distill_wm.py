# Copyright (c) 2018-2022, NVIDIA Corporation
# Agent with World Model integration for InterMimic distillation

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from isaacgym.torch_utils import *
import numpy as np
import torch 
from torch import nn
import collections

import learning.intermimic_agent_distill as intermimic_agent_distill
from .world_model import WorldModel


class InterMimicAgentDistillWM(intermimic_agent_distill.InterMimicAgentDistill):
    """Agent with World Model integration for visual distillation"""
    
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        
        # World Model configuration
        self.use_world_model = config.get('use_world_model', False)
        self.wm_config = config.get('world_model_config', {})
        
        # Merge with network.world_model_config if available (network builder has encoder config)
        # This ensures encoder config (visual_input_type, etc.) is available
        # Note: self.model is created in super().__init__(), so it's available now
        if hasattr(self, 'network') and hasattr(self.network, 'params'):
            network_wm_config = self.network.params.get('world_model_config', {})
            if network_wm_config:
                # Merge: config takes precedence for training params, but use network for encoder config
                import copy
                merged_config = copy.deepcopy(self.wm_config)
                # Merge encoder config from network if not in config
                if 'encoder' not in merged_config and 'encoder' in network_wm_config:
                    merged_config['encoder'] = network_wm_config['encoder']
                # Also merge other architecture configs if missing
                for key in ['enable_multi_to_privi', 'multi_history_len', 'decoder', 'privileged_head', 'privi_dim']:
                    if key not in merged_config and key in network_wm_config:
                        merged_config[key] = network_wm_config[key]
                self.wm_config = merged_config
        
        # Enable multi-step to privileged mode
        self.enable_multi_to_privi = self.wm_config.get('enable_multi_to_privi', False)
        self.multi_history_len = self.wm_config.get('multi_history_len', 3)
        
        # Ensure numeric types for training parameters
        self.wm_train_interval = int(config.get('wm_train_interval', 10))
        self.wm_train_steps = int(config.get('wm_train_steps', 1))
        self.wm_batch_size = int(config.get('wm_batch_size', 16))
        self.wm_batch_length = int(config.get('wm_batch_length', 64))
        
        # History buffer for multi-step observations (if enable_multi_to_privi)
        # Use larger maxlen to store enough history across episodes
        # We'll extract the last multi_history_len steps when needed
        if self.enable_multi_to_privi:
            # Store more than multi_history_len to handle episode boundaries
            # maxlen should be at least multi_history_len * 2 to ensure we have enough data
            buffer_maxlen = max(self.multi_history_len * 2, 100)
            self.wm_history_buffer = collections.deque(maxlen=buffer_maxlen)
            self.wm_privi_buffer = collections.deque(maxlen=buffer_maxlen)
        
        # Ensure numeric types in wm_config
        for key in ['dyn_stoch', 'dyn_deter', 'dyn_hidden', 'dyn_rec_depth', 'dyn_discrete', 
                    'dyn_min_std', 'num_actions', 'prop_dim', 'model_lr', 'opt_eps', 
                    'grad_clip', 'weight_decay', 'dyn_scale', 'rep_scale', 'kl_free']:
            if key in self.wm_config and isinstance(self.wm_config[key], str):
                try:
                    if 'e' in str(self.wm_config[key]) or '.' in str(self.wm_config[key]):
                        self.wm_config[key] = float(self.wm_config[key])
                    else:
                        self.wm_config[key] = int(self.wm_config[key])
                except (ValueError, TypeError):
                    pass
        
        # World Model instance
        self.world_model = None
        if self.use_world_model:
            # Try to reuse World Model from network if available (to avoid duplicate initialization)
            # Network builder creates World Model during build(), we should reuse it
            if hasattr(self, 'model') and hasattr(self.model, 'a2c_network'):
                network_wm = getattr(self.model.a2c_network, 'world_model', None)
                if network_wm is not None:
                    # Reuse the World Model from network
                    self.world_model = network_wm
                    # Get visual input info from reused World Model
                    encoder_config = self.wm_config.get('encoder', {})
                    visual_backbone = encoder_config.get('visual_backbone', 'cnn')
                    visual_input_type = getattr(network_wm, 'visual_input_type', encoder_config.get('visual_input_type', 'depth_seg'))
                    visual_channels = getattr(network_wm, 'visual_channels', 2)
                    
                    print("\n" + "=" * 80)
                    print("Reusing World Model from Network Builder (to avoid duplicate initialization):")
                    print(f"  Encoder Type: {visual_backbone.upper()}")
                    print(f"  Visual Input Type: {visual_input_type.upper()}")
                    print(f"  Visual Channels: {visual_channels}")
                    print("=" * 80 + "\n")
                    
                    # Set privileged dimension if enable_multi_to_privi
                    if self.enable_multi_to_privi:
                        self._privi_dim = None
                    # World Model data buffer
                    self.wm_buffer = collections.deque(maxlen=10000)
                    self.wm_buffer_size = 0
                    # World Model training counter
                    self.wm_train_counter = 0
                    return
            
            # Otherwise, create new World Model instance
            # Get visual input type from encoder config
            encoder_config = self.wm_config.get('encoder', {})
            visual_input_type = encoder_config.get('visual_input_type', 'depth_seg')
            
            # Determine number of channels based on visual_input_type
            if visual_input_type == 'rgb':
                visual_channels = 3
            elif visual_input_type == 'depth' or visual_input_type == 'segmentation':
                visual_channels = 1
            elif visual_input_type == 'depth_seg':
                visual_channels = 2
            else:
                visual_channels = 2  # default
            
            obs_shape = {
                'prop': (self.wm_config.get('prop_dim', 64),),
                'image': (32, 32, visual_channels)
            }
            print("\n" + "=" * 80)
            print("Initializing World Model for Training/Testing:")
            visual_backbone = encoder_config.get('visual_backbone', 'cnn')
            # visual_input_type already read above at line 64, don't override it!
            print(f"  Encoder Type: {visual_backbone.upper()}")
            print(f"  Visual Input Type: {visual_input_type.upper()}")
            print(f"  Visual Channels: {visual_channels}")
            print(f"  Visual Observation Shape: {obs_shape['image']}")
            print(f"  Proprioceptive Dim: {obs_shape['prop'][0]}")
            print("=" * 80 + "\n")
            
            self.world_model = WorldModel(
                self.wm_config,
                obs_shape,
                use_camera=True,
                device=self.ppo_device
            )
            self.world_model.to(self.ppo_device)
            
            # Set privileged dimension if enable_multi_to_privi
            if self.enable_multi_to_privi:
                # Will be set dynamically when we collect first privileged data
                self._privi_dim = None
        
        # World Model data buffer
        self.wm_buffer = collections.deque(maxlen=10000)
        self.wm_buffer_size = 0
        
        # World Model training counter
        self.wm_train_counter = 0
        
        return
    
    def init_tensors(self):
        super().init_tensors()
        # Add world model related tensors if needed
        return
    
    def play_steps(self):
        """Play steps with world model data collection"""
        self.set_eval()
        
        epinfos = []
        update_list = self.update_list
        
        # Initialize DAgger beta coefficient
        beta_t = max(1 - max((self.epoch_num - 500) / 5000, 0), 0)
        
        for n in range(self.horizon_length):
            self.obs, self.expert = self.env_reset(self.done_indices)
            
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs, beta_t, self.expert['actions'].to(self.ppo_device))
            
            self.experience_buffer.update_data('expert', n, self.expert['mus'].to(self.ppo_device))
            
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            
            self.obs, rewards, self.dones, infos, self.expert = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])
            
            # Collect world model data
            if self.use_world_model and self.world_model is not None:
                # Get teacher obs (with privileged info) if enable_multi_to_privi
                teacher_obs = None
                if self.enable_multi_to_privi and hasattr(self.vec_env.env.task, 'obs_buf_teacher'):
                    teacher_obs = self.vec_env.env.task.obs_buf_teacher
                
                self._collect_wm_data(self.obs['obs'], res_dict['actions'], rewards, self.dones, teacher_obs)
                
                # Reset history buffers when episodes end (for multi-step mode)
                # Note: History buffers are global, so we clear them when any episode ends
                # The is_first flag in collected data will handle episode boundaries in training
                if self.enable_multi_to_privi and torch.any(self.dones):
                    # Clear history buffers to start fresh for new episodes
                    # This ensures clean history for the next episode
                    self.wm_history_buffer.clear()
                    self.wm_privi_buffer.clear()
                    if hasattr(self, 'wm_contact_buffer'):
                        self.wm_contact_buffer.clear()
                    if hasattr(self, 'wm_object_state_buffer'):
                        self.wm_object_state_buffer.clear()
            
            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            self.done_indices = all_done_indices[::self.num_agents]
            
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
        
        # Train world model periodically
        if self.use_world_model and self.world_model is not None:
            self.wm_train_counter += 1
            if self.wm_train_counter >= self.wm_train_interval and len(self.wm_buffer) >= self.wm_batch_size:
                wm_metrics = self._train_world_model()
                if hasattr(self, 'writer') and self.writer is not None:
                    for name, value in wm_metrics.items():
                        self.writer.add_scalar(f'WorldModel/{name}', value, self.epoch_num)
                self.wm_train_counter = 0
        
        return batch_dict
    
    def _collect_wm_data(self, obs, actions, rewards, dones, teacher_obs=None):
        """Collect data for world model training"""
        # Skip if world model is not enabled or not initialized
        if not self.use_world_model or self.world_model is None:
            return
        
        # Extract visual and proprioceptive observations
        batch_size = obs.shape[0]
        obs_dim = obs.shape[-1]
        
        # Calculate visual obs size: 32*32*2*3 = 6144
        # Structure: camera_history_buf is (num_envs, 3, 32*32*2) = (num_envs, 3, 2048)
        # When flattened: (num_envs, 6144)
        # Student obs should be 7355 (1211 proprioceptive + 6144 visual)
        # Teacher obs is 3198 (no visual)
        expected_visual_obs_size = 32 * 32 * 2 * 3  # resized_resolution * channels * history_len
        single_frame_size = 32 * 32 * 2  # 2048
        
        # Check if this is student obs (with visual) or teacher obs (without visual)
        if obs_dim < expected_visual_obs_size:
            # This is teacher obs (3198) - no visual information, skip WM data collection
            return
        
        # This should be student obs - extract visual from the end
        visual_obs_start_idx = obs_dim - expected_visual_obs_size
        visual_obs_raw = obs[:, visual_obs_start_idx:]
        prop_obs = obs[:, :visual_obs_start_idx]
        
        # Reshape visual obs to (B, 32, 32, 2)
        actual_visual_size = visual_obs_raw.shape[-1]
        try:
            if actual_visual_size == expected_visual_obs_size:
                # Standard case: 6144 = 3 frames, use most recent
                visual_obs_raw = visual_obs_raw.view(batch_size, 3, single_frame_size)
                visual_obs_raw = visual_obs_raw.view(batch_size, 3, 32, 32, 2)
                visual_obs = visual_obs_raw[:, -1, :, :, :]  # (B, 32, 32, 2)
            elif actual_visual_size == single_frame_size:
                visual_obs = visual_obs_raw.view(batch_size, 32, 32, 2)
            elif actual_visual_size == single_frame_size * 2:
                visual_obs_raw = visual_obs_raw.view(batch_size, 2, 32, 32, 2)
                visual_obs = visual_obs_raw[:, -1, :, :, :]
            else:
                if actual_visual_size % single_frame_size == 0:
                    num_frames = actual_visual_size // single_frame_size
                    visual_obs_raw = visual_obs_raw.view(batch_size, num_frames, 32, 32, 2)
                    visual_obs = visual_obs_raw[:, -1, :, :, :]
                else:
                    return  # Skip if cannot reshape
        except RuntimeError:
            return  # Skip if reshape fails
        
        # Ensure shape is (B, 32, 32, 2)
        if visual_obs.shape != (batch_size, 32, 32, 2):
            try:
                visual_obs = visual_obs.view(batch_size, 32, 32, 2)
            except RuntimeError:
                return
        
        # Ensure prop_obs has enough dimensions
        prop_dim = self.wm_config.get('prop_dim', 64)
        if prop_obs.shape[-1] < prop_dim:
            pad_size = prop_dim - prop_obs.shape[-1]
            prop_obs_padded = torch.cat([
                prop_obs,
                torch.zeros(batch_size, pad_size, device=prop_obs.device)
            ], dim=-1)
        else:
            prop_obs_padded = prop_obs[:, :prop_dim]
        
        # Extract privileged information from environment if available
        privileged_info = None
        contact_info = None
        object_state_info = None
        if self.enable_multi_to_privi:
            privileged_info = self._extract_privileged_info(teacher_obs, prop_obs)
        
        # Extract contact and object_state separately if enabled
        use_contact_head = self.wm_config.get('use_contact_head', False)
        use_object_state_head = self.wm_config.get('use_object_state_head', False)
        if use_contact_head or use_object_state_head:
            contact_info, object_state_info = self._extract_contact_and_object_state(prop_obs)
        
        # For multi-step mode: collect history
        if self.enable_multi_to_privi:
            # Store current step in history buffer
            self.wm_history_buffer.append({
                'image': visual_obs.cpu(),
                'prop': prop_obs_padded.cpu(),
            })
            if privileged_info is not None:
                self.wm_privi_buffer.append(privileged_info.cpu())
            if use_contact_head and contact_info is not None:
                if not hasattr(self, 'wm_contact_buffer'):
                    self.wm_contact_buffer = collections.deque(maxlen=1000)
                self.wm_contact_buffer.append(contact_info.cpu())
            if use_object_state_head and object_state_info is not None:
                if not hasattr(self, 'wm_object_state_buffer'):
                    self.wm_object_state_buffer = collections.deque(maxlen=1000)
                self.wm_object_state_buffer.append(object_state_info.cpu())
            
            # Only store in main buffer when we have enough history
            if len(self.wm_history_buffer) >= self.multi_history_len:
                # Collect multi-step history
                history_images = [h['image'] for h in list(self.wm_history_buffer)[-self.multi_history_len:]]
                history_props = [h['prop'] for h in list(self.wm_history_buffer)[-self.multi_history_len:]]
                
                # Current step (last in history)
                current_image = history_images[-1]
                current_prop = history_props[-1]
                
                # Privileged info (if available)
                current_privi = None
                if len(self.wm_privi_buffer) >= self.multi_history_len:
                    current_privi = self.wm_privi_buffer[-1]
                
                # Contact and object_state info (if available)
                current_contact = None
                if use_contact_head and hasattr(self, 'wm_contact_buffer') and len(self.wm_contact_buffer) >= self.multi_history_len:
                    current_contact = self.wm_contact_buffer[-1]
                
                current_object_state = None
                if use_object_state_head and hasattr(self, 'wm_object_state_buffer') and len(self.wm_object_state_buffer) >= self.multi_history_len:
                    current_object_state = self.wm_object_state_buffer[-1]
                
                # Store in main buffer
                for i in range(batch_size):
                    buffer_entry = {
                        'image_history': [h[i] for h in history_images],  # List of (32, 32, 2)
                        'prop_history': [h[i] for h in history_props],   # List of (prop_dim,)
                        'image': current_image[i],
                        'prop': current_prop[i],
                        'privileged': current_privi[i] if current_privi is not None else None,
                        'action': actions[i].cpu(),
                        'reward': rewards[i].cpu() if rewards.dim() == 1 else rewards[i, 0].cpu(),
                        'is_first': dones[i].cpu() if dones.dim() == 1 else dones[i, 0].cpu(),
                    }
                    if use_contact_head and current_contact is not None:
                        buffer_entry['contact'] = current_contact[i]
                    if use_object_state_head and current_object_state is not None:
                        buffer_entry['object_state'] = current_object_state[i]
                    self.wm_buffer.append(buffer_entry)
                    self.wm_buffer_size += 1
        else:
            # Original single-step mode
            # Extract contact and object_state separately if enabled
            use_contact_head = self.wm_config.get('use_contact_head', False)
            use_object_state_head = self.wm_config.get('use_object_state_head', False)
            if use_contact_head or use_object_state_head:
                contact_info, object_state_info = self._extract_contact_and_object_state(prop_obs)
            
            # Store in buffer
            for i in range(batch_size):
                buffer_entry = {
                    'image': visual_obs[i].cpu(),
                    'prop': prop_obs_padded[i].cpu(),
                    'action': actions[i].cpu(),
                    'reward': rewards[i].cpu() if rewards.dim() == 1 else rewards[i, 0].cpu(),
                    'is_first': dones[i].cpu() if dones.dim() == 1 else dones[i, 0].cpu(),
                }
                if use_contact_head and contact_info is not None:
                    buffer_entry['contact'] = contact_info[i].cpu()
                if use_object_state_head and object_state_info is not None:
                    buffer_entry['object_state'] = object_state_info[i].cpu()
                self.wm_buffer.append(buffer_entry)
                self.wm_buffer_size += 1
    
    def _extract_privileged_info(self, teacher_obs, student_prop_obs):
        """Extract privileged information from current observation
        
        Args:
            teacher_obs: Teacher observation buffer (may be None)
            student_prop_obs: Student proprioceptive observation (for shape reference)
        
        Returns:
            privileged: (B, privi_dim) tensor with target_states, ig, target_contact
        """
        try:
            env_task = self.vec_env.env.task
            if not hasattr(env_task, '_curr_obs'):
                return None
            
            # Get current observation with privileged info
            curr_obs_full = env_task._curr_obs.clone()
            batch_size = curr_obs_full.shape[0]
            
            # Extract privileged parts: target_states, ig, target_contact
            # Based on _strip_privileged_from_obs logic
            num_bodies = int(env_task._rigid_body_pos.shape[1]) if hasattr(env_task, '_rigid_body_pos') else 50
            dof_dim = int(env_task._dof_pos.shape[1]) if hasattr(env_task, '_dof_pos') else 153
            ts_dim = 13  # target_states dimension
            ig_len = num_bodies * 3
            contact_len = num_bodies
            
            # Calculate indices (same as _strip_privileged_from_obs)
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
            total_dim = curr_obs_full.shape[1]
            if tar_contact_start + tar_contact_len > total_dim:
                return None
            
            # Extract privileged info
            target_states = curr_obs_full[:, ts_start:ts_start + ts_dim]
            ig = curr_obs_full[:, ig_start:ig_start + ig_len]
            target_contact = curr_obs_full[:, tar_contact_start:tar_contact_start + tar_contact_len]
            
            # Concatenate privileged info
            privileged = torch.cat([target_states, ig, target_contact], dim=-1)
            
            # Store dimension and update world model privileged head if needed
            if self._privi_dim is None or self._privi_dim != privileged.shape[-1]:
                self._privi_dim = privileged.shape[-1]
                # Update world model privileged head with correct dimension
                if hasattr(self.world_model, 'heads') and 'privileged' in self.world_model.heads:
                    privi_config = self.wm_config.get('privileged_head', {})
                    from dreamer import networks
                    self.world_model.heads['privileged'] = networks.MLP(
                        self.world_model.feat_size,
                        (self._privi_dim,),
                        privi_config.get('layers', 3),
                        self.wm_config.get('units', 512),
                        self.wm_config.get('act', 'SiLU'),
                        self.wm_config.get('norm', True),
                        dist=privi_config.get('dist', 'normal'),
                        outscale=privi_config.get('outscale', 1.0),
                        device=self.ppo_device,
                        name="Privileged",
                    )
                    # Update config for future reference
                    self.wm_config['privi_dim'] = self._privi_dim
            
            return privileged
        except Exception as e:
            if not hasattr(self, '_privi_extract_error_shown'):
                print(f"Warning: Failed to extract privileged info: {e}")
                import traceback
                traceback.print_exc()
                self._privi_extract_error_shown = True
            return None
    
    def _extract_contact_and_object_state(self, student_prop_obs):
        """Extract contact and object_state information separately from current observation
        
        Args:
            student_prop_obs: Student proprioceptive observation (for shape reference)
        
        Returns:
            contact: (B, 1) tensor with target_contact
            object_state: (B, 13) tensor with target_states
        """
        try:
            env_task = self.vec_env.env.task
            if not hasattr(env_task, '_curr_obs'):
                return None, None
            
            # Get current observation with privileged info
            curr_obs_full = env_task._curr_obs.clone()
            batch_size = curr_obs_full.shape[0]
            
            # Extract privileged parts: target_states, target_contact
            num_bodies = int(env_task._rigid_body_pos.shape[1]) if hasattr(env_task, '_rigid_body_pos') else 50
            dof_dim = int(env_task._dof_pos.shape[1]) if hasattr(env_task, '_dof_pos') else 153
            ts_dim = 13  # target_states dimension
            ig_len = num_bodies * 3
            contact_len = num_bodies
            
            # Calculate indices (same as _strip_privileged_from_obs logic)
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
            total_dim = curr_obs_full.shape[1]
            if tar_contact_start + tar_contact_len > total_dim:
                return None, None
            
            # Extract contact and object_state separately
            target_states = curr_obs_full[:, ts_start:ts_start + ts_dim]  # (B, 13)
            target_contact = curr_obs_full[:, tar_contact_start:tar_contact_start + tar_contact_len]  # (B, 1)
            
            # Update world model heads if needed
            use_contact_head = self.wm_config.get('use_contact_head', False)
            use_object_state_head = self.wm_config.get('use_object_state_head', False)
            
            if use_contact_head:
                contact_dim = target_contact.shape[-1]
                if not hasattr(self, '_contact_dim') or self._contact_dim != contact_dim:
                    self._contact_dim = contact_dim
                    if hasattr(self.world_model, 'heads') and 'contact' in self.world_model.heads:
                        contact_config = self.wm_config.get('contact_head', {})
                        from dreamer import networks
                        self.world_model.heads['contact'] = networks.MLP(
                            self.world_model.feat_size,
                            (self._contact_dim,),
                            contact_config.get('layers', 2),
                            self.wm_config.get('units', 512),
                            self.wm_config.get('act', 'SiLU'),
                            self.wm_config.get('norm', True),
                            dist=contact_config.get('dist', 'normal'),
                            outscale=contact_config.get('outscale', 1.0),
                            device=self.ppo_device,
                            name="Contact",
                        )
                        self.wm_config['contact_dim'] = self._contact_dim
            
            if use_object_state_head:
                object_state_dim = target_states.shape[-1]
                if not hasattr(self, '_object_state_dim') or self._object_state_dim != object_state_dim:
                    self._object_state_dim = object_state_dim
                    if hasattr(self.world_model, 'heads') and 'object_state' in self.world_model.heads:
                        object_state_config = self.wm_config.get('object_state_head', {})
                        from dreamer import networks
                        self.world_model.heads['object_state'] = networks.MLP(
                            self.world_model.feat_size,
                            (self._object_state_dim,),
                            object_state_config.get('layers', 3),
                            self.wm_config.get('units', 512),
                            self.wm_config.get('act', 'SiLU'),
                            self.wm_config.get('norm', True),
                            dist=object_state_config.get('dist', 'normal'),
                            outscale=object_state_config.get('outscale', 1.0),
                            device=self.ppo_device,
                            name="ObjectState",
                        )
                        self.wm_config['object_state_dim'] = self._object_state_dim
            
            return target_contact, target_states
        except Exception as e:
            if not hasattr(self, '_contact_obj_extract_error_shown'):
                print(f"Warning: Failed to extract contact/object_state info: {e}")
                import traceback
                traceback.print_exc()
                self._contact_obj_extract_error_shown = True
            return None, None
    def _train_world_model(self):
        """Train world model on collected data"""
        if len(self.wm_buffer) < self.wm_batch_size:
            return {}
        
        # Print encoder info on first training step
        if not hasattr(self, '_wm_train_info_printed'):
            encoder_config = self.wm_config.get('encoder', {})
            visual_backbone = encoder_config.get('visual_backbone', 'cnn')
            visual_input_type = encoder_config.get('visual_input_type', 'depth_seg')
            print("\n" + "=" * 80)
            print("World Model Training Started:")
            print(f"  Encoder Type: {visual_backbone.upper()}")
            print(f"  Visual Input Type: {visual_input_type.upper()}")
            print(f"  Buffer Size: {len(self.wm_buffer)}")
            print(f"  Batch Size: {self.wm_batch_size}")
            print("=" * 80 + "\n")
            self._wm_train_info_printed = True
        
        metrics = {}
        total_loss = 0.0
        
        for step in range(self.wm_train_steps):
            # Sample batch from buffer
            batch_indices = np.random.choice(len(self.wm_buffer), self.wm_batch_size, replace=False)
            
            if self.enable_multi_to_privi:
                # Multi-step mode: collect history and privileged info
                use_contact_head = self.wm_config.get('use_contact_head', False)
                use_object_state_head = self.wm_config.get('use_object_state_head', False)
                
                batch_data = {
                    'image_history': [],
                    'prop_history': [],
                    'image': [],
                    'prop': [],
                    'privileged': [],
                    'action': [],
                    'reward': [],
                    'is_first': []
                }
                if use_contact_head:
                    batch_data['contact'] = []
                if use_object_state_head:
                    batch_data['object_state'] = []
                
                # Collect batch
                for idx in batch_indices:
                    sample = self.wm_buffer[idx]
                    batch_data['image_history'].append(sample['image_history'])
                    batch_data['prop_history'].append(sample['prop_history'])
                    batch_data['image'].append(sample['image'])
                    batch_data['prop'].append(sample['prop'])
                    if sample.get('privileged') is not None:
                        batch_data['privileged'].append(sample['privileged'])
                    if use_contact_head and sample.get('contact') is not None:
                        batch_data['contact'].append(sample['contact'])
                    if use_object_state_head and sample.get('object_state') is not None:
                        batch_data['object_state'].append(sample['object_state'])
                    batch_data['action'].append(sample['action'])
                    batch_data['reward'].append(sample['reward'])
                    batch_data['is_first'].append(sample['is_first'])
                
                # Convert to tensors
                # Image history: List[List[tensor]] -> (B, T, H, W, C)
                image_history_list = []
                for hist_list in batch_data['image_history']:
                    # Stack history for one sample: (T, H, W, C)
                    hist_tensor = torch.stack(hist_list, dim=0)
                    image_history_list.append(hist_tensor)
                batch_data['image_history'] = torch.stack(image_history_list, dim=0).to(self.ppo_device)
                
                # Prop history: List[List[tensor]] -> (B, T, D)
                prop_history_list = []
                for hist_list in batch_data['prop_history']:
                    hist_tensor = torch.stack(hist_list, dim=0)
                    prop_history_list.append(hist_tensor)
                batch_data['prop_history'] = torch.stack(prop_history_list, dim=0).to(self.ppo_device)
                
                # Current step
                batch_data['image'] = torch.stack(batch_data['image']).to(self.ppo_device)
                batch_data['prop'] = torch.stack(batch_data['prop']).to(self.ppo_device)
                
                # Privileged info
                if len(batch_data['privileged']) > 0:
                    batch_data['privileged'] = torch.stack(batch_data['privileged']).to(self.ppo_device)
                else:
                    batch_data.pop('privileged', None)
                
                # Contact info
                if use_contact_head and len(batch_data.get('contact', [])) > 0:
                    batch_data['contact'] = torch.stack(batch_data['contact']).to(self.ppo_device)
                elif use_contact_head:
                    batch_data.pop('contact', None)
                
                # Object state info
                if use_object_state_head and len(batch_data.get('object_state', [])) > 0:
                    batch_data['object_state'] = torch.stack(batch_data['object_state']).to(self.ppo_device)
                elif use_object_state_head:
                    batch_data.pop('object_state', None)
                
                # Other fields
                batch_data['action'] = torch.stack(batch_data['action']).to(self.ppo_device)
                batch_data['reward'] = torch.stack(batch_data['reward']).to(self.ppo_device)
                batch_data['is_first'] = torch.stack(batch_data['is_first']).to(self.ppo_device)
            else:
                # Single-step mode
                use_contact_head = self.wm_config.get('use_contact_head', False)
                use_object_state_head = self.wm_config.get('use_object_state_head', False)
                
                batch_data = {
                    'image': [],
                    'prop': [],
                    'action': [],
                    'reward': [],
                    'is_first': []
                }
                if use_contact_head:
                    batch_data['contact'] = []
                if use_object_state_head:
                    batch_data['object_state'] = []
                
                # Collect batch
                for idx in batch_indices:
                    sample = self.wm_buffer[idx]
                    batch_data['image'].append(sample['image'])
                    batch_data['prop'].append(sample['prop'])
                    if use_contact_head and sample.get('contact') is not None:
                        batch_data['contact'].append(sample['contact'])
                    if use_object_state_head and sample.get('object_state') is not None:
                        batch_data['object_state'].append(sample['object_state'])
                    batch_data['action'].append(sample['action'])
                    batch_data['reward'].append(sample['reward'])
                    batch_data['is_first'].append(sample['is_first'])
                
                # Convert to tensors
                batch_data_tensors = {
                    'image': torch.stack(batch_data['image']).to(self.ppo_device),
                    'prop': torch.stack(batch_data['prop']).to(self.ppo_device),
                    'action': torch.stack(batch_data['action']).to(self.ppo_device),
                    'reward': torch.stack(batch_data['reward']).to(self.ppo_device),
                    'is_first': torch.stack(batch_data['is_first']).to(self.ppo_device),
                }
                if use_contact_head and len(batch_data.get('contact', [])) > 0:
                    batch_data_tensors['contact'] = torch.stack(batch_data['contact']).to(self.ppo_device)
                if use_object_state_head and len(batch_data.get('object_state', [])) > 0:
                    batch_data_tensors['object_state'] = torch.stack(batch_data['object_state']).to(self.ppo_device)
                batch_data = batch_data_tensors
                
                # Add sequence dimension if needed
                if batch_data['image'].dim() == 4:  # (B, H, W, C)
                    # Add time dimension: (B, T=1, H, W, C)
                    batch_data = {k: v.unsqueeze(1) if v.dim() == len(batch_data['image'].shape) - 1 
                                 else v.unsqueeze(1) for k, v in batch_data.items()}
            
            # Train world model
            step_metrics = self.world_model.train_step(batch_data)
            metrics.update(step_metrics)
            total_loss += step_metrics.get('kl', 0.0)
        
        metrics['avg_loss'] = total_loss / self.wm_train_steps
        return metrics
    
    def calc_gradients(self, input_dict):
        """Calculate gradients with world model support"""
        # Reset world model state in network if needed
        if self.use_world_model and hasattr(self.model, 'a2c_network'):
            if hasattr(self.model.a2c_network, 'reset_wm_state'):
                dones = input_dict.get('dones', None)
                self.model.a2c_network.reset_wm_state(dones)
        
        # Call parent gradient calculation
        return super().calc_gradients(input_dict)
    
    def _log_train_info(self, train_info, frame):
        """Log training info including world model metrics"""
        super()._log_train_info(train_info, frame)
        # World model metrics are logged in play_steps
        return
