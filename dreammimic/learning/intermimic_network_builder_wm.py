# Copyright (c) 2018-2022, NVIDIA Corporation
# Network builder with World Model integration for InterMimic distillation

from rl_games.algos_torch import network_builder
from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn
import numpy as np
from .world_model import WorldModel


class InterMimicBuilderWM(network_builder.A2CBuilder):
    """Network builder with World Model integration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world_model = None
        self.wm_config = None
        return

    class Network(network_builder.A2CBuilder.Network):
        """Network with World Model encoder integration"""
        
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            
            # World Model configuration
            self.use_world_model = params.get('use_world_model', False)
            self.wm_config = params.get('world_model_config', {})
            self.wm_feature_dim = params.get('wm_feature_dim', 200)  # dyn_deter from world model
            self.wm_latent_dim = params.get('wm_latent_dim', 32)  # compressed WM features
            
            # World Model instance (will be set externally)
            self.world_model = None
            
            # World Model feature encoder for actor
            if self.use_world_model:
                # Encoder to compress world model features
                wm_encoder_layers = []
                wm_encoder_layers.append(nn.Linear(self.wm_feature_dim, 128))
                wm_encoder_layers.append(nn.ELU())
                wm_encoder_layers.append(nn.Linear(128, self.wm_latent_dim))
                wm_encoder_layers.append(nn.ELU())
                self.wm_feature_encoder = nn.Sequential(*wm_encoder_layers)
                
                # World Model feature encoder for critic
                critic_wm_encoder_layers = []
                critic_wm_encoder_layers.append(nn.Linear(self.wm_feature_dim, 128))
                critic_wm_encoder_layers.append(nn.ELU())
                critic_wm_encoder_layers.append(nn.Linear(128, self.wm_latent_dim))
                critic_wm_encoder_layers.append(nn.ELU())
                self.critic_wm_feature_encoder = nn.Sequential(*critic_wm_encoder_layers)
            
            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            return
        
        def set_world_model(self, world_model):
            """Set world model instance"""
            self.world_model = world_model
        
        def extract_visual_obs(self, obs):
            """Extract visual observations from full observation vector"""
            # Assuming visual obs is at the end of obs vector
            # This should match the environment's observation structure
            # For InterMimic: visual obs is concatenated at the end
            batch_size = obs.shape[0]
            obs_dim = obs.shape[-1]
            
            # Expected visual obs size: 32*32*2*3 = 6144
            # Student obs should be 7355 (1211 proprioceptive + 6144 visual)
            # Teacher obs is 3198 (no visual)
            expected_visual_size = 32 * 32 * 2 * 3  # 6144
            expected_student_obs_size = 1211 + expected_visual_size  # 7355
            single_frame_size = 32 * 32 * 2  # 2048
            
            # Check if this is student obs (with visual) or teacher obs (without visual)
            if obs_dim < expected_visual_size:
                # This is teacher obs (3198) - no visual information
                # Return zeros for visual obs and use all obs as proprioceptive
                visual_obs = torch.zeros(batch_size, 2, 32, 32, device=obs.device)
                proprioceptive_obs = obs  # All obs is proprioceptive
                return visual_obs, proprioceptive_obs
            
            # This should be student obs - extract visual from the end
            visual_obs_start_idx = obs_dim - expected_visual_size
            visual_obs = obs[:, visual_obs_start_idx:]
            proprioceptive_obs = obs[:, :visual_obs_start_idx]
            
            actual_visual_size = visual_obs.shape[-1]
            
            # Handle different visual obs sizes
            try:
                if actual_visual_size == expected_visual_size:
                    # Standard case: 3 frames (6144)
                    visual_obs = visual_obs.view(batch_size, 3, 32 * 32 * 2)
                    visual_obs = visual_obs.view(batch_size, 3, 32, 32, 2)
                    visual_obs = visual_obs[:, -1, :, :, :]  # Use most recent frame
                    visual_obs = visual_obs.permute(0, 3, 1, 2)  # (B, 2, 32, 32)
                elif actual_visual_size == single_frame_size:
                    # Single frame case (2048)
                    visual_obs = visual_obs.view(batch_size, 32, 32, 2)
                    visual_obs = visual_obs.permute(0, 3, 1, 2)  # (B, 2, 32, 32)
                elif actual_visual_size == single_frame_size * 2:
                    # Two frames case (4096)
                    visual_obs = visual_obs.view(batch_size, 2, 32, 32, 2)
                    visual_obs = visual_obs[:, -1, :, :, :]  # Use most recent frame
                    visual_obs = visual_obs.permute(0, 3, 1, 2)  # (B, 2, 32, 32)
                else:
                    # Unknown size - return zeros to avoid crash
                    if not hasattr(self, '_extract_visual_warning_shown'):
                        print(f"Warning: Unknown visual obs size {actual_visual_size} (obs_dim={obs_dim}), using zeros")
                        self._extract_visual_warning_shown = True
                    visual_obs = torch.zeros(batch_size, 2, 32, 32, device=obs.device)
            except RuntimeError as e:
                if not hasattr(self, '_extract_visual_error_shown'):
                    print(f"Error reshaping visual obs in extract_visual_obs: {e}")
                    print(f"Visual obs shape: {visual_obs.shape}, size: {actual_visual_size}, obs_dim: {obs_dim}")
                    self._extract_visual_error_shown = True
                visual_obs = torch.zeros(batch_size, 2, 32, 32, device=obs.device)
            
            return visual_obs, proprioceptive_obs
        
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            
            # Extract world model features if enabled
            wm_features = None
            if self.use_world_model and self.world_model is not None:
                try:
                    # Check if this is student obs (with visual) or teacher obs (without visual)
                    obs_dim = obs.shape[-1]
                    expected_visual_size = 32 * 32 * 2 * 3  # 6144
                    
                    # Skip world model if this is teacher obs (no visual information)
                    if obs_dim < expected_visual_size:
                        # Teacher obs - skip world model, use None features
                        wm_features = None
                    else:
                        # Student obs - extract visual and proprioceptive observations
                        visual_obs, prop_obs = self.extract_visual_obs(obs)
                        
                        # Check if visual obs is all zeros (indicating teacher obs or error)
                        if visual_obs.abs().sum() < 1e-6:
                            # Visual obs is all zeros, skip world model
                            wm_features = None
                        else:
                            # Prepare world model input
                            # Reshape visual obs for world model: (B, 2, 32, 32) -> (B, 32, 32, 2)
                            visual_obs_wm = visual_obs.permute(0, 2, 3, 1)  # (B, 32, 32, 2)
                            
                            # Ensure prop_obs has enough dimensions
                            prop_dim = self.wm_config.get('prop_dim', 64)
                            if prop_obs.shape[-1] < prop_dim:
                                # Pad with zeros if needed
                                pad_size = prop_dim - prop_obs.shape[-1]
                                prop_obs_padded = torch.cat([
                                    prop_obs,
                                    torch.zeros(prop_obs.shape[0], pad_size, device=prop_obs.device)
                                ], dim=-1)
                            else:
                                prop_obs_padded = prop_obs[:, :prop_dim]
                            
                            # World model expects dict with 'image' and 'prop'
                            wm_obs = {
                                'image': visual_obs_wm,
                                'prop': prop_obs_padded,
                                'is_first': torch.zeros(obs.shape[0], device=obs.device)
                            }
                            
                            # Encode with world model
                            embed = self.world_model.encode(wm_obs)
                            
                            # Get world model state (use previous state or initialize)
                            if not hasattr(self, '_wm_state') or self._wm_state is None:
                                self._wm_state = self.world_model.dynamics.initial(obs.shape[0])
                            
                            # Update world model state
                            action_placeholder = torch.zeros(obs.shape[0], self.wm_config.get('num_actions', 153), device=obs.device)
                            self._wm_state, _ = self.world_model.dynamics.obs_step(
                                self._wm_state, action_placeholder, embed, 
                                torch.zeros(obs.shape[0], device=obs.device), sample=False
                            )
                            
                            # Get deterministic features
                            wm_features = self.world_model.dynamics.get_deter_feat(self._wm_state)
                    
                except Exception as e:
                    if not hasattr(self, '_wm_forward_error_shown'):
                        print(f"Warning: World Model forward failed: {e}")
                        import traceback
                        traceback.print_exc()
                        self._wm_forward_error_shown = True
                    wm_features = None
            
            actor_outputs = self.eval_actor(obs, wm_features)
            value = self.eval_critic(obs, wm_features)

            output = actor_outputs + (value, states)
            return output

        def eval_actor(self, obs, wm_features=None):
            """Evaluate actor with optional world model features"""
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            
            # Store original CNN output size
            cnn_output_size = a_out.shape[-1]
            
            # Concatenate world model features if available
            if self.use_world_model and wm_features is not None:
                wm_encoded = self.wm_feature_encoder(wm_features)
                a_out = torch.cat([a_out, wm_encoded], dim=-1)
            
            # Adjust MLP if needed (only first time)
            if self.use_world_model and wm_features is not None and not hasattr(self, '_mlp_adjusted'):
                # Check if MLP input size matches
                first_mlp_layer = None
                for module in self.actor_mlp:
                    if isinstance(module, nn.Linear):
                        first_mlp_layer = module
                        break
                
                if first_mlp_layer is not None:
                    expected_input_size = cnn_output_size + self.wm_latent_dim
                    if first_mlp_layer.in_features != expected_input_size:
                        # Need to create adapter layer or rebuild MLP
                        # For now, use a simple adapter
                        adapter = nn.Linear(cnn_output_size + self.wm_latent_dim, first_mlp_layer.in_features)
                        self.actor_mlp_adapter = adapter
                        self._mlp_adjusted = True
                    else:
                        self._mlp_adjusted = True
            
            # Apply adapter if exists
            if hasattr(self, 'actor_mlp_adapter'):
                a_out = self.actor_mlp_adapter(a_out)
            
            a_out = self.actor_mlp(a_out)
                     
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs, wm_features=None):
            """Evaluate critic with optional world model features"""
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            
            # Store original CNN output size
            cnn_output_size = c_out.shape[-1]
            
            # Concatenate world model features if available (Scheme 3)
            if self.use_world_model and wm_features is not None:
                wm_encoded = self.critic_wm_feature_encoder(wm_features)
                c_out = torch.cat([c_out, wm_encoded], dim=-1)
            
            # Adjust MLP if needed (only first time)
            if self.use_world_model and wm_features is not None and not hasattr(self, '_critic_mlp_adjusted'):
                # Check if MLP input size matches
                first_mlp_layer = None
                for module in self.critic_mlp:
                    if isinstance(module, nn.Linear):
                        first_mlp_layer = module
                        break
                
                if first_mlp_layer is not None:
                    expected_input_size = cnn_output_size + self.wm_latent_dim
                    if first_mlp_layer.in_features != expected_input_size:
                        # Need to create adapter layer
                        adapter = nn.Linear(cnn_output_size + self.wm_latent_dim, first_mlp_layer.in_features)
                        self.critic_mlp_adapter = adapter
                        self._critic_mlp_adjusted = True
                    else:
                        self._critic_mlp_adjusted = True
            
            # Apply adapter if exists
            if hasattr(self, 'critic_mlp_adapter'):
                c_out = self.critic_mlp_adapter(c_out)
            
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value
        
        def reset_wm_state(self, dones=None):
            """Reset world model state"""
            if hasattr(self, '_wm_state'):
                if dones is not None:
                    batch_size = self._wm_state['deter'].shape[0]
                    for key in self._wm_state:
                        self._wm_state[key][dones] = 0.0
                else:
                    self._wm_state = None

    def build(self, name, **kwargs):
        """Build network with world model support"""
        # Add world model config to params
        if 'world_model_config' not in self.params:
            self.params['world_model_config'] = {}
        if 'use_world_model' not in self.params:
            self.params['use_world_model'] = False
        
        net = InterMimicBuilderWM.Network(self.params, **kwargs)
        
        # Initialize world model if enabled
        if self.params.get('use_world_model', False):
            wm_config = self.params.get('world_model_config', {})
            encoder_config = wm_config.get('encoder', {})
            visual_backbone = encoder_config.get('visual_backbone', 'cnn')
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
                'prop': (wm_config.get('prop_dim', 64),),
                'image': (32, 32, visual_channels)
            }
            
            print("\n" + "=" * 80)
            print("World Model Network Builder (Training/Testing):")
            print(f"  Encoder Type: {visual_backbone.upper()}")
            print(f"  Visual Input Type: {visual_input_type.upper()}")
            print(f"  Visual Channels: {visual_channels}")
            print(f"  Visual Observation Shape: {obs_shape['image']}")
            print(f"  Proprioceptive Dim: {obs_shape['prop'][0]}")
            print("=" * 80 + "\n")
            
            world_model = WorldModel(
                wm_config,
                obs_shape,
                use_camera=True,
                device=kwargs.get('device', 'cuda:0')
            )
            net.set_world_model(world_model)
            self.world_model = world_model
        
        return net
