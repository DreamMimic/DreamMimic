# Copyright (c) 2018-2022, NVIDIA Corporation
# Network builder with World Model integration for G1 robot
# G1-specific implementation to avoid affecting SMPLX training

from rl_games.algos_torch import network_builder
from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn
import numpy as np
from .world_model_g1 import WorldModelG1


class InterMimicBuilderWMG1(network_builder.A2CBuilder):
    """Network builder with World Model integration for G1"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world_model = None
        self.wm_config = None
        return

    class Network(network_builder.A2CBuilder.Network):
        """Network with World Model encoder integration for G1"""
        
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
            """Extract visual observations from full observation vector - G1 version"""
            batch_size = obs.shape[0]
            obs_dim = obs.shape[-1]
            
            # G1 student obs: 8135 (1983 proprioceptive + 6144 visual)
            # G1 teacher obs: 2153 (no visual)
            expected_visual_size = 32 * 32 * 2 * 3  # 6144
            single_frame_size = 32 * 32 * 2  # 2048
            
            # Check if this is student obs (with visual) or teacher obs (without visual)
            if obs_dim < expected_visual_size:
                # This is teacher obs (2153) - no visual information
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
                        print(f"[G1 WM] Warning: Unknown visual obs size {actual_visual_size} (obs_dim={obs_dim}), using zeros")
                        self._extract_visual_warning_shown = True
                    visual_obs = torch.zeros(batch_size, 2, 32, 32, device=obs.device)
            except RuntimeError as e:
                if not hasattr(self, '_extract_visual_error_shown'):
                    print(f"[G1 WM] Error reshaping visual obs: {e}")
                    print(f"  Visual obs shape: {visual_obs.shape}, size: {actual_visual_size}, obs_dim: {obs_dim}")
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
                    obs_dim = obs.shape[-1]
                    expected_visual_size = 32 * 32 * 2 * 3  # 6144
                    
                    # Skip world model if this is teacher obs (no visual information)
                    if obs_dim < expected_visual_size:
                        wm_features = None
                    else:
                        # Student obs - extract visual and proprioceptive observations
                        visual_obs, prop_obs = self.extract_visual_obs(obs)
                        
                        # Check if visual obs is all zeros
                        if visual_obs.abs().sum() < 1e-6:
                            wm_features = None
                        else:
                            # Prepare world model input
                            visual_obs_wm = visual_obs.permute(0, 2, 3, 1)  # (B, 32, 32, 2)
                            
                            # G1 prop_dim is 1983
                            prop_dim = self.wm_config.get('prop_dim', 1983)
                            if prop_obs.shape[-1] < prop_dim:
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
                            
                            # Get world model state
                            if not hasattr(self, '_wm_state') or self._wm_state is None:
                                self._wm_state = self.world_model.dynamics.initial(obs.shape[0])
                            
                            # Update world model state
                            action_placeholder = torch.zeros(obs.shape[0], self.wm_config.get('num_actions', 43), device=obs.device)
                            self._wm_state, _ = self.world_model.dynamics.obs_step(
                                self._wm_state, action_placeholder, embed, 
                                torch.zeros(obs.shape[0], device=obs.device), sample=False
                            )
                            
                            # Get deterministic features
                            wm_features = self.world_model.dynamics.get_deter_feat(self._wm_state)
                    
                except Exception as e:
                    if not hasattr(self, '_wm_forward_error_shown'):
                        print(f"[G1 WM] Warning: World Model forward failed: {e}")
                        import traceback
                        traceback.print_exc()
                        self._wm_forward_error_shown = True
                    wm_features = None
            
            # Reset world model state on episode boundaries
            if states is not None:
                dones = states.get('dones', None)
                if dones is not None and torch.any(dones):
                    if hasattr(self, '_wm_state') and self._wm_state is not None:
                        batch_size = obs.shape[0]
                        self._wm_state = self.world_model.dynamics.initial(batch_size)
            
            # Original forward pass
            cnn_out = self.encoder_cnn(obs)
            cnn_out = cnn_out.reshape([-1, self.cnn_out_size])
            
            # Concatenate world model features if available
            if wm_features is not None:
                wm_encoded = self.wm_feature_encoder(wm_features)
                cnn_out = torch.cat([cnn_out, wm_encoded], dim=-1)
            
            mlp_input = cnn_out
            if self.has_rnn:
                mlp_input, states = self._forward_rnn(mlp_input, states)
            
            mu = self.actor_mlp(mlp_input)
            if self.is_discrete:
                mu = mu.view(mu.shape[0], -1, self.space_config['actions_num'])
            
            if self.central_value:
                value = self.value_mlp(mlp_input)
            else:
                # Critic also uses world model features
                critic_cnn_out = self.critic_encoder_cnn(obs)
                critic_cnn_out = critic_cnn_out.reshape([-1, self.critic_cnn_out_size])
                
                if wm_features is not None:
                    critic_wm_encoded = self.critic_wm_feature_encoder(wm_features)
                    critic_cnn_out = torch.cat([critic_cnn_out, critic_wm_encoded], dim=-1)
                
                value = self.critic_mlp(critic_cnn_out)
            
            return self._build_value(value, mu, states)
        
        def reset_wm_state(self, dones=None):
            """Reset world model state"""
            if hasattr(self, '_wm_state') and self._wm_state is not None:
                if dones is not None:
                    batch_size = self._wm_state['stoch'].shape[0]
                else:
                    batch_size = 1
                self._wm_state = self.world_model.dynamics.initial(batch_size)
        
        def eval_critic(self, obs, wm_features=None):
            """Evaluate critic with optional world model features"""
            critic_cnn_out = self.critic_encoder_cnn(obs)
            critic_cnn_out = critic_cnn_out.reshape([-1, self.critic_cnn_out_size])
            
            if wm_features is not None and self.use_world_model:
                critic_wm_encoded = self.critic_wm_feature_encoder(wm_features)
                critic_cnn_out = torch.cat([critic_cnn_out, critic_wm_encoded], dim=-1)
            
            value = self.critic_mlp(critic_cnn_out)
            return value
    
    def build(self, name, **kwargs):
        """Build network with World Model"""
        print(f"[G1 WM Network Builder] Building network '{name}'...")
        net = super().build(name, **kwargs)
        print(f"[G1 WM Network Builder] Base network built successfully!")
        
        # Initialize world model if enabled
        if self.params.get('use_world_model', False):
            print("[G1 WM Network Builder] Initializing World Model...")
            wm_config = self.params.get('world_model_config', {})
            
            # Get encoder config
            encoder_config = wm_config.get('encoder', {})
            visual_backbone = encoder_config.get('visual_backbone', 'cnn')
            visual_input_type = encoder_config.get('visual_input_type', 'depth_seg')
            
            # Determine visual channels
            if visual_input_type == 'rgb':
                visual_channels = 3
            elif visual_input_type in ['depth', 'segmentation']:
                visual_channels = 1
            elif visual_input_type == 'depth_seg':
                visual_channels = 2
            else:
                visual_channels = 2
            
            # G1 obs_shape
            obs_shape = {
                'prop': (wm_config.get('prop_dim', 1983),),  # G1 prop_dim
                'image': (32, 32, visual_channels)
            }
            
            print("\n" + "=" * 80)
            print("G1 World Model Network Builder:")
            print(f"  Encoder Type: {visual_backbone.upper()}")
            print(f"  Visual Input Type: {visual_input_type.upper()}")
            print(f"  Visual Channels: {visual_channels}")
            print(f"  Visual Observation Shape: {obs_shape['image']}")
            print(f"  Proprioceptive Dim: {obs_shape['prop'][0]}")
            print("=" * 80 + "\n")
            
            print("[G1 WM Network Builder] Creating WorldModelG1 instance...")
            world_model = WorldModelG1(
                wm_config,
                obs_shape,
                use_camera=True,
                device=kwargs.get('device', 'cuda:0')
            )
            print("[G1 WM Network Builder] WorldModelG1 created successfully!")
            
            print("[G1 WM Network Builder] Setting World Model to network...")
            net.set_world_model(world_model)
            self.world_model = world_model
            print("[G1 WM Network Builder] World Model setup complete!")
        else:
            print("[G1 WM Network Builder] World Model disabled, skipping initialization.")
        
        return net
