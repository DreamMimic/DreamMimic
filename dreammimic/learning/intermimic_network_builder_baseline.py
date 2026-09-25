# Copyright (c) 2018-2022, NVIDIA Corporation
# Baseline network builder for visual distillation without world model
# Supports ResNet18, ResNet34, and simple CNN backbones

from rl_games.algos_torch import network_builder
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available, ResNet backbones will not work")

class InterMimicBuilderBaseline(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # Store actions_num before calling super
            actions_num = kwargs.get('actions_num')
            input_shape = kwargs.get('input_shape')
            
            # Visual encoder configuration
            visual_backbone = params.get('visual_backbone', 'resnet18')
            visual_feature_dim = params.get('visual_feature_dim', 512)
            proprioceptive_output_dim = 256  # Output dimension of proprioceptive encoder
            
            # Calculate combined feature dimension
            combined_feature_dim = visual_feature_dim + proprioceptive_output_dim
            
            # Temporarily modify params to disable CNN and set correct input shape
            original_input_shape = kwargs.get('input_shape')
            
            # Create a copy of params to avoid modifying the original
            import copy
            modified_params = copy.deepcopy(params)
            
            # Completely remove CNN from params so parent class doesn't build it
            # Parent class checks 'cnn' in params to set has_cnn, so we need to remove it entirely
            original_cnn = modified_params.pop('cnn', None)
            
            # Modify input_shape to be 1D with combined_feature_dim
            # Parent class expects input_shape as tuple/list for shape calculation
            # For 1D input, use (combined_feature_dim,) format
            kwargs['input_shape'] = (combined_feature_dim,)
            
            # Call parent __init__ with modified params and kwargs
            super().__init__(modified_params, **kwargs)
            
            # Restore original values (for reference, though not strictly necessary)
            kwargs['input_shape'] = original_input_shape
            if original_cnn is not None:
                modified_params['cnn'] = original_cnn
            
            # Store configuration
            self.visual_backbone = visual_backbone
            self.visual_feature_dim = visual_feature_dim
            
            # Extract visual and proprioceptive dimensions from original input
            if isinstance(original_input_shape, (list, tuple)):
                obs_dim = original_input_shape[0] if len(original_input_shape) > 0 else original_input_shape
            elif isinstance(original_input_shape, int):
                obs_dim = original_input_shape
            else:
                obs_dim = original_input_shape[-1] if hasattr(original_input_shape, '__len__') else original_input_shape
            
            # Expected visual obs size: 32*32*2*3 = 6144
            expected_visual_size = 32 * 32 * 2 * 3  # 6144
            self.visual_obs_size = expected_visual_size
            self.proprioceptive_obs_size = obs_dim - expected_visual_size if obs_dim >= expected_visual_size else obs_dim
            
            # Build visual encoder
            self.visual_encoder = self._build_visual_encoder()
            
            # Build proprioceptive encoder (simple MLP)
            self.proprioceptive_encoder = self._build_proprioceptive_encoder()
            
            # Override actor/critic CNNs to be empty (we'll use combined features directly)
            self.actor_cnn = nn.Sequential()
            if self.separate:
                self.critic_cnn = nn.Sequential()
            
            return
        
        def _build_visual_encoder(self):
            """Build visual encoder backbone"""
            if self.visual_backbone == 'resnet18':
                if not TORCHVISION_AVAILABLE:
                    raise ImportError("torchvision required for ResNet18")
                resnet = models.resnet18(pretrained=False)
                # Modify first layer for 2-channel input (depth + segmentation)
                resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # Remove final FC layer, use features from avgpool
                resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc layer
                # Add projection to desired feature dimension
                visual_encoder = nn.Sequential(
                    resnet,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, self.visual_feature_dim),
                    nn.ReLU()
                )
            elif self.visual_backbone == 'resnet34':
                if not TORCHVISION_AVAILABLE:
                    raise ImportError("torchvision required for ResNet34")
                resnet = models.resnet34(pretrained=False)
                resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
                resnet = nn.Sequential(*list(resnet.children())[:-1])
                visual_encoder = nn.Sequential(
                    resnet,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, self.visual_feature_dim),
                    nn.ReLU()
                )
            elif self.visual_backbone == 'simple_cnn':
                # Simple CNN baseline
                visual_encoder = nn.Sequential(
                    nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0),  # 32x32 -> 7x7
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # 7x7 -> 2x2
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 2x2 -> 1x1
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(64, self.visual_feature_dim),
                    nn.ReLU()
                )
            else:
                raise ValueError(f"Unknown visual_backbone: {self.visual_backbone}")
            
            return visual_encoder
        
        def _build_proprioceptive_encoder(self):
            """Build proprioceptive encoder (simple MLP)"""
            layers = []
            in_dim = self.proprioceptive_obs_size
            hidden_dims = [512, 256]  # Simple 2-layer MLP
            
            for out_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            
            return nn.Sequential(*layers)
        
        
        def extract_visual_obs(self, obs):
            """Extract visual observations from full observation vector"""
            batch_size = obs.shape[0]
            obs_dim = obs.shape[-1]
            
            expected_visual_size = self.visual_obs_size
            single_frame_size = 32 * 32 * 2  # 2048
            
            # Check if this is student obs (with visual) or teacher obs (without visual)
            if obs_dim < expected_visual_size:
                # Teacher obs - return zeros for visual
                visual_obs = torch.zeros(batch_size, 2, 32, 32, device=obs.device)
                proprioceptive_obs = obs
                return visual_obs, proprioceptive_obs
            
            # Student obs - extract visual from the end
            visual_obs_start_idx = obs_dim - expected_visual_size
            visual_obs_raw = obs[:, visual_obs_start_idx:]
            proprioceptive_obs = obs[:, :visual_obs_start_idx]
            
            # Reshape visual obs: (B, 6144) -> (B, 3, 32, 32, 2) -> (B, 2, 32, 32) (use most recent frame)
            try:
                visual_obs_raw = visual_obs_raw.view(batch_size, 3, single_frame_size)
                visual_obs_raw = visual_obs_raw.view(batch_size, 3, 32, 32, 2)
                visual_obs = visual_obs_raw[:, -1, :, :, :]  # (B, 32, 32, 2)
                visual_obs = visual_obs.permute(0, 3, 1, 2)  # (B, 2, 32, 32)
            except RuntimeError:
                # Fallback: try single frame
                try:
                    visual_obs_raw = visual_obs_raw.view(batch_size, single_frame_size)
                    visual_obs = visual_obs_raw.view(batch_size, 32, 32, 2)
                    visual_obs = visual_obs.permute(0, 3, 1, 2)  # (B, 2, 32, 32)
                except RuntimeError:
                    # Last resort: return zeros
                    visual_obs = torch.zeros(batch_size, 2, 32, 32, device=obs.device)
            
            return visual_obs, proprioceptive_obs
        
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            
            # Extract features ONCE to avoid duplicate ResNet18 forward passes
            # This significantly reduces memory usage (ResNet18 is large)
            visual_obs, proprioceptive_obs = self.extract_visual_obs(obs)
            visual_features = self.visual_encoder(visual_obs)
            proprioceptive_features = self.proprioceptive_encoder(proprioceptive_obs)
            features = torch.cat([visual_features, proprioceptive_features], dim=1)
            
            # Pass pre-computed features to actor and critic
            actor_outputs = self.eval_actor_with_features(features)
            value = self.eval_critic_with_features(features)
            
            output = actor_outputs + (value, states)
            return output
        
        def eval_actor(self, obs):
            """Evaluate actor - accepts raw observations and extracts features internally"""
            # Extract and encode features from raw observations
            # This handles both teacher obs (3198 dim) and student obs (7355 dim)
            visual_obs, proprioceptive_obs = self.extract_visual_obs(obs)
            visual_features = self.visual_encoder(visual_obs)
            proprioceptive_features = self.proprioceptive_encoder(proprioceptive_obs)
            features = torch.cat([visual_features, proprioceptive_features], dim=1)
            return self.eval_actor_with_features(features)
        
        def eval_actor_with_features(self, features):
            """Evaluate actor with pre-computed features (memory efficient)"""
            # Process through actor MLP (built by parent class)
            a_out = self.actor_mlp(features)
            
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
            
            return None
        
        def eval_critic(self, obs):
            """Evaluate critic - accepts raw observations and extracts features internally"""
            # Extract and encode features from raw observations
            # This handles both teacher obs (3198 dim) and student obs (7355 dim)
            visual_obs, proprioceptive_obs = self.extract_visual_obs(obs)
            visual_features = self.visual_encoder(visual_obs)
            proprioceptive_features = self.proprioceptive_encoder(proprioceptive_obs)
            features = torch.cat([visual_features, proprioceptive_features], dim=1)
            return self.eval_critic_with_features(features)
        
        def eval_critic_with_features(self, features):
            """Evaluate critic with pre-computed features (memory efficient)"""
            # Process through critic MLP (built by parent class)
            c_out = self.critic_mlp(features)
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = InterMimicBuilderBaseline.Network(self.params, **kwargs)
        return net
