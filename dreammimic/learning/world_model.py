# Copyright (c) 2018-2022, NVIDIA Corporation
# World Model implementation for InterMimic distillation
# Based on Dreamer architecture from WMP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd
import numpy as np
import re

# Import from local dreamer module (in InterMimic/dreamer)
import sys
import os

# First try to import from local dreamer module in InterMimic project
local_dreamer_path = os.path.join(os.path.dirname(__file__), '../..', 'dreamer')
local_dreamer_path = os.path.abspath(local_dreamer_path)

if os.path.exists(local_dreamer_path):
    # Add parent directory to path so we can import dreamer
    dreamer_parent = os.path.dirname(local_dreamer_path)
    if dreamer_parent not in sys.path:
        sys.path.insert(0, dreamer_parent)
    from dreamer import networks, tools
else:
    # Fallback: try to import directly (if dreamer is installed as package)
    try:
        from dreamer import networks, tools
    except ImportError:
        raise ImportError(f"Cannot find dreamer module. Expected at: {local_dreamer_path}")

# Import torchvision for ResNet backbones
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available, ResNet backbones will not work")



class ResNetMultiEncoder(nn.Module):
    """Multi-encoder with ResNet visual backbone and MLP proprioceptive encoder"""
    
    def __init__(self, visual_encoder, prop_encoder, visual_feature_dim, prop_output_dim, visual_channels=2):
        super(ResNetMultiEncoder, self).__init__()
        self.visual_encoder = visual_encoder
        self.prop_encoder = prop_encoder
        self.visual_feature_dim = visual_feature_dim
        self.prop_output_dim = prop_output_dim
        self.visual_channels = visual_channels
        self.outdim = visual_feature_dim + prop_output_dim
    
    def forward(self, obs):
        """
        Forward pass
        
        Args:
            obs: dict with keys:
                - 'image': (B, H, W, C) or (B, C, H, W) tensor
                - 'prop': (B, D) tensor
        
        Returns:
            embed: (B, visual_feature_dim + prop_output_dim) concatenated features
        """
        # Extract visual and proprioceptive observations
        image = obs.get('image')
        prop = obs.get('prop')
        
        # Process visual observation
        if image is not None:
            # Handle different input formats
            if image.dim() == 4:
                # (B, H, W, C) -> (B, C, H, W)
                if image.shape[-1] == self.visual_channels or (image.shape[-1] in [1, 2, 3] and image.shape[1] != self.visual_channels):
                    image = image.permute(0, 3, 1, 2)
            # Now image should be (B, C, H, W)
            # Ensure channel dimension matches
            if image.shape[1] != self.visual_channels:
                # Handle channel mismatch - pad or slice as needed
                if image.shape[1] < self.visual_channels:
                    # Pad with zeros
                    pad_channels = self.visual_channels - image.shape[1]
                    padding = torch.zeros(image.shape[0], pad_channels, image.shape[2], image.shape[3], 
                                        device=image.device, dtype=image.dtype)
                    image = torch.cat([image, padding], dim=1)
                else:
                    # Take first N channels
                    image = image[:, :self.visual_channels, :, :]
            
            visual_features = self.visual_encoder(image)  # (B, visual_feature_dim)
        else:
            batch_size = prop.shape[0] if prop is not None else 1
            visual_features = torch.zeros(batch_size, self.visual_feature_dim, 
                                         device=prop.device if prop is not None else 'cuda:0')
        
        # Process proprioceptive observation
        if prop is not None:
            prop_features = self.prop_encoder(prop)  # (B, prop_output_dim)
        else:
            batch_size = visual_features.shape[0]
            prop_features = torch.zeros(batch_size, self.prop_output_dim,
                                       device=visual_features.device)
        
        # Concatenate features
        embed = torch.cat([visual_features, prop_features], dim=-1)  # (B, outdim)
        
        return embed


class WorldModel(nn.Module):
    """World Model for visual distillation - based on Dreamer architecture"""
    
    def __init__(self, config, obs_shape, use_camera=True, device='cuda:0'):
        super(WorldModel, self).__init__()
        
        # Helper function to convert string to numeric type
        def to_numeric(value, default, is_int=False):
            if isinstance(value, str):
                try:
                    if 'e' in value or '.' in value:
                        return float(value)
                    else:
                        return int(value) if is_int else float(value)
                except (ValueError, TypeError):
                    return default
            return value if value is not None else default
        
        # Convert precision
        precision = config.get('precision', 32)
        if isinstance(precision, str):
            precision = int(precision)
        self._use_amp = True if precision == 16 else False
        
        # Store converted config values for later use
        self._config = config.copy() if isinstance(config, dict) else config
        # Convert and store numeric config values
        for key in ['kl_free', 'dyn_scale', 'rep_scale']:
            if key in self._config:
                self._config[key] = to_numeric(self._config[key], 
                                               {'kl_free': 1.0, 'dyn_scale': 0.5, 'rep_scale': 0.1}[key])
        
        self.device = device
        
        # Ensure numeric types for dynamics config - define num_actions FIRST
        num_actions = to_numeric(config.get('num_actions', 153), 153, is_int=True)
        dyn_stoch = to_numeric(config.get('dyn_stoch', 32), 32, is_int=True)
        dyn_deter = to_numeric(config.get('dyn_deter', 200), 200, is_int=True)
        dyn_hidden = to_numeric(config.get('dyn_hidden', 200), 200, is_int=True)
        dyn_rec_depth = to_numeric(config.get('dyn_rec_depth', 1), 1, is_int=True)
        dyn_discrete = to_numeric(config.get('dyn_discrete', 0), 0, is_int=True)
        dyn_min_std = to_numeric(config.get('dyn_min_std', 0.1), 0.1)
        unimix_ratio = to_numeric(config.get('unimix_ratio', 0.01), 0.01)
        
        # Store num_actions for use in forward
        self._num_actions = num_actions
        
        # Encoder: visual + proprioceptive observations
        encoder_config = config.get('encoder', {})
        # Get visual backbone type (default: 'cnn' for compatibility)
        visual_backbone = encoder_config.get('visual_backbone', 'cnn')
        # Get visual input type (default: 'depth_seg' for compatibility)
        visual_input_type = encoder_config.get('visual_input_type', 'depth_seg')
        # Options: 'rgb', 'depth', 'segmentation', 'depth_seg'
        
        # Determine number of input channels based on visual_input_type
        if visual_input_type == 'rgb':
            visual_channels = 3
        elif visual_input_type == 'depth' or visual_input_type == 'segmentation':
            visual_channels = 1
        elif visual_input_type == 'depth_seg':
            visual_channels = 2
        else:
            raise ValueError(f"Unknown visual_input_type: {visual_input_type}. Options: 'rgb', 'depth', 'segmentation', 'depth_seg'")
        
        # Build encoder based on visual backbone type
        if visual_backbone == 'resnet18' or visual_backbone == 'resnet34':
            # Use ResNet-based encoder
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision required for ResNet backbones")
            
            # Build ResNet visual encoder
            if visual_backbone == 'resnet18':
                resnet = models.resnet18(pretrained=False)
            else:  # resnet34
                resnet = models.resnet34(pretrained=False)
            
            # Modify first layer for variable-channel input
            resnet.conv1 = nn.Conv2d(visual_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Remove final FC layer
            resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
            
            # ResNet feature dimension (512 for ResNet18/34)
            resnet_feature_dim = 512
            visual_feature_dim = encoder_config.get('visual_feature_dim', 512)
            
            # Build visual encoder with ResNet
            visual_encoder = nn.Sequential(
                resnet_backbone,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(resnet_feature_dim, visual_feature_dim),
                nn.ReLU()
            )
            
            # Build proprioceptive encoder (MLP)
            prop_dim = obs_shape.get('prop', (64,))[0]
            mlp_layers = encoder_config.get('mlp_layers', 5)
            mlp_units = encoder_config.get('mlp_units', 1024)
            act = encoder_config.get('act', 'SiLU')
            
            # Build MLP layers
            mlp_modules = []
            in_dim = prop_dim
            for i in range(mlp_layers):
                mlp_modules.append(nn.Linear(in_dim, mlp_units))
                if encoder_config.get('norm', True):
                    mlp_modules.append(nn.LayerNorm(mlp_units))
                if act == 'SiLU':
                    mlp_modules.append(nn.SiLU())
                else:
                    mlp_modules.append(nn.ReLU())
                in_dim = mlp_units
            
            prop_encoder = nn.Sequential(*mlp_modules)
            prop_output_dim = mlp_units
            
            # Create custom encoder that combines visual and proprioceptive
            self.encoder = ResNetMultiEncoder(
                visual_encoder=visual_encoder,
                prop_encoder=prop_encoder,
                visual_feature_dim=visual_feature_dim,
                prop_output_dim=prop_output_dim,
                visual_channels=visual_channels
            )
            self.embed_size = visual_feature_dim + prop_output_dim
        else:
            # Use original CNN-based MultiEncoder (default)
            # Note: MultiEncoder expects obs_shape['image'] to match visual_channels
            # We need to update obs_shape if visual_input_type changes
            image_shape = obs_shape.get('image', (32, 32, 2))
            if isinstance(image_shape, tuple) and len(image_shape) == 3:
                # Update channel dimension
                image_shape = (image_shape[0], image_shape[1], visual_channels)
                obs_shape = obs_shape.copy()
                obs_shape['image'] = image_shape
            
            encoder_kwargs = {
                'mlp_keys': encoder_config.get('mlp_keys', '.*'),
                'cnn_keys': encoder_config.get('cnn_keys', 'image'),
                'act': encoder_config.get('act', 'SiLU'),
                'norm': encoder_config.get('norm', True),
                'cnn_depth': encoder_config.get('cnn_depth', 32),
                'kernel_size': encoder_config.get('kernel_size', 4),
                'minres': encoder_config.get('minres', 4),
                'mlp_layers': encoder_config.get('mlp_layers', 5),
                'mlp_units': encoder_config.get('mlp_units', 1024),
                'symlog_inputs': encoder_config.get('symlog_inputs', True),
                'use_camera': use_camera
            }
            self.encoder = networks.MultiEncoder(obs_shape, **encoder_kwargs)
            self.embed_size = self.encoder.outdim
        
        # Store visual backbone type and input type for later use
        self.visual_backbone = visual_backbone
        self.visual_input_type = visual_input_type
        self.visual_channels = visual_channels
        
        # Print encoder information
        print("=" * 80)
        print("World Model Encoder Configuration:")
        print(f"  Visual Backbone: {visual_backbone.upper()}")
        print(f"  Visual Input Type: {visual_input_type.upper()}")
        print(f"  Visual Channels: {visual_channels}")
        visual_format_map = {
            'rgb': 'RGB (3 channels)',
            'depth': 'Depth only (1 channel)',
            'segmentation': 'Segmentation only (1 channel)',
            'depth_seg': 'Depth + Segmentation (2 channels)'
        }
        print(f"  Visual Observation Format: {visual_format_map.get(visual_input_type, visual_input_type)}")
        print(f"  Visual Input Shape: {obs_shape.get('image', 'N/A')}")
        print(f"  Embedding Size: {self.embed_size}")
        if visual_backbone == 'resnet18' or visual_backbone == 'resnet34':
            print(f"  Visual Feature Dim: {visual_feature_dim}")
            print(f"  Proprioceptive Feature Dim: {prop_output_dim}")
        print("=" * 80)
        
        # Dynamics: RSSM
        self.dynamics = networks.RSSM(
            dyn_stoch,
            dyn_deter,
            dyn_hidden,
            dyn_rec_depth,
            dyn_discrete,
            config.get('act', 'SiLU'),
            config.get('norm', True),
            config.get('dyn_mean_act', 'none'),
            config.get('dyn_std_act', 'softplus'),
            dyn_min_std,
            unimix_ratio,
            config.get('initial', 'learned'),
            num_actions,
            self.embed_size,
            device,
        )
        
        # Feature size for policy
        if dyn_discrete:
            self.feat_size = dyn_stoch * dyn_discrete + dyn_deter
        else:
            self.feat_size = dyn_stoch + dyn_deter
        
        # Enable multi-step to privileged mode
        self.enable_multi_to_privi = config.get('enable_multi_to_privi', False)
        self.multi_history_len = config.get('multi_history_len', 3)  # Number of history steps
        
        # Multi-step encoder for history (if enable_multi_to_privi)
        if self.enable_multi_to_privi:
            # Encoder for multi-step history: processes each step independently then aggregates
            # We'll use a temporal encoder that processes history
            multi_encoder_config = config.get('multi_encoder', {})
            # Use same visual backbone as main encoder
            if self.visual_backbone == 'resnet18' or self.visual_backbone == 'resnet34':
                # Reuse the same ResNet encoder structure
                if not TORCHVISION_AVAILABLE:
                    raise ImportError("torchvision required for ResNet backbones")
                
                if self.visual_backbone == 'resnet18':
                    resnet = models.resnet18(pretrained=False)
                else:
                    resnet = models.resnet34(pretrained=False)
                
                resnet.conv1 = nn.Conv2d(self.visual_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
                
                resnet_feature_dim = 512
                visual_feature_dim = encoder_config.get('visual_feature_dim', 512)
                
                visual_encoder_multi = nn.Sequential(
                    resnet_backbone,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(resnet_feature_dim, visual_feature_dim),
                    nn.ReLU()
                )
                
                prop_dim = obs_shape.get('prop', (64,))[0]
                mlp_layers = encoder_config.get('mlp_layers', 5)
                mlp_units = encoder_config.get('mlp_units', 1024)
                act = encoder_config.get('act', 'SiLU')
                
                mlp_modules = []
                in_dim = prop_dim
                for i in range(mlp_layers):
                    mlp_modules.append(nn.Linear(in_dim, mlp_units))
                    if encoder_config.get('norm', True):
                        mlp_modules.append(nn.LayerNorm(mlp_units))
                    if act == 'SiLU':
                        mlp_modules.append(nn.SiLU())
                    else:
                        mlp_modules.append(nn.ReLU())
                    in_dim = mlp_units
                
                prop_encoder_multi = nn.Sequential(*mlp_modules)
                prop_output_dim = mlp_units
                
                self.multi_encoder = ResNetMultiEncoder(
                    visual_encoder=visual_encoder_multi,
                    prop_encoder=prop_encoder_multi,
                    visual_feature_dim=visual_feature_dim,
                    prop_output_dim=prop_output_dim,
                    visual_channels=self.visual_channels
                )
            else:
                # Use original CNN-based MultiEncoder
                encoder_kwargs = {
                    'mlp_keys': encoder_config.get('mlp_keys', '.*'),
                    'cnn_keys': encoder_config.get('cnn_keys', 'image'),
                    'act': encoder_config.get('act', 'SiLU'),
                    'norm': encoder_config.get('norm', True),
                    'cnn_depth': encoder_config.get('cnn_depth', 32),
                    'kernel_size': encoder_config.get('kernel_size', 4),
                    'minres': encoder_config.get('minres', 4),
                    'mlp_layers': encoder_config.get('mlp_layers', 5),
                    'mlp_units': encoder_config.get('mlp_units', 1024),
                    'symlog_inputs': encoder_config.get('symlog_inputs', True),
                    'use_camera': use_camera
                }
                self.multi_encoder = networks.MultiEncoder(obs_shape, **encoder_kwargs)
            # Temporal aggregation: simple MLP to aggregate history embeddings
            self.temporal_aggregator = nn.Sequential(
                nn.Linear(self.embed_size * self.multi_history_len, self.embed_size * 2),
                nn.SiLU() if config.get('act', 'SiLU') == 'SiLU' else nn.ReLU(),
                nn.Linear(self.embed_size * 2, self.embed_size),
            )
        else:
            self.multi_encoder = None
            self.temporal_aggregator = None
        
        # Decoder (optional, for reconstruction)
        self.heads = nn.ModuleDict()
        if config.get('use_decoder', True) or self.enable_multi_to_privi:
            decoder_config = config.get('decoder', {})
            # Provide default values for decoder if not in config
            decoder_kwargs = {
                'mlp_keys': decoder_config.get('mlp_keys', '.*'),
                'cnn_keys': decoder_config.get('cnn_keys', 'image'),
                'act': decoder_config.get('act', 'SiLU'),
                'norm': decoder_config.get('norm', True),
                'cnn_depth': decoder_config.get('cnn_depth', 32),
                'kernel_size': decoder_config.get('kernel_size', 4),
                'minres': decoder_config.get('minres', 4),
                'mlp_layers': decoder_config.get('mlp_layers', 5),
                'mlp_units': decoder_config.get('mlp_units', 1024),
                'cnn_sigmoid': decoder_config.get('cnn_sigmoid', False),
                'image_dist': decoder_config.get('image_dist', 'mse'),
                'vector_dist': decoder_config.get('vector_dist', 'symlog_mse'),
                'outscale': decoder_config.get('outscale', 1.0),
                'use_camera': use_camera
            }
            self.heads["decoder"] = networks.MultiDecoder(self.feat_size, obs_shape, **decoder_kwargs)
        
        # Privileged information predictor (for enable_multi_to_privi mode)
        if self.enable_multi_to_privi:
            privi_config = config.get('privileged_head', {})
            # Privileged info: target_states (13) + ig (num_bodies * 3) + target_contact (1)
            # We'll get the actual size from data collection, but provide a default
            privi_dim = config.get('privi_dim', 13 + 50 * 3 + 1)  # Default estimate
            self.heads["privileged"] = networks.MLP(
                self.feat_size,
                (privi_dim,),
                privi_config.get('layers', 3),
                config.get('units', 512),
                config.get('act', 'SiLU'),
                config.get('norm', True),
                dist=privi_config.get('dist', 'normal'),
                outscale=privi_config.get('outscale', 1.0),
                device=device,
                name="Privileged",
            )
            # Store privi_dim for dynamic update
            self._privi_dim = privi_dim
        
        # Reward predictor (optional)
        if config.get('use_reward_head', False):
            self.heads["reward"] = networks.MLP(
                self.feat_size,
                (255,) if config.get('reward_head', {}).get('dist') == 'symlog_disc' else (),
                config.get('reward_head', {}).get('layers', 2),
                config.get('units', 512),
                config.get('act', 'SiLU'),
                config.get('norm', True),
                dist=config.get('reward_head', {}).get('dist', 'normal'),
                outscale=config.get('reward_head', {}).get('outscale', 1.0),
                device=device,
                name="Reward",
            )
        
        # Contact information head (optional, predicts target_contact)
        if config.get('use_contact_head', False):
            contact_config = config.get('contact_head', {})
            contact_dim = config.get('contact_dim', 1)  # Default: target_contact is 1-dim
            self.heads["contact"] = networks.MLP(
                self.feat_size,
                (contact_dim,),
                contact_config.get('layers', 2),
                config.get('units', 512),
                config.get('act', 'SiLU'),
                config.get('norm', True),
                dist=contact_config.get('dist', 'normal'),  # 'normal' for continuous, 'binary' for binary classification
                outscale=contact_config.get('outscale', 1.0),
                device=device,
                name="Contact",
            )
            self._contact_dim = contact_dim
        
        # Object state head (optional, predicts target_states)
        if config.get('use_object_state_head', False):
            object_state_config = config.get('object_state_head', {})
            object_state_dim = config.get('object_state_dim', 13)  # Default: target_states is 13-dim
            self.heads["object_state"] = networks.MLP(
                self.feat_size,
                (object_state_dim,),
                object_state_config.get('layers', 3),
                config.get('units', 512),
                config.get('act', 'SiLU'),
                config.get('norm', True),
                dist=object_state_config.get('dist', 'normal'),
                outscale=object_state_config.get('outscale', 1.0),
                device=device,
                name="ObjectState",
            )
            self._object_state_dim = object_state_dim
        
        # Optimizer - ensure numeric types
        model_lr = to_numeric(config.get('model_lr', 1e-4), 1e-4)
        opt_eps = to_numeric(config.get('opt_eps', 1e-8), 1e-8)
        grad_clip = to_numeric(config.get('grad_clip', 1000), 1000)
        weight_decay = to_numeric(config.get('weight_decay', 0.0), 0.0)
        
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            model_lr,
            opt_eps,
            grad_clip,
            weight_decay,
            opt=config.get('opt', 'adam'),
            use_amp=self._use_amp,
        )
        
        self._scales = dict(
            reward=config.get('reward_head', {}).get('loss_scale', 1.0),
            image=1.0,
            privileged=config.get('privileged_head', {}).get('loss_scale', 1.0),
            contact=config.get('contact_head', {}).get('loss_scale', 1.0),
            object_state=config.get('object_state_head', {}).get('loss_scale', 1.0),
        )
        
        # World model latent state (for inference)
        self._wm_latent_state = None
        
    def preprocess(self, obs):
        """Preprocess observations"""
        assert "is_first" in obs
        obs = {k: torch.Tensor(v).to(self.device) if not isinstance(v, torch.Tensor) else v.to(self.device) 
               for k, v in obs.items()}
        return obs
    
    def encode(self, obs):
        """Encode observations to embeddings"""
        obs = self.preprocess(obs)
        embed = self.encoder(obs)
        return embed
    
    def encode_multi_history(self, image_history, prop_history):
        """Encode multi-step history observations
        
        Args:
            image_history: Tensor (B, T, H, W, C) - batch of history sequences
            prop_history: Tensor (B, T, D) - batch of history sequences
        
        Returns:
            embed: (B, embed_size) aggregated embedding
        """
        if not self.enable_multi_to_privi:
            raise ValueError("encode_multi_history called but enable_multi_to_privi is False")
        
        batch_size = image_history.shape[0]
        history_len = image_history.shape[1]
        
        # Encode each step independently
        embeddings = []
        for t in range(history_len):
            obs_t = {
                'image': image_history[:, t],  # (B, H, W, C)
                'prop': prop_history[:, t],    # (B, D)
            }
            embed_t = self.multi_encoder(obs_t)  # (B, embed_size)
            embeddings.append(embed_t)
        
        # Concatenate all embeddings: (B, T * embed_size)
        embed_concat = torch.cat(embeddings, dim=-1)
        
        # Aggregate through temporal aggregator: (B, embed_size)
        embed_agg = self.temporal_aggregator(embed_concat)
        
        return embed_agg
    
    def get_feat(self, state):
        """Get feature representation from state"""
        return self.dynamics.get_feat(state)
    
    def get_deter_feat(self, state):
        """Get deterministic feature from state"""
        return self.dynamics.get_deter_feat(state)
    
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """Single observation step"""
        return self.dynamics.obs_step(prev_state, prev_action, embed, is_first, sample)
    
    def forward(self, obs_dict):
        """Forward pass for training"""
        obs = self.preprocess(obs_dict)
        embed = self.encoder(obs)
        
        # Get actions and is_first
        action = obs_dict.get('action', torch.zeros(obs['is_first'].shape[0], self._num_actions).to(self.device))
        is_first = obs['is_first']
        
        # Observe dynamics
        post, prior = self.dynamics.observe(embed, action, is_first)
        
        # Get features
        feat = self.dynamics.get_feat(post)
        
        # Predictions
        preds = {}
        for name, head in self.heads.items():
            pred = head(feat)
            if isinstance(pred, dict):
                preds.update(pred)
            else:
                preds[name] = pred
        
        return {
            'post': post,
            'prior': prior,
            'feat': feat,
            'embed': embed,
            'preds': preds
        }
    
    def train_step(self, data):
        """Training step"""
        data = self.preprocess(data)
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Handle multi-step history mode
                if self.enable_multi_to_privi and 'image_history' in data and 'prop_history' in data:
                    # Multi-step mode: encode history
                    embed = self.encode_multi_history(data['image_history'], data['prop_history'])
                    # For multi-step, we use single action (current step)
                    action = data.get('action', torch.zeros(embed.shape[0], self._num_actions).to(self.device))
                    # is_first should be True only for first step in sequence
                    is_first = data.get('is_first', torch.zeros(embed.shape[0], device=self.device))
                else:
                    # Single-step mode: standard encoding
                    embed = self.encoder(data)
                    action = data.get('action', torch.zeros(embed.shape[0], self._num_actions).to(self.device))
                    is_first = data.get('is_first', torch.zeros(embed.shape[0], device=self.device))
                
                # Observe dynamics (for multi-step, we only do one step)
                # Add sequence dimension if needed
                if embed.dim() == 2:
                    embed = embed.unsqueeze(1)  # (B, 1, embed_size)
                    action = action.unsqueeze(1)  # (B, 1, action_dim)
                    is_first = is_first.unsqueeze(1)  # (B, 1)
                
                post, prior = self.dynamics.observe(embed, action, is_first)
                
                # Remove sequence dimension if added
                if post['deter'].dim() == 3 and post['deter'].shape[1] == 1:
                    post = {k: v.squeeze(1) for k, v in post.items()}
                    prior = {k: v.squeeze(1) for k, v in prior.items()}
                
                # KL loss - values already converted in __init__
                kl_free = self._config.get('kl_free', 1.0)
                dyn_scale = self._config.get('dyn_scale', 0.5)
                rep_scale = self._config.get('rep_scale', 0.1)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                
                # Predictions
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.get('grad_heads', ['decoder', 'reward', 'privileged'])
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if isinstance(pred, dict):
                        preds.update(pred)
                    else:
                        preds[name] = pred
                
                # Losses
                losses = {}
                for name, pred in preds.items():
                    if name == 'privileged' and 'privileged' in data:
                        # Privileged information loss
                        privi_target = data['privileged']
                        if privi_target.dim() > 2:
                            privi_target = privi_target.squeeze(1)  # Remove sequence dim if present
                        if hasattr(pred, 'log_prob'):
                            loss = -pred.log_prob(privi_target)
                        else:
                            loss = F.mse_loss(pred, privi_target)
                        losses[name] = loss
                    elif name == 'contact' and 'contact' in data:
                        # Contact information loss (target_contact)
                        contact_target = data['contact']
                        if contact_target.dim() > 2:
                            contact_target = contact_target.squeeze(1)  # Remove sequence dim if present
                        if hasattr(pred, 'log_prob'):
                            loss = -pred.log_prob(contact_target)
                        else:
                            # For binary classification, use BCE loss if dist is 'binary'
                            contact_config = self._config.get('contact_head', {})
                            if contact_config.get('dist') == 'binary':
                                loss = F.binary_cross_entropy_with_logits(
                                    pred if pred.dim() > 1 else pred.unsqueeze(-1),
                                    contact_target if contact_target.dim() > 1 else contact_target.unsqueeze(-1)
                                )
                            else:
                                loss = F.mse_loss(pred, contact_target)
                        losses[name] = loss
                    elif name == 'object_state' and 'object_state' in data:
                        # Object state loss (target_states)
                        object_state_target = data['object_state']
                        if object_state_target.dim() > 2:
                            object_state_target = object_state_target.squeeze(1)  # Remove sequence dim if present
                        if hasattr(pred, 'log_prob'):
                            loss = -pred.log_prob(object_state_target)
                        else:
                            loss = F.mse_loss(pred, object_state_target)
                        losses[name] = loss
                    elif name == 'decoder' and isinstance(pred, dict):
                        # Decoder outputs dict with 'image' and 'prop'
                        decoder_loss = 0.0
                        if 'image' in pred and 'image' in data:
                            image_target = data['image']
                            if image_target.dim() > 4:
                                image_target = image_target.squeeze(1)  # Remove sequence dim
                            if hasattr(pred['image'], 'log_prob'):
                                decoder_loss += -pred['image'].log_prob(image_target).mean()
                            else:
                                decoder_loss += F.mse_loss(pred['image'], image_target)
                        if 'prop' in pred and 'prop' in data:
                            prop_target = data['prop']
                            if prop_target.dim() > 2:
                                prop_target = prop_target.squeeze(1)  # Remove sequence dim
                            if hasattr(pred['prop'], 'log_prob'):
                                decoder_loss += -pred['prop'].log_prob(prop_target).mean()
                            else:
                                decoder_loss += F.mse_loss(pred['prop'], prop_target)
                        losses[name] = decoder_loss
                    elif name in data:
                        target = data[name]
                        if target.dim() > 2 and target.shape[1] == 1:
                            target = target.squeeze(1)  # Remove sequence dim
                        if hasattr(pred, 'log_prob'):
                            loss = -pred.log_prob(target)
                        else:
                            # MSE loss for reconstruction
                            loss = F.mse_loss(pred, target)
                        losses[name] = loss
                
                # Scale losses
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss.mean()
            
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        
        metrics.update({f"{name}_loss": loss.detach().cpu().item() for name, loss in losses.items()})
        metrics["kl"] = kl_value.mean().detach().cpu().item()
        metrics["dyn_loss"] = dyn_loss.mean().detach().cpu().item()
        metrics["rep_loss"] = rep_loss.mean().detach().cpu().item()
        
        return metrics
    
    def reset(self, batch_size):
        """Reset world model state"""
        self._wm_latent_state = self.dynamics.initial(batch_size)
        return self._wm_latent_state
    
    def update_state(self, obs, action, is_first):
        """Update world model state during rollout"""
        embed = self.encode(obs)
        if self._wm_latent_state is None or torch.any(is_first):
            self._wm_latent_state = self.dynamics.initial(obs['is_first'].shape[0])
        
        self._wm_latent_state, _ = self.dynamics.obs_step(
            self._wm_latent_state, action, embed, is_first, sample=False
        )
        return self._wm_latent_state
