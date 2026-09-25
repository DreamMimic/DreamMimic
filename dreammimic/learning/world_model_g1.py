# Copyright (c) 2018-2022, NVIDIA Corporation
# World Model implementation for G1 robot - based on Dreamer architecture
# This is a G1-specific implementation to avoid affecting SMPLX training

import torch
import torch.nn as nn
from torch import distributions as torchd
import numpy as np

# Import from local dreamer module
import sys
import os
local_dreamer_path = os.path.join(os.path.dirname(__file__), '../..', 'dreamer')
local_dreamer_path = os.path.abspath(local_dreamer_path)

if os.path.exists(local_dreamer_path):
    dreamer_parent = os.path.dirname(local_dreamer_path)
    if dreamer_parent not in sys.path:
        sys.path.insert(0, dreamer_parent)
    from dreamer import networks, tools
else:
    try:
        from dreamer import networks, tools
    except ImportError:
        raise ImportError(f"Cannot find dreamer module. Expected at: {local_dreamer_path}")

to_np = lambda x: x.detach().cpu().numpy()


class WorldModelG1(nn.Module):
    """World Model for G1 robot - simplified implementation based on Dreamer"""
    
    def __init__(self, config, obs_shape, use_camera=True, device='cuda:0'):
        print(f"[WorldModelG1] Initializing World Model on device {device}...")
        super(WorldModelG1, self).__init__()
        
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
        
        # Store config
        self._config = config.copy() if isinstance(config, dict) else config
        self.device = device
        
        # Convert numeric config values
        num_actions = to_numeric(config.get('num_actions', 43), 43, is_int=True)
        dyn_stoch = to_numeric(config.get('dyn_stoch', 32), 32, is_int=True)
        dyn_deter = to_numeric(config.get('dyn_deter', 200), 200, is_int=True)
        dyn_hidden = to_numeric(config.get('dyn_hidden', 200), 200, is_int=True)
        dyn_rec_depth = to_numeric(config.get('dyn_rec_depth', 1), 1, is_int=True)
        dyn_discrete = to_numeric(config.get('dyn_discrete', 0), 0, is_int=True)
        dyn_min_std = to_numeric(config.get('dyn_min_std', 0.1), 0.1)
        unimix_ratio = to_numeric(config.get('unimix_ratio', 0.01), 0.01)
        
        self._num_actions = num_actions
        
        # Encoder configuration
        encoder_config = config.get('encoder', {})
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
        
        # Update obs_shape if needed
        image_shape = obs_shape.get('image', (32, 32, 2))
        if isinstance(image_shape, tuple) and len(image_shape) == 3:
            image_shape = (image_shape[0], image_shape[1], visual_channels)
            obs_shape = obs_shape.copy()
            obs_shape['image'] = image_shape
        
        # Build encoder
        print("[WorldModelG1] Building encoder...")
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
        print(f"[WorldModelG1] Encoder built, embed_size={self.embed_size}")
        
        # Store visual config
        self.visual_backbone = visual_backbone
        self.visual_input_type = visual_input_type
        self.visual_channels = visual_channels
        
        # Dynamics: RSSM
        print("[WorldModelG1] Building dynamics (RSSM)...")
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
        print("[WorldModelG1] Dynamics built successfully!")
        
        # Feature size
        if dyn_discrete:
            self.feat_size = dyn_stoch * dyn_discrete + dyn_deter
        else:
            self.feat_size = dyn_stoch + dyn_deter
        
        # Multi-step to privileged mode
        self.enable_multi_to_privi = config.get('enable_multi_to_privi', False)
        self.multi_history_len = config.get('multi_history_len', 5)
        
        # Multi-step encoder (if enabled)
        if self.enable_multi_to_privi:
            self.multi_encoder = networks.MultiEncoder(obs_shape, **encoder_kwargs)
            # Temporal aggregator
            self.temporal_aggregator = nn.Sequential(
                nn.Linear(self.embed_size * self.multi_history_len, self.embed_size * 2),
                nn.SiLU() if config.get('act', 'SiLU') == 'SiLU' else nn.ReLU(),
                nn.Linear(self.embed_size * 2, self.embed_size),
            )
        else:
            self.multi_encoder = None
            self.temporal_aggregator = None
        
        # Heads
        self.heads = nn.ModuleDict()
        
        # Decoder (always enabled for reconstruction)
        decoder_config = config.get('decoder', {})
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
        
        # Privileged information head (if enabled)
        if self.enable_multi_to_privi:
            privi_config = config.get('privileged_head', {})
            privi_dim = config.get('privi_dim', 170)  # G1 default: target_states(13) + ig(52*3=156) + target_contact(1)
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
            self._privi_dim = privi_dim
        
        # Reward head (if enabled)
        if config.get('use_reward_head', False):
            reward_config = config.get('reward_head', {})
            self.heads["reward"] = networks.MLP(
                self.feat_size,
                (255,) if reward_config.get('dist') == 'symlog_disc' else (),
                reward_config.get('layers', 2),
                config.get('units', 512),
                config.get('act', 'SiLU'),
                config.get('norm', True),
                dist=reward_config.get('dist', 'normal'),
                outscale=reward_config.get('outscale', 1.0),
                device=device,
                name="Reward",
            )
        
        # Verify grad_heads
        grad_heads = config.get('grad_heads', ['decoder', 'reward', 'privileged'])
        for name in grad_heads:
            if name not in self.heads:
                print(f"Warning: grad_head '{name}' not in heads. Available: {list(self.heads.keys())}")
        
        # Optimizer
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
        
        # Loss scales
        self._scales = dict(
            reward=config.get('reward_head', {}).get('loss_scale', 1.0),
            image=1.0,
            privileged=config.get('privileged_head', {}).get('loss_scale', 1.0),
        )
        
        # World model latent state (for inference)
        self._wm_latent_state = None
        
        # Move to device
        print(f"[WorldModelG1] Moving model to device {device}...")
        self.to(device)
        print("[WorldModelG1] Model moved to device successfully!")
        
        print("=" * 80)
        print("G1 World Model Configuration:")
        print(f"  Visual Backbone: {visual_backbone.upper()}")
        print(f"  Visual Input Type: {visual_input_type.upper()}")
        print(f"  Visual Channels: {visual_channels}")
        print(f"  Embedding Size: {self.embed_size}")
        print(f"  Feature Size: {self.feat_size}")
        print(f"  Enable Multi-to-Privileged: {self.enable_multi_to_privi}")
        if self.enable_multi_to_privi:
            print(f"  Multi History Length: {self.multi_history_len}")
        print(f"  Heads: {list(self.heads.keys())}")
        print(f"  Grad Heads: {grad_heads}")
        print("=" * 80)
        print("[WorldModelG1] Initialization complete!")
    
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
        """Encode multi-step history observations"""
        if not self.enable_multi_to_privi:
            raise ValueError("encode_multi_history called but enable_multi_to_privi is False")
        
        batch_size = image_history.shape[0]
        history_len = image_history.shape[1]
        
        # Encode each step independently
        embeddings = []
        for t in range(history_len):
            obs_t = {
                'image': image_history[:, t],
                'prop': prop_history[:, t],
            }
            embed_t = self.multi_encoder(obs_t)
            embeddings.append(embed_t)
        
        # Concatenate and aggregate
        embed_concat = torch.cat(embeddings, dim=-1)
        embed_agg = self.temporal_aggregator(embed_concat)
        
        return embed_agg
    
    def train_step(self, data):
        """Training step - simplified implementation following Dreamer"""
        data = self.preprocess(data)
        
        # Use RequiresGrad context manager to ensure gradients flow correctly
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Handle multi-step history mode
                if self.enable_multi_to_privi and 'image_history' in data and 'prop_history' in data:
                    embed = self.encode_multi_history(data['image_history'], data['prop_history'])
                    action = data.get('action', torch.zeros(embed.shape[0], self._num_actions).to(self.device))
                    is_first = data.get('is_first', torch.zeros(embed.shape[0], device=self.device))
                else:
                    embed = self.encoder(data)
                    action = data.get('action', torch.zeros(embed.shape[0], self._num_actions).to(self.device))
                    is_first = data.get('is_first', torch.zeros(embed.shape[0], device=self.device))
                
                # Add sequence dimension if needed
                if embed.dim() == 2:
                    embed = embed.unsqueeze(1)
                    action = action.unsqueeze(1)
                    is_first = is_first.unsqueeze(1)
                
                # Observe dynamics
                post, prior = self.dynamics.observe(embed, action, is_first)
                
                # Remove sequence dimension if added
                if post['deter'].dim() == 3 and post['deter'].shape[1] == 1:
                    post = {k: v.squeeze(1) for k, v in post.items()}
                    prior = {k: v.squeeze(1) for k, v in prior.items()}
                
                # KL loss
                kl_free = self._config.get('kl_free', 1.0)
                dyn_scale = self._config.get('dyn_scale', 0.5)
                rep_scale = self._config.get('rep_scale', 0.1)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                
                # Predictions
                preds = {}
                grad_heads = self._config.get('grad_heads', ['decoder', 'reward', 'privileged'])
                for name, head in self.heads.items():
                    grad_head = name in grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    
                    # Add sequence dimension for decoder if needed
                    if name == 'decoder' and feat.dim() == 2:
                        feat = feat.unsqueeze(1)
                    
                    pred = head(feat)
                    if isinstance(pred, dict):
                        preds[name] = pred
                    else:
                        preds[name] = pred
                
                # Losses
                losses = {}
                for name, pred in preds.items():
                    # Skip heads not in grad_heads
                    if name not in grad_heads:
                        continue
                    
                    # Skip if head doesn't exist in data (except decoder)
                    if name not in data and name != 'decoder':
                        continue
                    
                    if name == 'decoder' and isinstance(pred, dict):
                        # Decoder outputs dict with 'image' and 'prop'
                        decoder_loss = None
                        
                        if 'image' in pred and 'image' in data:
                            image_target = data['image']
                            if image_target.dim() > 4:
                                image_target = image_target.squeeze(1)
                            
                            pred_image = pred['image']
                            if hasattr(pred_image, 'log_prob'):
                                image_loss = -pred_image.log_prob(image_target).mean()
                            else:
                                image_loss = nn.functional.mse_loss(pred_image, image_target)
                            
                            decoder_loss = image_loss
                        
                        if 'prop' in pred and 'prop' in data:
                            prop_target = data['prop']
                            if prop_target.dim() > 2:
                                prop_target = prop_target.squeeze(1)
                            
                            pred_prop = pred['prop']
                            if hasattr(pred_prop, 'log_prob'):
                                prop_loss = -pred_prop.log_prob(prop_target).mean()
                            else:
                                prop_loss = nn.functional.mse_loss(pred_prop, prop_target)
                            
                            if decoder_loss is None:
                                decoder_loss = prop_loss
                            else:
                                decoder_loss = decoder_loss + prop_loss
                        
                        if decoder_loss is not None:
                            losses[name] = decoder_loss
                    
                    elif name == 'privileged' and 'privileged' in data:
                        privi_target = data['privileged']
                        if privi_target.dim() > 2:
                            privi_target = privi_target.squeeze(1)
                        
                        if hasattr(pred, 'log_prob'):
                            loss = -pred.log_prob(privi_target)
                        else:
                            loss = nn.functional.mse_loss(pred, privi_target)
                        losses[name] = loss.mean()
                    
                    elif name in data:
                        target = data[name]
                        if target.dim() > 2 and target.shape[1] == 1:
                            target = target.squeeze(1)
                        
                        if hasattr(pred, 'log_prob'):
                            loss = -pred.log_prob(target)
                        else:
                            loss = nn.functional.mse_loss(pred, target)
                        losses[name] = loss.mean()
                
                # Scale losses
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                
                # Total loss
                model_loss = sum(scaled.values()) + kl_loss.mean()
            
            # Optimize
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        
        # Update metrics
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
