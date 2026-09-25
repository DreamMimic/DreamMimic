import copy

from rl_games.algos_torch import network_builder
import torch
import torch.nn as nn


class InterMimicBuilderStudentWM(network_builder.A2CBuilder):
    """Policy builder for student proprio + world-model feature observations."""

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.get("actions_num")
            original_input_shape = kwargs.get("input_shape")
            obs_dim = int(original_input_shape[0] if isinstance(original_input_shape, (tuple, list)) else original_input_shape)

            wm_cfg = params.get("student_wm", {})
            self.prop_dim = int(wm_cfg.get("prop_dim", 1041))
            self.wm_feature_dim = int(wm_cfg.get("wm_feature_dim", obs_dim - self.prop_dim))
            self.prop_feature_dim = int(wm_cfg.get("prop_feature_dim", 256))
            self.wm_latent_dim = int(wm_cfg.get("wm_latent_dim", 64))
            self.prop_hidden = list(wm_cfg.get("prop_hidden", [512, 256]))
            self.wm_hidden = list(wm_cfg.get("wm_hidden", [128]))
            self.use_transformer = bool(wm_cfg.get("use_transformer", False))
            self.transformer_dim = int(wm_cfg.get("transformer_dim", 256))
            self.transformer_heads = int(wm_cfg.get("transformer_heads", 8))
            self.transformer_layers = int(wm_cfg.get("transformer_layers", 2))
            self.transformer_ff_dim = int(wm_cfg.get("transformer_ff_dim", 512))
            self.transformer_dropout = float(wm_cfg.get("transformer_dropout", 0.1))
            self.transformer_pool = str(wm_cfg.get("transformer_pool", "mean")).lower()
            self.wm_deter_dim = int(wm_cfg.get("wm_deter_dim", 0))
            self.wm_prediction_dim = int(wm_cfg.get("wm_prediction_dim", 0))
            self.split_wm_tokens = self.wm_deter_dim > 0 and self.wm_prediction_dim > 0
            expected = self.prop_dim + self.wm_feature_dim
            if obs_dim != expected:
                raise ValueError(
                    f"Student WM policy expected obs dim {expected} (= {self.prop_dim}+{self.wm_feature_dim}), got {obs_dim}"
                )

            if self.use_transformer:
                if self.transformer_dim <= 0:
                    raise ValueError("student_wm.transformer_dim must be > 0")
                if self.transformer_heads <= 0:
                    raise ValueError("student_wm.transformer_heads must be > 0")
                if self.transformer_layers <= 0:
                    raise ValueError("student_wm.transformer_layers must be > 0")
                if self.transformer_ff_dim <= 0:
                    raise ValueError("student_wm.transformer_ff_dim must be > 0")
                if self.transformer_pool not in ("mean", "first"):
                    raise ValueError("student_wm.transformer_pool must be one of: mean, first")
                if self.transformer_dim % self.transformer_heads != 0:
                    raise ValueError(
                        "student_wm.transformer_dim must be divisible by student_wm.transformer_heads"
                    )
                if self.split_wm_tokens and self.wm_deter_dim + self.wm_prediction_dim != self.wm_feature_dim:
                    raise ValueError(
                        "student_wm.wm_deter_dim + student_wm.wm_prediction_dim must equal student_wm.wm_feature_dim"
                    )
                feature_dim = self.transformer_dim
            else:
                feature_dim = self.prop_feature_dim + self.wm_latent_dim
            modified_params = copy.deepcopy(params)
            modified_params.pop("cnn", None)
            kwargs["input_shape"] = (feature_dim,)
            super().__init__(modified_params, **kwargs)
            kwargs["input_shape"] = original_input_shape

            if self.is_continuous and (not self.space_config["learn_sigma"]):
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                sigma_init(self.sigma)

            self.expected_obs_dim = expected
            self.prop_encoder = self._build_feature_mlp(self.prop_dim, self.prop_hidden, self.prop_feature_dim)
            if self.use_transformer:
                self.prop_token_proj = nn.Linear(self.prop_feature_dim, self.transformer_dim)
                if self.split_wm_tokens:
                    self.wm_deter_token_proj = nn.Linear(self.wm_deter_dim, self.transformer_dim)
                    self.wm_prediction_token_proj = nn.Linear(self.wm_prediction_dim, self.transformer_dim)
                else:
                    self.wm_token_proj = nn.Linear(self.wm_feature_dim, self.transformer_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.transformer_dim,
                    nhead=self.transformer_heads,
                    dim_feedforward=self.transformer_ff_dim,
                    dropout=self.transformer_dropout,
                    activation="gelu",
                    batch_first=True,
                )
                self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
            else:
                self.wm_encoder = self._build_feature_mlp(self.wm_feature_dim, self.wm_hidden, self.wm_latent_dim)
            self.actor_cnn = nn.Sequential()
            if self.separate:
                self.critic_cnn = nn.Sequential()

        @staticmethod
        def _build_feature_mlp(in_dim, hidden_dims, out_dim):
            layers = []
            last = in_dim
            for hidden in hidden_dims:
                layers.extend([nn.Linear(last, int(hidden)), nn.ELU()])
                last = int(hidden)
            layers.extend([nn.Linear(last, out_dim), nn.ELU()])
            return nn.Sequential(*layers)

        def _encode_obs(self, obs):
            if obs.shape[-1] != self.expected_obs_dim:
                raise RuntimeError(
                    f"Student WM policy received obs dim {obs.shape[-1]}, expected {self.expected_obs_dim}. "
                    "Teacher obs (3198) or raw visual student obs (7185) must not be fed to this policy."
                )
            prop = obs[:, : self.prop_dim]
            wm_feature = obs[:, self.prop_dim : self.prop_dim + self.wm_feature_dim]
            prop_feat = self.prop_encoder(prop)
            if not self.use_transformer:
                wm_feat = self.wm_encoder(wm_feature)
                return torch.cat([prop_feat, wm_feat], dim=-1)

            prop_token = self.prop_token_proj(prop_feat)
            if self.split_wm_tokens:
                wm_deter = wm_feature[:, : self.wm_deter_dim]
                wm_prediction = wm_feature[:, self.wm_deter_dim : self.wm_deter_dim + self.wm_prediction_dim]
                tokens = torch.stack(
                    [
                        prop_token,
                        self.wm_deter_token_proj(wm_deter),
                        self.wm_prediction_token_proj(wm_prediction),
                    ],
                    dim=1,
                )
            else:
                tokens = torch.stack([prop_token, self.wm_token_proj(wm_feature)], dim=1)
            mixed = self.token_mixer(tokens)
            if self.transformer_pool == "first":
                return mixed[:, 0, :]
            return mixed.mean(dim=1)

        def forward(self, obs_dict):
            states = obs_dict.get("rnn_states", None)
            features = self._encode_obs(obs_dict["obs"])
            actor_outputs = self.eval_actor_with_features(features)
            value = self.eval_critic_with_features(features)
            return actor_outputs + (value, states)

        def eval_actor(self, obs):
            return self.eval_actor_with_features(self._encode_obs(obs))

        def eval_critic(self, obs):
            return self.eval_critic_with_features(self._encode_obs(obs))

        def eval_actor_with_features(self, features):
            a_out = self.actor_mlp(features)
            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config["fixed_sigma"]:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma
            if self.is_discrete:
                return self.logits(a_out)
            if self.is_multi_discrete:
                return [logit(a_out) for logit in self.logits]
            return None

        def eval_critic_with_features(self, features):
            c_out = self.critic_mlp(features)
            return self.value_act(self.value(c_out))

    def build(self, name, **kwargs):
        return InterMimicBuilderStudentWM.Network(self.params, **kwargs)
