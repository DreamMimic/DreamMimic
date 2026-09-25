import copy

from rl_games.algos_torch import network_builder
import torch
import torch.nn as nn


class InterMimicBuilderStudentVis(network_builder.A2CBuilder):
    """Student policy with an explicit image encoder.

    Observation layout is a flat tensor:
      [student_proprio, image_history]
    where image_history is stored as T frames of H*W*C values. The network
    encodes image_history into a compact feature before concatenating it with
    proprioceptive features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.get("actions_num")
            original_input_shape = kwargs.get("input_shape")
            obs_dim = self._shape_to_dim(original_input_shape)

            self.image_height = int(params.get("image_height", 32))
            self.image_width = int(params.get("image_width", 32))
            self.image_channels = int(params.get("image_channels", 2))
            self.image_history_len = int(params.get("image_history_len", 3))
            self.history_as_channels = bool(params.get("history_as_channels", True))
            self.visual_feature_dim = int(params.get("visual_feature_dim", 64))
            self.proprioceptive_feature_dim = int(params.get("proprioceptive_feature_dim", 256))
            self.proprioceptive_hidden = list(params.get("proprioceptive_hidden", [512, 256]))

            self.visual_obs_size = (
                self.image_height
                * self.image_width
                * self.image_channels
                * self.image_history_len
            )
            self.proprioceptive_obs_size = int(params.get("proprioceptive_dim", obs_dim - self.visual_obs_size))
            self.expected_obs_dim = self.proprioceptive_obs_size + self.visual_obs_size

            if obs_dim != self.expected_obs_dim:
                raise ValueError(
                    "InterMimicBuilderStudentVis expected obs dim "
                    f"{self.expected_obs_dim} (= proprio {self.proprioceptive_obs_size} "
                    f"+ visual {self.visual_obs_size}), got {obs_dim}. "
                    "This usually means the env wrapper is still returning teacher obs."
                )

            combined_feature_dim = self.visual_feature_dim + self.proprioceptive_feature_dim
            modified_params = copy.deepcopy(params)
            modified_params.pop("cnn", None)
            kwargs["input_shape"] = (combined_feature_dim,)
            super().__init__(modified_params, **kwargs)
            kwargs["input_shape"] = original_input_shape

            if self.is_continuous and (not self.space_config["learn_sigma"]):
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                    requires_grad=False,
                )
                sigma_init(self.sigma)

            self.visual_encoder = self._build_visual_encoder()
            self.proprioceptive_encoder = self._build_proprioceptive_encoder()
            self.actor_cnn = nn.Sequential()
            if self.separate:
                self.critic_cnn = nn.Sequential()

        @staticmethod
        def _shape_to_dim(shape):
            if isinstance(shape, int):
                return shape
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                return int(shape[0])
            if hasattr(shape, "__len__"):
                return int(shape[-1])
            return int(shape)

        def _build_visual_encoder(self):
            in_channels = self.image_channels * self.image_history_len if self.history_as_channels else self.image_channels
            cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, self.image_height, self.image_width)
                flat_dim = cnn(dummy).flatten(1).shape[1]
            return nn.Sequential(cnn, nn.Flatten(), nn.Linear(flat_dim, self.visual_feature_dim), nn.ELU())

        def _build_proprioceptive_encoder(self):
            layers = []
            in_dim = self.proprioceptive_obs_size
            for hidden_dim in self.proprioceptive_hidden:
                layers.append(nn.Linear(in_dim, int(hidden_dim)))
                layers.append(nn.ELU())
                in_dim = int(hidden_dim)
            layers.append(nn.Linear(in_dim, self.proprioceptive_feature_dim))
            layers.append(nn.ELU())
            return nn.Sequential(*layers)

        def _split_obs(self, obs):
            obs_dim = obs.shape[-1]
            if obs_dim != self.expected_obs_dim:
                raise RuntimeError(
                    "Student visual policy received obs dim "
                    f"{obs_dim}, expected {self.expected_obs_dim}. "
                    "Check that VecTaskDAggerStudentObsWrapper is used."
                )
            proprio = obs[:, : self.proprioceptive_obs_size]
            visual_flat = obs[:, self.proprioceptive_obs_size :]
            visual = visual_flat.reshape(
                obs.shape[0],
                self.image_history_len,
                self.image_height,
                self.image_width,
                self.image_channels,
            )
            if self.history_as_channels:
                visual = visual.permute(0, 1, 4, 2, 3).reshape(
                    obs.shape[0],
                    self.image_history_len * self.image_channels,
                    self.image_height,
                    self.image_width,
                )
            else:
                visual = visual[:, -1].permute(0, 3, 1, 2)
            return proprio, visual

        def _encode_obs(self, obs):
            proprio, visual = self._split_obs(obs)
            visual_features = self.visual_encoder(visual)
            proprio_features = self.proprioceptive_encoder(proprio)
            return torch.cat([proprio_features, visual_features], dim=-1)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            features = self._encode_obs(obs)
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
        return InterMimicBuilderStudentVis.Network(self.params, **kwargs)
