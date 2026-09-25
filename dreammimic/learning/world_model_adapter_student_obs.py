from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
import os
import sys

import torch
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dreamer.models import WorldModel


def _load_dreamer_defaults() -> Dict[str, Any]:
    with open(os.path.join(_REPO_ROOT, "dreamer", "configs.yaml"), "r") as f:
        raw = yaml.safe_load(f)
    return dict(raw["defaults"])


def build_wm_config(device: str, num_actions: int, overrides: Dict[str, Any]) -> SimpleNamespace:
    cfg = _load_dreamer_defaults()
    cfg["device"] = device
    cfg["num_actions"] = int(num_actions)
    for key, value in (overrides or {}).items():
        cfg[key] = value

    float_keys = ("model_lr", "opt_eps", "grad_clip", "kl_free", "dyn_scale", "rep_scale", "weight_decay", "dyn_min_std")
    int_keys = ("dyn_deter", "dyn_stoch", "dyn_discrete", "dyn_hidden", "dyn_rec_depth", "batch_size", "batch_length", "train_steps_per_iter", "train_start_steps", "units", "precision", "privi_dim", "contact_dim", "object_state_dim")
    for key in float_keys:
        if key in cfg:
            cfg[key] = float(cfg[key])
    for key in int_keys:
        if key in cfg and not isinstance(cfg[key], bool):
            cfg[key] = int(cfg[key])
    return SimpleNamespace(**cfg)


class StudentObsWorldModelAdapter:
    """Dreamer/RSSM adapter for InterMimic student visual observations."""

    def __init__(
        self,
        num_envs: int,
        num_actions: int,
        prop_dim: int,
        image_shape: Tuple[int, int, int],
        wm_cfg: Dict[str, Any],
        device: torch.device,
    ):
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.prop_dim = int(prop_dim)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.device = torch.device(device)
        self.image_on_cpu = bool(wm_cfg.get("image_on_cpu", True))
        self.replay_on_cpu = bool(wm_cfg.get("replay_on_cpu", True))
        self.use_fullheads = bool(wm_cfg.get("use_fullheads", False))
        self.use_privileged_head = self.use_fullheads or bool(wm_cfg.get("use_privileged_head", False))
        self.use_contact_head = self.use_fullheads or bool(wm_cfg.get("use_contact_head", False))
        self.use_object_state_head = self.use_fullheads or bool(wm_cfg.get("use_object_state_head", False))
        self.use_head_outputs_in_obs = bool(wm_cfg.get("use_head_outputs_in_obs", False))
        self.policy_head_outputs = list(
            wm_cfg.get("policy_head_outputs", ["reward", "privileged", "contact", "object_state"])
        )
        self.policy_feature_source = str(wm_cfg.get("policy_feature_source", "deter")).lower()
        if self.policy_feature_source not in ("deter", "stoch"):
            raise ValueError("worldModel.policy_feature_source must be one of: deter, stoch")
        self.reset_policy_state_each_step = bool(wm_cfg.get("reset_policy_state_each_step", False))
        self.zero_action_conditioning = bool(wm_cfg.get("zero_action_conditioning", False))

        self.wm_config = build_wm_config(str(self.device), self.num_actions, wm_cfg)
        self.world_model = WorldModel(
            self.wm_config,
            {"prop": (self.prop_dim,), "image": self.image_shape},
            use_camera=True,
        ).to(self.device)
        self.privi_dim = int(wm_cfg.get("privi_dim", getattr(self.wm_config, "privi_dim", 164)))
        self.contact_dim = int(wm_cfg.get("contact_dim", getattr(self.wm_config, "contact_dim", 1)))
        self.object_state_dim = int(wm_cfg.get("object_state_dim", getattr(self.wm_config, "object_state_dim", 13)))
        if self.policy_feature_source == "stoch":
            self.wm_deter_dim = int(self.wm_config.dyn_stoch) * int(self.wm_config.dyn_discrete or 1)
        else:
            self.wm_deter_dim = int(self.wm_config.dyn_deter)
        self.policy_head_output_dim = self._compute_policy_head_output_dim()
        self.wm_feature_dim = self.wm_deter_dim + self.policy_head_output_dim

        self._wm_latent = None
        self._wm_prev_action = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._wm_is_first = torch.ones((self.num_envs,), device=self.device)
        self._wm_feature = torch.zeros((self.num_envs, self.wm_feature_dim), device=self.device)

        self.replay_capacity = int(wm_cfg.get("replay_capacity", 4096))
        self.train_after_warmup_steps = int(wm_cfg.get("train_after_warmup_steps", 10000))
        self.train_steps_per_epoch = int(wm_cfg.get("train_steps_per_epoch", 10))
        self.batch_size = int(wm_cfg.get("batch_size", self.wm_config.batch_size))
        self.batch_length = int(wm_cfg.get("batch_length", self.wm_config.batch_length))
        reward_head = getattr(self.wm_config, "reward_head", {}) or {}
        self._reward_needs_event_dim = self.use_fullheads and reward_head.get("dist", "symlog_disc") != "symlog_disc"

        replay_device = torch.device("cpu") if self.replay_on_cpu else self.device
        img_device = torch.device("cpu") if self.image_on_cpu else replay_device
        h, w, c = self.image_shape
        self.ring = {
            "prop": torch.zeros((self.replay_capacity, self.num_envs, self.prop_dim), device=replay_device),
            "image": torch.zeros((self.replay_capacity, self.num_envs, h, w, c), device=img_device),
            "action": torch.zeros((self.replay_capacity, self.num_envs, self.num_actions), device=replay_device),
            "reward": torch.zeros((self.replay_capacity, self.num_envs), device=replay_device),
            "is_first": torch.zeros((self.replay_capacity, self.num_envs), device=replay_device),
        }
        if self.use_privileged_head:
            self.ring["privileged"] = torch.zeros((self.replay_capacity, self.num_envs, self.privi_dim), device=replay_device)
        if self.use_contact_head:
            self.ring["contact"] = torch.zeros((self.replay_capacity, self.num_envs, self.contact_dim), device=replay_device)
        if self.use_object_state_head:
            self.ring["object_state"] = torch.zeros((self.replay_capacity, self.num_envs, self.object_state_dim), device=replay_device)
        self._write_idx = 0
        self._filled = 0
        self._total_appended = 0
        self._dropped_bad_batches = 0
        self._sanitized_replay_rows = 0

    @torch.no_grad()
    def cache_prev_action(self, actions: torch.Tensor):
        if self.zero_action_conditioning:
            self._wm_prev_action.zero_()
            return
        self._wm_prev_action = self._sanitize_tensor(
            actions.detach().to(self.device, non_blocking=True), clamp=(-1.0, 1.0)
        )

    @torch.no_grad()
    def step(self, prop: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        prop = self._sanitize_tensor(prop.to(self.device, non_blocking=True))
        image = self._sanitize_tensor(image.to(self.device, non_blocking=True), clamp=(-10.0, 10.0))
        embed = self.world_model.encoder({"prop": prop, "image": image, "is_first": self._wm_is_first})
        prev_latent = None if self.reset_policy_state_each_step else self._wm_latent
        prev_action = torch.zeros_like(self._wm_prev_action) if self.zero_action_conditioning else self._wm_prev_action
        is_first = torch.ones_like(self._wm_is_first) if self.reset_policy_state_each_step else self._wm_is_first
        self._wm_latent, _ = self.world_model.dynamics.obs_step(
            prev_latent,
            prev_action,
            embed,
            is_first,
        )
        if self._latent_has_nonfinite(self._wm_latent):
            self._wm_latent = None
            self._wm_is_first[:] = 1.0
            self._wm_feature.zero_()
            return self._wm_feature
        deter_feature = self._policy_core_feature(self._wm_latent).detach()
        if self.policy_head_output_dim > 0:
            head_feature = self._predict_policy_head_outputs(self._wm_latent)
            self._wm_feature = torch.cat([deter_feature, head_feature], dim=-1).detach()
        else:
            self._wm_feature = deter_feature
        self._wm_feature = self._sanitize_tensor(self._wm_feature)
        self._wm_is_first.zero_()
        return self._wm_feature

    def _compute_policy_head_output_dim(self) -> int:
        if not self.use_head_outputs_in_obs:
            return 0
        dims = {
            "reward": 1,
            "privileged": self.privi_dim,
            "contact": self.contact_dim,
            "object_state": self.object_state_dim,
        }
        return sum(int(dims.get(name, 0)) for name in self.policy_head_outputs)

    def _policy_core_feature(self, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.policy_feature_source == "stoch":
            stoch = latent["stoch"]
            return stoch.reshape(stoch.shape[0], -1)
        return self.world_model.dynamics.get_deter_feat(latent)

    @torch.no_grad()
    def _predict_policy_head_outputs(self, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.world_model.dynamics.get_feat(latent)
        heads = getattr(self.world_model, "heads", {})
        outputs = []
        for name in self.policy_head_outputs:
            head = heads[name] if name in heads else None
            if head is None:
                continue
            pred = head(feat)
            value = pred.mode() if hasattr(pred, "mode") else getattr(pred, "mean", None)
            if value is None:
                continue
            if value.dim() > 2:
                value = value.reshape(value.shape[0], -1)
            elif value.dim() == 1:
                value = value.unsqueeze(-1)
            outputs.append(value.float())
        if outputs:
            out = torch.cat(outputs, dim=-1)
            if out.shape[-1] < self.policy_head_output_dim:
                pad = self.policy_head_output_dim - out.shape[-1]
                out = torch.cat([out, torch.zeros((out.shape[0], pad), device=out.device, dtype=out.dtype)], dim=-1)
            return out[:, : self.policy_head_output_dim]
        return torch.zeros((self.num_envs, self.policy_head_output_dim), device=self.device)

    def reset_envs(self, env_ids):
        if env_ids is None:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        if env_ids.numel() == 0:
            return
        self._wm_is_first[env_ids] = 1.0
        self._wm_prev_action[env_ids] = 0.0
        self._wm_feature[env_ids] = 0.0
        if self._wm_latent is not None:
            for value in self._wm_latent.values():
                value[env_ids] = 0.0

    @property
    def feature(self):
        return self._wm_feature

    @torch.no_grad()
    def reconstruct_current_image(self) -> Optional[torch.Tensor]:
        if self._wm_latent is None:
            return None
        heads = getattr(self.world_model, "heads", None)
        decoder = heads["decoder"] if heads is not None and "decoder" in heads else None
        if decoder is None:
            return None
        self.world_model.eval()
        feat = self.world_model.dynamics.get_feat(self._wm_latent)
        image_dist = decoder(feat).get("image", None)
        if image_dist is None:
            return None
        image = image_dist.mode() if hasattr(image_dist, "mode") else getattr(image_dist, "mean", None)
        if image is None:
            return None
        return self._sanitize_tensor(image.detach().float())

    def append(self, prop, image, action, reward, is_first, targets: Optional[Dict[str, torch.Tensor]] = None):
        idx = self._write_idx
        replay_device = torch.device("cpu") if self.replay_on_cpu else self.device
        prop = self._sanitize_tensor(prop.detach().to(replay_device, non_blocking=True))
        img_device = torch.device("cpu") if self.image_on_cpu else replay_device
        image = self._sanitize_tensor(image.detach().to(img_device, non_blocking=True), clamp=(-10.0, 10.0))
        action = self._sanitize_tensor(action.detach().to(replay_device, non_blocking=True), clamp=(-1.0, 1.0))
        if self.zero_action_conditioning:
            action.zero_()
        reward = self._sanitize_tensor(reward.detach().to(replay_device, non_blocking=True).view(-1), clamp=(-1e6, 1e6))
        is_first = is_first.detach().to(replay_device, non_blocking=True).float().view(-1)
        bad_rows = (
            self._row_has_nonfinite(prop)
            | self._row_has_nonfinite(image)
            | self._row_has_nonfinite(action)
            | ~torch.isfinite(reward)
        )
        if torch.any(bad_rows):
            self._sanitized_replay_rows += int(bad_rows.sum().item())
            is_first[bad_rows] = 1.0
        self.ring["prop"][idx] = prop
        self.ring["image"][idx] = image
        self.ring["action"][idx] = action
        self.ring["reward"][idx] = reward
        self.ring["is_first"][idx] = is_first
        targets = targets or {}
        if self.use_privileged_head:
            self.ring["privileged"][idx] = self._target_or_zeros(targets.get("privileged"), self.privi_dim, replay_device)
        if self.use_contact_head:
            self.ring["contact"][idx] = self._target_or_zeros(targets.get("contact"), self.contact_dim, replay_device)
        if self.use_object_state_head:
            self.ring["object_state"][idx] = self._target_or_zeros(targets.get("object_state"), self.object_state_dim, replay_device)
        self._write_idx = (idx + 1) % self.replay_capacity
        self._filled = min(self._filled + 1, self.replay_capacity)
        self._total_appended += self.num_envs

    def _target_or_zeros(self, value: Optional[torch.Tensor], dim: int, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.zeros((self.num_envs, dim), device=device)
        out = value.detach()
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        out = out.to(device, non_blocking=True).float()
        if out.shape[0] != self.num_envs:
            rows = min(out.shape[0], self.num_envs)
            padded = torch.zeros((self.num_envs, out.shape[-1]), device=device, dtype=out.dtype)
            padded[:rows] = out[:rows]
            out = padded
        if out.shape[-1] > dim:
            out = out[:, :dim]
        elif out.shape[-1] < dim:
            pad = dim - out.shape[-1]
            out = torch.cat([out, torch.zeros((self.num_envs, pad), device=device, dtype=out.dtype)], dim=-1)
        return self._sanitize_tensor(out)

    def _sanitize_tensor(self, value: torch.Tensor, clamp: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        if not torch.is_floating_point(value):
            return value
        value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if clamp is not None:
            value = torch.clamp(value, clamp[0], clamp[1])
        return value

    def _row_has_nonfinite(self, value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(value.shape[0], -1)
        return ~torch.isfinite(flat).all(dim=-1)

    def _latent_has_nonfinite(self, latent: Optional[Dict[str, torch.Tensor]]) -> bool:
        if latent is None:
            return False
        return any(torch.is_floating_point(value) and not torch.isfinite(value).all() for value in latent.values())

    def can_train(self) -> bool:
        return self._filled >= self.batch_length + 1 and self._total_appended >= self.train_after_warmup_steps

    def sample_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._filled < self.batch_length + 1:
            return None
        b = self.batch_size
        t = self.batch_length
        max_start = self._filled - t
        if max_start <= 0:
            return None
        env_ids = torch.randint(0, self.num_envs, (b,), device=self.device)
        starts = torch.randint(0, max_start, (b,), device=self.device)
        oldest = (self._write_idx - self._filled) % self.replay_capacity
        offsets = torch.arange(t, device=self.device)
        physical = (oldest + starts[:, None] + offsets[None, :]) % self.replay_capacity
        env_expand = env_ids[:, None].expand(-1, t)
        physical_cpu = physical.cpu()
        env_cpu = env_expand.cpu()
        if self.image_on_cpu:
            image = self.ring["image"][physical_cpu, env_cpu].to(self.device, non_blocking=True)
        else:
            image = self.ring["image"][physical, env_expand]
        if self.replay_on_cpu:
            prop = self.ring["prop"][physical_cpu, env_cpu].to(self.device, non_blocking=True)
            action = self.ring["action"][physical_cpu, env_cpu].to(self.device, non_blocking=True)
            reward = self.ring["reward"][physical_cpu, env_cpu].to(self.device, non_blocking=True)
            is_first = self.ring["is_first"][physical_cpu, env_cpu].to(self.device, non_blocking=True).clone()
            optional = {
                key: self.ring[key][physical_cpu, env_cpu].to(self.device, non_blocking=True)
                for key in ("privileged", "contact", "object_state")
                if key in self.ring
            }
        else:
            prop = self.ring["prop"][physical, env_expand]
            action = self.ring["action"][physical, env_expand]
            reward = self.ring["reward"][physical, env_expand]
            is_first = self.ring["is_first"][physical, env_expand].clone()
            optional = {
                key: self.ring[key][physical, env_expand]
                for key in ("privileged", "contact", "object_state")
                if key in self.ring
            }
        is_first[:, 0] = 1.0
        if self._reward_needs_event_dim and reward.dim() == 2:
            reward = reward.unsqueeze(-1)
        batch = {
            "prop": prop,
            "image": image,
            "action": action,
            "reward": reward,
            "is_first": is_first,
        }
        batch.update(optional)
        batch = self._sanitize_batch(batch)
        return batch

    def _sanitize_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                clamp = (-1.0, 1.0) if key == "action" else None
                batch[key] = self._sanitize_tensor(value, clamp=clamp)
        return batch

    def train_epoch(self) -> Dict[str, float]:
        if not self.can_train():
            return {}
        metrics_accum = []
        self.world_model.train()
        for _ in range(self.train_steps_per_epoch):
            batch = self.sample_batch()
            if batch is None:
                break
            batch = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            if not self._batch_is_finite(batch):
                self._dropped_bad_batches += 1
                continue
            try:
                _, _, metrics = self.world_model._train(batch)
            except ValueError as exc:
                if "invalid values" not in str(exc) and "nan" not in str(exc).lower():
                    raise
                self._dropped_bad_batches += 1
                print(f"[student_obs_wm] skipped non-finite WM batch: {exc}")
                continue
            except RuntimeError as exc:
                if "nan" not in str(exc).lower() and "inf" not in str(exc).lower():
                    raise
                self._dropped_bad_batches += 1
                print(f"[student_obs_wm] skipped unstable WM batch: {exc}")
                continue
            metrics_accum.append(metrics)
        if not metrics_accum:
            return {"skipped_bad_batches": float(self._dropped_bad_batches), "sanitized_replay_rows": float(self._sanitized_replay_rows)}
        out = {}
        for key in metrics_accum[0].keys():
            vals = [m[key] for m in metrics_accum if key in m]
            try:
                out[key] = float(torch.as_tensor(vals).float().mean().item())
            except Exception:
                pass
        out["skipped_bad_batches"] = float(self._dropped_bad_batches)
        out["sanitized_replay_rows"] = float(self._sanitized_replay_rows)
        return out

    def _batch_is_finite(self, batch: Dict[str, Any]) -> bool:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                if not torch.isfinite(value).all():
                    return False
        return True

    def state_dict(self):
        return {
            "world_model": self.world_model.state_dict(),
            "wm_optimizer": self.world_model._model_opt._opt.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True):
        if not state:
            return
        if "world_model" in state:
            self.world_model.load_state_dict(state["world_model"], strict=False)
        if load_optimizer and "wm_optimizer" in state:
            try:
                self.world_model._model_opt._opt.load_state_dict(state["wm_optimizer"])
            except Exception as exc:
                print(f"[student_obs_wm] optimizer restore skipped: {exc}")
