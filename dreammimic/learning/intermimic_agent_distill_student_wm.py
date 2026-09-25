import torch

from learning.intermimic_agent_distill import InterMimicAgentDistill


class InterMimicAgentDistillStudentWM(InterMimicAgentDistill):
    """DAgger/RL distillation agent that trains the env-owned world model."""

    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.enable_multistep_latent_distill = bool(config.get("enable_multistep_latent_distill", False))
        self.latent_distill_coef = float(config.get("latent_distill_coef", 0.1))
        self.latent_distill_horizon = int(config.get("latent_distill_horizon", 3))
        self.latent_distill_stoch_coef = float(config.get("latent_distill_stoch_coef", 0.1))
        if self.enable_multistep_latent_distill:
            print("  Multi-step latent distillation enabled")
            print(f"  Latent horizon: {self.latent_distill_horizon}")
            print(f"  Latent coef: {self.latent_distill_coef}")
            print(f"  Latent stoch coef: {self.latent_distill_stoch_coef}")

    def train_epoch(self):
        train_info = super().train_epoch()
        adapter = self._get_wm_adapter()
        if adapter is not None:
            wm_metrics = adapter.train_epoch()
            if wm_metrics:
                for key, value in wm_metrics.items():
                    train_info[f"wm/{key}"] = value
                if hasattr(self, "writer") and self.writer is not None:
                    for key, value in wm_metrics.items():
                        self.writer.add_scalar(f"world_model/{key}", value, self.epoch_num)
        return train_info

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        adapter = self._get_wm_adapter()
        if adapter is not None:
            state["student_world_model"] = adapter.state_dict()
        return state

    def set_full_state_weights(self, weights):
        super().set_full_state_weights(weights)
        adapter = self._get_wm_adapter()
        if adapter is not None and "student_world_model" in weights:
            adapter.load_state_dict(weights["student_world_model"], load_optimizer=not bool(getattr(self, "is_test", False)))

    def _get_wm_adapter(self):
        try:
            return self.vec_env.env.task.world_model_adapter
        except AttributeError:
            return None

    def _init_extra_distill_tensors(self, batch_shape):
        if not self.enable_multistep_latent_distill:
            return
        adapter = self._get_wm_adapter()
        if adapter is None:
            return
        dynamics = adapter.world_model.dynamics
        deter_dim = int(getattr(dynamics, "_deter", adapter.wm_deter_dim))
        stoch_dim = int(getattr(dynamics, "_stoch", 32))
        discrete_dim = int(getattr(dynamics, "_discrete", 0))
        self._latent_stoch_shape = (stoch_dim, discrete_dim) if discrete_dim else (stoch_dim,)
        self.experience_buffer.tensor_dict["wm_latent_deter"] = torch.zeros(
            (*batch_shape, deter_dim), dtype=torch.float32, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict["wm_latent_stoch"] = torch.zeros(
            (*batch_shape, *self._latent_stoch_shape), dtype=torch.float32, device=self.ppo_device
        )
        self.tensor_list += ["wm_latent_deter", "wm_latent_stoch"]

    def _record_extra_distill_step(self, step_idx):
        if not self.enable_multistep_latent_distill:
            return
        adapter = self._get_wm_adapter()
        if adapter is None:
            return
        latent = getattr(adapter, "_wm_latent", None)
        if latent is None:
            return
        if "deter" in latent:
            self.experience_buffer.update_data("wm_latent_deter", step_idx, latent["deter"].detach().to(self.ppo_device))
        if "stoch" in latent:
            self.experience_buffer.update_data("wm_latent_stoch", step_idx, latent["stoch"].detach().to(self.ppo_device))

    def _compute_extra_distill_loss(self, student_actions, expert_actions, input_dict):
        if not self.enable_multistep_latent_distill or self.latent_distill_coef <= 0:
            return None
        adapter = self._get_wm_adapter()
        if adapter is None or adapter.world_model is None:
            return None
        start_deter = input_dict.get("wm_latent_deter", None)
        start_stoch = input_dict.get("wm_latent_stoch", None)
        if start_deter is None or start_stoch is None:
            return None
        horizon = max(int(self.latent_distill_horizon), 1)
        wm = adapter.world_model
        dynamics = wm.dynamics

        prev_requires_grad = [p.requires_grad for p in wm.parameters()]
        for p in wm.parameters():
            p.requires_grad_(False)
        try:
            student_state = {
                "deter": start_deter.detach(),
                "stoch": start_stoch.detach(),
            }
            teacher_state = {
                "deter": start_deter.detach(),
                "stoch": start_stoch.detach(),
            }
            student_action = student_actions
            teacher_action = expert_actions.detach()
            total = student_actions.new_tensor(0.0)
            for _ in range(horizon):
                student_state = dynamics.img_step(student_state, student_action, sample=False)
                with torch.no_grad():
                    teacher_state = dynamics.img_step(teacher_state, teacher_action, sample=False)
                deter_loss = (student_state["deter"] - teacher_state["deter"].detach()).pow(2).mean(dim=-1)
                stoch_student = student_state["stoch"].reshape(student_state["stoch"].shape[0], -1)
                stoch_teacher = teacher_state["stoch"].detach().reshape(teacher_state["stoch"].shape[0], -1)
                stoch_loss = (stoch_student - stoch_teacher).pow(2).mean(dim=-1)
                total = total + (deter_loss + self.latent_distill_stoch_coef * stoch_loss).mean()
            return self.latent_distill_coef * total / float(horizon)
        finally:
            for p, requires_grad in zip(wm.parameters(), prev_requires_grad):
                p.requires_grad_(requires_grad)

    def _prepare_extra_distill_dataset(self, batch_dict):
        if not self.enable_multistep_latent_distill:
            return
        if "wm_latent_deter" in batch_dict:
            self.dataset.values_dict["wm_latent_deter"] = batch_dict["wm_latent_deter"]
        if "wm_latent_stoch" in batch_dict:
            self.dataset.values_dict["wm_latent_stoch"] = batch_dict["wm_latent_stoch"]

    def _log_train_info(self, train_info, frame):
        wm_keys = [key for key in train_info.keys() if key.startswith("wm/")]
        stripped = {key: train_info.pop(key) for key in wm_keys}
        super()._log_train_info(train_info, frame)
        for key, value in stripped.items():
            try:
                scalar = float(torch.as_tensor(value).float().mean().item())
            except Exception:
                continue
            self.writer.add_scalar(f"world_model/{key[3:]}", scalar, frame)
            train_info[key] = value
