import torch

from env.tasks.intermimic_all_noref_vis import InterMimic_All_NoRef_Vis


class InterMimic_All_NoRef_Vis_StudentObs(InterMimic_All_NoRef_Vis):
    """No-reference visual distillation env that exposes student observations.

    The parent class already computes full teacher observations for expert
    actions. This subclass keeps that path intact, but makes the public
    observation buffer and task.num_obs reflect the no-ref visual student obs.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )
        self._sync_student_obs_size()

    def _sync_student_obs_size(self):
        obs_size = int(self.student_obs_buf.shape[1])
        self._num_obs = obs_size
        self.num_obs = obs_size
        self.cfg["env"]["numObs"] = obs_size
        self.cfg["env"]["numObservations"] = obs_size

    @staticmethod
    def _is_empty_env_ids(env_ids):
        if env_ids is None:
            return False
        try:
            return env_ids.numel() == 0
        except AttributeError:
            return len(env_ids) == 0

    def _compute_student_obs_from_current(self, env_ids=None):
        if self._is_empty_env_ids(env_ids):
            self.obs_buf = self.student_obs_buf
            return

        if env_ids is None:
            student_obs = self._curr_obs.clone()
            if self.remove_privileged_obs:
                student_obs = self._strip_privileged_from_obs(student_obs)

            if self.enable_camera and hasattr(self, "camera_history_buf"):
                visual_obs = self.camera_history_buf.reshape(self.num_envs, -1)
                student_obs = torch.cat([student_obs, visual_obs], dim=-1)

            if student_obs.shape[1] != self.student_obs_buf.shape[1]:
                self.student_obs_buf = torch.zeros(
                    (self.num_envs, student_obs.shape[1]),
                    device=self.device,
                    dtype=torch.float,
                )
                self._sync_student_obs_size()

            self.student_obs_buf[:] = student_obs
            self.obs_buf = self.student_obs_buf
            return

        student_obs = self._curr_obs[env_ids].clone()
        if self.remove_privileged_obs:
            student_obs = self._strip_privileged_from_obs(student_obs)

        if self.enable_camera and hasattr(self, "camera_history_buf"):
            visual_dim = self.camera_history_buf.shape[1] * self.camera_history_buf.shape[2]
            visual_obs = self.camera_history_buf[env_ids].reshape(len(env_ids), visual_dim)
            student_obs = torch.cat([student_obs, visual_obs], dim=-1)

        if student_obs.shape[1] != self.student_obs_buf.shape[1]:
            self.student_obs_buf = torch.zeros(
                (self.num_envs, student_obs.shape[1]),
                device=self.device,
                dtype=torch.float,
            )
            self._sync_student_obs_size()

        self.student_obs_buf[env_ids] = student_obs
        self.obs_buf = self.student_obs_buf

    def post_physics_step(self):
        super().post_physics_step()
        # Parent updates camera history after computing observations. Refresh the
        # student buffer so the policy receives the newest image feature source.
        self._compute_student_obs_from_current()
        self._sync_student_obs_size()

    def reset(self, env_ids=None):
        if self._is_empty_env_ids(env_ids):
            self.obs_buf = self.student_obs_buf
            return

        super().reset(env_ids=env_ids)
        self._compute_student_obs_from_current(env_ids)
        self._sync_student_obs_size()
        return
