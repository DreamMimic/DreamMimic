from gym import spaces
import numpy as np
import torch

from env.tasks.vec_task_wrappers import VecTaskPythonWrapper


class VecTaskDAggerStudentObsWrapper(VecTaskPythonWrapper):
    """DAgger wrapper that returns student obs while keeping teacher actions."""

    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)
        self._amp_obs_space = spaces.Box(
            np.ones(task.get_num_amp_obs()) * -np.Inf,
            np.ones(task.get_num_amp_obs()) * np.Inf,
        )

    @staticmethod
    def _is_empty_env_ids(env_ids):
        if env_ids is None:
            return False
        try:
            return env_ids.numel() == 0
        except AttributeError:
            return len(env_ids) == 0

    def reset(self, env_ids=None):
        if not self._is_empty_env_ids(env_ids):
            self.task.reset(env_ids)
        return (
            torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
            {"actions": self.task.action_buf, "mus": self.task.mu_buf},
        )

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.step(actions_tensor)
        return (
            torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
            self.task.rew_buf.to(self.rl_device),
            self.task.reset_buf.to(self.rl_device),
            self.task.extras,
            {"actions": self.task.action_buf, "mus": self.task.mu_buf},
        )
