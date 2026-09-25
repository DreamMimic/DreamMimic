import torch

from env.tasks.intermimic_all_noref_vis_student_obs import InterMimic_All_NoRef_Vis_StudentObs
from env.tasks.intermimic import InterMimic
from learning.world_model_adapter_student_obs import StudentObsWorldModelAdapter


class InterMimic_All_NoRef_Vis_StudentObs_WM(InterMimic_All_NoRef_Vis_StudentObs):
    """Student-observation visual distillation with a Dreamer/RSSM world model."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._wm_cfg = cfg.get("env", {}).get("worldModel", {})
        self._wm_prop_dim = int(self._wm_cfg.get("prop_dim", 1041))
        self._wm_image_h = int(self._wm_cfg.get("image_h", 32))
        self._wm_image_w = int(self._wm_cfg.get("image_w", 32))
        self._wm_image_c = int(self._wm_cfg.get("image_c", 2))
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        self.world_model_adapter = StudentObsWorldModelAdapter(
            num_envs=self.num_envs,
            num_actions=self.num_actions,
            prop_dim=self._wm_prop_dim,
            image_shape=(self._wm_image_h, self._wm_image_w, self._wm_image_c),
            wm_cfg=self._wm_cfg,
            device=torch.device(self.device),
        )
        self._wm_feature_dim = self.world_model_adapter.wm_feature_dim
        self.student_obs_buf = torch.zeros(
            (self.num_envs, self._wm_prop_dim + self._wm_feature_dim),
            device=self.device,
            dtype=torch.float,
        )
        self.obs_buf = self.student_obs_buf
        self._sync_student_obs_size()

    def _compute_observations(self, env_ids=None):
        """Compute teacher and student observations without mixing dimensions.

        Teacher obs remains the full 3198-dim reference observation used by the
        teacher policies. Student obs is only 1041-dim no-priv proprio plus the
        WM deter feature; raw image history is consumed by the WM, not exposed to
        the policy.
        """
        original_obs_buf = self.obs_buf
        self.obs_buf = self.obs_buf_teacher
        InterMimic._compute_observations(self, env_ids)
        self.obs_buf = original_obs_buf

        if hasattr(self, "world_model_adapter"):
            self._compute_student_obs_from_current(env_ids)
        else:
            # During construction the WM adapter is not available yet. Keep the
            # public student buffer at its configured size instead of letting the
            # visual parent resize it to raw-image observations.
            self.obs_buf = self.student_obs_buf
        return

    def _split_student_raw_obs(self, env_ids=None):
        if env_ids is None:
            raw = self._curr_obs.clone()
            num_rows = self.num_envs
        else:
            raw = self._curr_obs[env_ids].clone()
            num_rows = len(env_ids)

        if self.remove_privileged_obs:
            prop = self._strip_privileged_from_obs(raw)
        else:
            prop = raw
        if prop.shape[-1] > self._wm_prop_dim:
            prop = prop[:, : self._wm_prop_dim]
        elif prop.shape[-1] < self._wm_prop_dim:
            pad = self._wm_prop_dim - prop.shape[-1]
            prop = torch.cat([prop, torch.zeros(num_rows, pad, device=self.device, dtype=prop.dtype)], dim=-1)

        if self.enable_camera and hasattr(self, "camera_history_buf"):
            image_flat = self.camera_history_buf[:, -1]
            if env_ids is not None:
                image_flat = image_flat[env_ids]
            image = image_flat.reshape(num_rows, self._wm_image_h, self._wm_image_w, self._wm_image_c)
        else:
            image = torch.zeros(
                (num_rows, self._wm_image_h, self._wm_image_w, self._wm_image_c),
                device=self.device,
                dtype=torch.float,
            )
        return prop, image

    def _compute_student_obs_from_current(self, env_ids=None):
        if self._is_empty_env_ids(env_ids):
            self.obs_buf = self.student_obs_buf
            return

        if not hasattr(self, "world_model_adapter"):
            return super()._compute_student_obs_from_current(env_ids)

        prop, image = self._split_student_raw_obs(env_ids)
        if env_ids is None:
            actions_now = getattr(self, "actions", None)
            if actions_now is None:
                actions_now = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
            self.world_model_adapter.cache_prev_action(actions_now)
            wm_feature = self.world_model_adapter.step(prop, image)
            self.student_obs_buf[:] = torch.cat([prop, wm_feature], dim=-1)
            self.obs_buf = self.student_obs_buf
            return

        # For partial resets, use the cached feature until the next full step updates RSSM.
        wm_feature = self.world_model_adapter.feature[env_ids]
        self.student_obs_buf[env_ids] = torch.cat([prop, wm_feature], dim=-1)
        self.obs_buf = self.student_obs_buf

    def post_physics_step(self):
        super().post_physics_step()
        if not hasattr(self, "world_model_adapter"):
            return
        prop, image = self._split_student_raw_obs(None)
        actions_now = getattr(self, "actions", None)
        if actions_now is None:
            actions_now = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        is_first = (self.progress_buf <= 1).float()
        targets = self._extract_wm_fullhead_targets() if self._wm_fullheads_enabled() else None
        self.world_model_adapter.append(prop, image, actions_now, self.rew_buf, is_first, targets=targets)

    def _wm_fullheads_enabled(self):
        if not hasattr(self, "world_model_adapter"):
            return False
        adapter = self.world_model_adapter
        return bool(
            getattr(adapter, "use_fullheads", False)
            or getattr(adapter, "use_privileged_head", False)
            or getattr(adapter, "use_contact_head", False)
            or getattr(adapter, "use_object_state_head", False)
        )

    def _extract_wm_fullhead_targets(self):
        raw = getattr(self, "_curr_obs", None)
        if raw is None or raw.numel() == 0:
            return None
        try:
            num_bodies = int(self._rigid_body_pos.shape[1])
            dof_dim = int(self._dof_pos.shape[1])
            ts_dim = int(self._target_states.shape[1]) if hasattr(self, "_target_states") else 13
            ig_len = num_bodies * 3
            contact_len = num_bodies

            idx = 0
            idx += 3                           # root_pos
            idx += 4                           # root_rot
            idx += dof_dim                     # dof_pos
            idx += dof_dim                     # dof_vel
            idx += num_bodies * 3              # body_pos
            idx += num_bodies * 4              # body_rot
            idx += num_bodies * 3              # body_vel
            idx += num_bodies * 3              # body_rot_vel
            ts_start = idx
            ig_start = ts_start + ts_dim
            contact_start = ig_start + ig_len
            tar_contact_start = contact_start + contact_len
            tar_contact_end = tar_contact_start + 1
            if tar_contact_end > raw.shape[1]:
                return None

            object_state = raw[:, ts_start : ts_start + ts_dim].detach()
            ig = raw[:, ig_start : ig_start + ig_len].detach()
            contact = raw[:, tar_contact_start:tar_contact_end].detach()
            privileged = torch.cat([object_state, ig, contact], dim=-1)
            return {
                "privileged": privileged,
                "contact": contact,
                "object_state": object_state,
            }
        except Exception as exc:
            if not hasattr(self, "_wm_fullhead_extract_warned"):
                print(f"[student_obs_wm] full-head target extraction disabled after error: {exc}")
                self._wm_fullhead_extract_warned = True
            return None

    def reset(self, env_ids=None):
        if self._is_empty_env_ids(env_ids):
            self.obs_buf = self.student_obs_buf
            return

        if hasattr(self, "world_model_adapter"):
            if env_ids is None:
                reset_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            else:
                reset_ids = env_ids
            self.world_model_adapter.reset_envs(reset_ids)

        super().reset(env_ids=env_ids)
        self._compute_student_obs_from_current(env_ids)
        self._sync_student_obs_size()
