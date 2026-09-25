import os

import numpy as np
import torch
try:
    import cv2
except Exception:
    cv2 = None

from env.tasks.intermimic_all_noref_vis_student_obs import InterMimic_All_NoRef_Vis_StudentObs
from env.tasks.intermimic import InterMimic
from learning.world_model_adapter_student_obs import StudentObsWorldModelAdapter
from utils import torch_utils


class InterMimic_All_NoRef_Vis_StudentObs_WM(InterMimic_All_NoRef_Vis_StudentObs):
    """Student-observation visual distillation with a Dreamer/RSSM world model."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._wm_cfg = cfg.get("env", {}).get("worldModel", {})
        self._wm_prop_dim = int(self._wm_cfg.get("prop_dim", 1041))
        self._wm_image_h = int(self._wm_cfg.get("image_h", 32))
        self._wm_image_w = int(self._wm_cfg.get("image_w", 32))
        self._wm_image_c = int(self._wm_cfg.get("image_c", 2))
        env_cfg = cfg.get("env", {})
        self._record_video = bool(env_cfg.get("recordVideo", False))
        self._record_env_idx = int(env_cfg.get("recordEnvIdx", 0))
        self._record_dir = env_cfg.get("recordDir", "record_video")
        self._record_every = max(1, int(env_cfg.get("recordEvery", 1)))
        self._record_max_frames = max(1, int(env_cfg.get("recordMaxFrames", 300)))
        self._record_fps = max(1, int(env_cfg.get("recordFps", 30)))
        self._record_frames = self._new_record_frame_buffer()
        self._record_episode_idx = 0
        ref_goal_cfg = cfg.get("env", {}).get("studentRefGoal", {})
        self._student_ref_goal_enabled = bool(ref_goal_cfg.get("enabled", False))
        self._student_ref_goal_horizons = [int(x) for x in ref_goal_cfg.get("horizons", [1, 16])]
        object_goal_cfg = cfg.get("env", {}).get("studentObjectGoal", {})
        self._student_object_goal_enabled = bool(object_goal_cfg.get("enabled", False))
        self._student_object_goal_horizons = object_goal_cfg.get("horizons", None)
        if self._student_object_goal_horizons is not None:
            self._student_object_goal_horizons = [
                int(x) for x in self._student_object_goal_horizons
            ]
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
        if self._student_ref_goal_enabled:
            ref_goal = self._compute_student_ref_goal(env_ids, num_rows)
            prop = torch.cat([prop, ref_goal], dim=-1)
        if self._student_object_goal_enabled:
            object_goal = self._compute_student_object_goal(env_ids)
            prop = torch.cat([prop, object_goal], dim=-1)
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

    def _compute_student_ref_goal(self, env_ids, num_rows):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        data_id = self.data_id[env_ids]
        ref_index = self.ref_index[env_ids]
        curr_root_pos = self._rigid_body_pos[env_ids, 0, :]
        curr_root_rot = self._rigid_body_rot[env_ids, 0, :]
        heading_inv = torch_utils.calc_heading_quat_inv(curr_root_rot)
        key_ids = self._key_body_ids
        goals = []
        for horizon in self._student_ref_goal_horizons:
            next_ts = torch.clamp(
                self.progress_buf[env_ids] + int(horizon),
                max=self.max_episode_length[data_id] - 1,
            )
            ref_root_pos = self.extract_ref_component("root_pos", data_id, ref_index, next_ts)
            ref_root_rot = self.extract_ref_component("root_rot", data_id, ref_index, next_ts)
            ref_body_pos = self.extract_data_component(
                "body_pos", obs=self.hoi_data[data_id, next_ts]
            ).view(num_rows, -1, 3)[:, key_ids, :]
            ref_obj_pos = self.extract_ref_component("obj_pos", data_id, ref_index, next_ts)
            ref_obj_rot = self.extract_ref_component("obj_rot", data_id, ref_index, next_ts)

            root_delta = torch_utils.quat_rotate(heading_inv, ref_root_pos - curr_root_pos)
            root_rot_local = torch_utils.quat_mul(heading_inv, ref_root_rot)
            key_rel = ref_body_pos - curr_root_pos.unsqueeze(1)
            heading_expand = heading_inv.unsqueeze(1).repeat(1, key_rel.shape[1], 1).reshape(-1, 4)
            key_rel = torch_utils.quat_rotate(heading_expand, key_rel.reshape(-1, 3)).reshape(num_rows, -1)
            obj_rel = torch_utils.quat_rotate(heading_inv, ref_obj_pos - curr_root_pos)
            obj_rot_local = torch_utils.quat_mul(heading_inv, ref_obj_rot)
            goals.append(torch.cat([root_delta, root_rot_local, key_rel, obj_rel, obj_rot_local], dim=-1))
        goal = torch.cat(goals, dim=-1) if goals else torch.zeros((num_rows, 0), device=self.device)
        return torch.clamp(torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

    def _compute_student_object_goal(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        data_id = self.data_id[env_ids]
        # Use the original reference buffer for the episode-level goal pose, not
        # any physical-buffer state inserted by the curriculum.
        ref_index = torch.zeros_like(data_id)
        curr_root_pos = self._rigid_body_pos[env_ids, 0, :]
        curr_root_rot = self._rigid_body_rot[env_ids, 0, :]
        heading_inv = torch_utils.calc_heading_quat_inv(curr_root_rot)
        if self._student_object_goal_horizons is None:
            goal_ts_list = [self.max_episode_length[data_id] - 1]
        else:
            goal_ts_list = [
                torch.clamp(
                    self.progress_buf[env_ids] + int(horizon),
                    max=self.max_episode_length[data_id] - 1,
                )
                for horizon in self._student_object_goal_horizons
            ]
        goals = []
        for goal_ts in goal_ts_list:
            goal_obj_pos = self.extract_ref_component("obj_pos", data_id, ref_index, goal_ts)
            goal_obj_rot = self.extract_ref_component("obj_rot", data_id, ref_index, goal_ts)
            goal_obj_pos_local = torch_utils.quat_rotate(heading_inv, goal_obj_pos - curr_root_pos)
            goal_obj_rot_local = torch_utils.quat_mul(heading_inv, goal_obj_rot)
            goals.append(torch.cat([goal_obj_pos_local, goal_obj_rot_local], dim=-1))
        goal = torch.cat(goals, dim=-1)
        return torch.clamp(torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

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
        self._record_wm_video_frame(image)
        self._maybe_save_finished_record_episode()

    def _new_record_frame_buffer(self):
        return {
            "depth": [],
            "seg": [],
            "recon_depth": [],
            "recon_seg": [],
        }

    def _record_wm_video_frame(self, image):
        if not self._record_video or cv2 is None:
            return
        if len(self._record_frames["depth"]) >= self._record_max_frames:
            return
        if int(self.global_step_counter) % self._record_every != 0:
            return
        if image is None or image.shape[-1] < 2:
            return
        env_idx = min(max(self._record_env_idx, 0), image.shape[0] - 1)
        recon = self.world_model_adapter.reconstruct_current_image()
        if recon is None or recon.shape[-1] < 2 or env_idx >= recon.shape[0]:
            return

        truth = image[env_idx].detach().float().cpu().numpy()
        pred = recon[env_idx].detach().float().cpu().numpy()
        self._record_frames["depth"].append(self._to_uint8_frame(truth[..., 0], binary=False))
        self._record_frames["seg"].append(self._to_uint8_frame(truth[..., 1], binary=True))
        self._record_frames["recon_depth"].append(self._to_uint8_frame(pred[..., 0], binary=False))
        self._record_frames["recon_seg"].append(self._to_uint8_frame(pred[..., 1], binary=True))

    def _maybe_save_finished_record_episode(self):
        if not self._record_video:
            return
        env_idx = min(max(self._record_env_idx, 0), self.num_envs - 1)
        if bool(self.reset_buf[env_idx].item()):
            self._save_record_episode(partial=False)

    def _to_uint8_frame(self, frame, binary=False):
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        if binary:
            frame = (frame > 0.5).astype(np.float32)
        else:
            frame = frame.astype(np.float32)
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)

    def save_recorded_wm_videos(self):
        if not self._record_video:
            return
        self._save_record_episode(partial=True)

    def _save_record_episode(self, partial=False):
        frame_count = len(self._record_frames["depth"])
        if frame_count == 0:
            return
        if cv2 is None:
            print("[record_video] OpenCV is unavailable; cannot write mp4 videos.")
            return
        os.makedirs(self._record_dir, exist_ok=True)
        suffix = "partial" if partial else f"ep{self._record_episode_idx:04d}"
        for name, frames in self._record_frames.items():
            path = os.path.join(self._record_dir, f"env{self._record_env_idx}_{suffix}_{name}.mp4")
            self._write_grayscale_video(path, frames)
            print(f"[record_video] Saved {name} video: {path}")
        self._record_frames = self._new_record_frame_buffer()
        if not partial:
            self._record_episode_idx += 1

    def _write_grayscale_video(self, path, frames):
        h, w = frames[0].shape
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self._record_fps),
            (w, h),
            True,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()

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
