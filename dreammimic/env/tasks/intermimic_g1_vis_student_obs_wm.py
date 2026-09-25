import os
from collections import defaultdict

import numpy as np
import yaml
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch
from functorch import make_functional
from rl_games.algos_torch import torch_ext
from torch.func import vmap
from torchvision import transforms

from env.tasks.intermimic import InterMimic
from env.tasks.intermimic_g1gmrv4 import InterMimicG1
from learning import intermimic_models_teacher, intermimic_network_builder
from learning.world_model_adapter_student_obs import StudentObsWorldModelAdapter
from utils import torch_utils


def _all_checkpoint_paths(path):
    if os.path.isfile(path):
        return [path]
    paths = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".pth"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


class InterMimicG1_Vis_StudentObs_WM(InterMimicG1):
    """G1 teacher-student visual distillation with an env-owned RSSM world model."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg.get("env", {})
        sensor_cfg = cfg.get("sensor", {})
        self._student_wm_camera_all_envs = True
        self.global_step_counter = 0
        self.teacher_obs_size = int(env_cfg.get("teacherObsSize", env_cfg.get("numObs", 2153)))
        self.remove_privileged_obs = bool(env_cfg.get("removePrivilegedObs", True))
        self._wm_cfg = env_cfg.get("worldModel", {})
        self._wm_prop_dim = int(self._wm_cfg.get("prop_dim", 969))
        self._wm_image_h = int(self._wm_cfg.get("image_h", 32))
        self._wm_image_w = int(self._wm_cfg.get("image_w", 32))
        self._wm_image_c = int(self._wm_cfg.get("image_c", 2))
        self._student_ref_goal_cfg = env_cfg.get("studentRefGoal", {})
        self._student_ref_goal_enabled = bool(self._student_ref_goal_cfg.get("enabled", True))
        self._student_ref_goal_horizons = [int(x) for x in self._student_ref_goal_cfg.get("horizons", [1, 16])]

        self.enable_camera = bool(sensor_cfg.get("enableCamera", True))
        self.visual_input_type = sensor_cfg.get("visual_input_type", "depth_seg")
        self.depth_clip_lower = float(sensor_cfg.get("depth_clip_lower", 0.15))
        self.camera_history_len = int(sensor_cfg.get("history_len", 1))
        self.camera_sensor_dict = defaultdict(list)
        self.camera_handles = []
        if self.enable_camera:
            self.resize_transform = transforms.Resize((self._wm_image_h, self._wm_image_w))

        cfg["env"]["numObs"] = self.teacher_obs_size
        cfg["env"]["numObservations"] = self.teacher_obs_size
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        self.obs_buf_teacher = torch.zeros((self.num_envs, self.teacher_obs_size), device=self.device, dtype=torch.float)
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

        self.action_buf = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.mu_buf = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self._load_teacher_policies(cfg)

    def _sync_student_obs_size(self):
        obs_size = int(self.student_obs_buf.shape[1])
        self._num_obs = obs_size
        self.num_obs = obs_size
        self.cfg["env"]["numObs"] = obs_size
        self.cfg["env"]["numObservations"] = obs_size

    def _load_teacher_policies(self, cfg):
        config = {
            "actions_num": self.num_actions,
            "input_shape": (self.teacher_obs_size,),
            "num_seqs": self.num_envs,
            "value_size": 1,
        }
        with open(os.path.join(os.getcwd(), cfg["env"]["teacherPolicyCFG"]), "r") as f:
            cfg_teacher = yaml.load(f, Loader=yaml.SafeLoader)
        builder = intermimic_network_builder.InterMimicBuilder()
        builder.load(cfg_teacher["params"]["network"])
        network = intermimic_models_teacher.ModelInterMimicContinuous(builder)

        self.models = []
        self.functional_models = []
        self.params_list = []
        self.running_means = []
        self.running_vars = []
        self.models_subid = []
        for model_path in _all_checkpoint_paths(cfg["env"]["teacherPolicy"]):
            subid = int(os.path.basename(model_path).split(".")[0][3:])
            ckpt = torch_ext.load_checkpoint(model_path)
            model = network.build(config).to(self.device)
            model.load_state_dict(ckpt["model"])
            f_model, params = make_functional(model)
            self.models.append(model)
            self.functional_models.append(f_model)
            self.params_list.append(params)
            self.running_means.append(ckpt["running_mean_std"]["running_mean"].to(self.device).float())
            self.running_vars.append(ckpt["running_mean_std"]["running_var"].to(self.device).float())
            self.models_subid.append(subid)
        if not self.models:
            raise RuntimeError(f"No teacher checkpoints found in {cfg['env']['teacherPolicy']}")
        self.params_zip = list(zip(*self.params_list))
        self.stacked_params = tuple(torch.stack(p_tensors, dim=0) for p_tensors in self.params_zip)
        self.running_means_all = torch.stack(self.running_means).float()
        self.running_vars_all = torch.stack(self.running_vars).float()

    def create_sim(self):
        if self.enable_camera and getattr(self, "graphics_device_id", None) == -1:
            self.graphics_device_id = self.device_id
        if self.enable_camera and not isinstance(getattr(self, "camera_handles", None), list):
            self.camera_handles = []
        super().create_sim()
        if self.enable_camera:
            self._setup_camera_tensors()

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        if not getattr(self, "_student_wm_camera_all_envs", False):
            return super()._build_env(env_id, env_ptr, humanoid_asset)
        old_enable_camera = getattr(self, "enable_camera", False)
        self.enable_camera = False
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self.enable_camera = old_enable_camera
        if self.enable_camera:
            if not isinstance(self.camera_handles, list):
                self.camera_handles = []
            self.camera_handles.append(self._create_onboard_camera(env_ptr, self.humanoid_handles[env_id]))

    def _create_onboard_camera(self, env_handle, actor_handle):
        camera_cfg = self.cfg["sensor"]["onboard_camera"]
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = int(camera_cfg["resolution"][1])
        camera_props.width = int(camera_cfg["resolution"][0])
        if camera_cfg.get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = float(camera_cfg["horizontal_fov"])
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        if camera_handle is None:
            return None

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*camera_cfg["position"])
        local_transform.r = gymapi.Quat.from_euler_zyx(*camera_cfg["rotation"])
        for body_name in (camera_cfg.get("attach_link", "d435_link"), "mid360_link", "head_link", "torso_link", "pelvis"):
            body_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, body_name)
            if body_handle != -1:
                self.gym.attach_camera_to_body(camera_handle, env_handle, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                return camera_handle
        return None

    def _setup_camera_tensors(self):
        self.camera_sensor_dict = defaultdict(list)
        for env_i, env_handle in enumerate(self.envs):
            if env_i >= len(self.camera_handles) or self.camera_handles[env_i] is None:
                self.camera_sensor_dict["forward_depth"].append(None)
                self.camera_sensor_dict["forward_seg"].append(None)
                continue
            try:
                depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, self.camera_handles[env_i], gymapi.IMAGE_DEPTH)
                seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, self.camera_handles[env_i], gymapi.IMAGE_SEGMENTATION)
                self.camera_sensor_dict["forward_depth"].append(gymtorch.wrap_tensor(depth_tensor))
                self.camera_sensor_dict["forward_seg"].append(gymtorch.wrap_tensor(seg_tensor))
            except Exception:
                self.camera_sensor_dict["forward_depth"].append(None)
                self.camera_sensor_dict["forward_seg"].append(None)

    def _update_camera_history(self):
        if not self.enable_camera or not self.camera_sensor_dict.get("forward_depth"):
            return
        if any(t is None for t in self.camera_sensor_dict["forward_depth"]) or any(t is None for t in self.camera_sensor_dict["forward_seg"]):
            return
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            depth = torch.stack(self.camera_sensor_dict["forward_depth"]).to(self.device).clone()
            seg = torch.stack(self.camera_sensor_dict["forward_seg"]).to(self.device).clone()
        finally:
            self.gym.end_access_image_tensors(self.sim)
        depth[depth < -2] = -2
        depth[depth > -self.depth_clip_lower] = 0
        depth = torch.clamp(-depth / 2.0, 0.0, 1.0)
        mask = (seg >= int(self.cfg.get("sensor", {}).get("segmentation_id", 2))).float()
        image = torch.stack([depth, mask], dim=-1)
        image = image.permute(0, 3, 1, 2)
        image = self.resize_transform(image).permute(0, 2, 3, 1).contiguous()
        if not hasattr(self, "camera_history_buf"):
            flat_dim = self._wm_image_h * self._wm_image_w * self._wm_image_c
            self.camera_history_buf = torch.zeros((self.num_envs, self.camera_history_len, flat_dim), device=self.device)
        flat = image.reshape(self.num_envs, -1)
        self.camera_history_buf = torch.where(
            (self.progress_buf <= 1)[:, None, None],
            torch.stack([flat] * self.camera_history_len, dim=1),
            self.camera_history_buf,
        )
        self.camera_history_buf = torch.cat([self.camera_history_buf[:, 1:], flat.unsqueeze(1)], dim=1)

    def _compute_observations(self, env_ids=None):
        if not hasattr(self, "obs_buf_teacher"):
            return super()._compute_observations(env_ids)
        original_obs_buf = self.obs_buf
        self.obs_buf = self.obs_buf_teacher
        InterMimicG1._compute_observations(self, env_ids)
        self.obs_buf = original_obs_buf
        if hasattr(self, "world_model_adapter"):
            self._compute_student_obs_from_current(env_ids)
        else:
            self.obs_buf = original_obs_buf

    def _strip_privileged_from_obs(self, obs):
        num_bodies = int(self._rigid_body_pos.shape[1])
        dof_dim = int(self._dof_pos.shape[1])
        ts_dim = int(self._target_states.shape[1]) if hasattr(self, "_target_states") else 13
        idx = 0
        idx += 3 + 4 + dof_dim + dof_dim
        idx += num_bodies * 3 + num_bodies * 4 + num_bodies * 3 + num_bodies * 3
        ts_start = idx
        ig_start = ts_start + ts_dim
        contact_start = ig_start + num_bodies * 3
        tar_contact_start = contact_start + num_bodies
        mask = torch.ones(obs.shape[-1], dtype=torch.bool, device=obs.device)
        mask[ts_start:ts_start + ts_dim] = False
        mask[ig_start:ig_start + num_bodies * 3] = False
        if tar_contact_start < obs.shape[-1]:
            mask[tar_contact_start:tar_contact_start + 1] = False
        return obs[:, mask]

    def _compute_student_ref_goal(self, env_ids, num_rows):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        data_id = self.data_id[env_ids]
        curr_root_pos = self._rigid_body_pos[env_ids, 0, :]
        curr_root_rot = self._rigid_body_rot[env_ids, 0, :]
        heading_inv = torch_utils.calc_heading_quat_inv(curr_root_rot)
        goals = []
        for horizon in self._student_ref_goal_horizons:
            next_ts = torch.clamp(self.progress_buf[env_ids] + int(horizon), max=self.max_episode_length[data_id] - 1)
            ref_obs = self.hoi_data[data_id, next_ts]
            ref_root_pos = self.extract_data_component("root_pos", obs=ref_obs)
            ref_root_rot = self.extract_data_component("root_rot", obs=ref_obs)
            ref_body_pos = self.extract_data_component("body_pos", obs=ref_obs).view(num_rows, -1, 3)[:, self._key_body_ids_gt, :]
            ref_obj_pos = self.extract_data_component("obj_pos", obs=ref_obs)
            ref_obj_rot = self.extract_data_component("obj_rot", obs=ref_obs)
            root_delta = torch_utils.quat_rotate(heading_inv, ref_root_pos - curr_root_pos)
            root_rot_local = torch_utils.quat_mul(heading_inv, ref_root_rot)
            key_rel = ref_body_pos - curr_root_pos.unsqueeze(1)
            heading_expand = heading_inv.unsqueeze(1).repeat(1, key_rel.shape[1], 1).reshape(-1, 4)
            key_rel = torch_utils.quat_rotate(heading_expand, key_rel.reshape(-1, 3)).reshape(num_rows, -1)
            obj_rel = torch_utils.quat_rotate(heading_inv, ref_obj_pos - curr_root_pos)
            obj_rot_local = torch_utils.quat_mul(heading_inv, ref_obj_rot)
            goals.append(torch.cat([root_delta, root_rot_local, key_rel, obj_rel, obj_rot_local], dim=-1))
        goal = torch.cat(goals, dim=-1)
        return torch.clamp(torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

    def _split_student_raw_obs(self, env_ids=None):
        if env_ids is None:
            raw = self._curr_obs.clone()
            num_rows = self.num_envs
        else:
            raw = self._curr_obs[env_ids].clone()
            num_rows = len(env_ids)
        prop = self._strip_privileged_from_obs(raw) if self.remove_privileged_obs else raw
        if self._student_ref_goal_enabled:
            prop = torch.cat([prop, self._compute_student_ref_goal(env_ids, num_rows)], dim=-1)
        if prop.shape[-1] > self._wm_prop_dim:
            prop = prop[:, : self._wm_prop_dim]
        elif prop.shape[-1] < self._wm_prop_dim:
            prop = torch.cat([prop, torch.zeros(num_rows, self._wm_prop_dim - prop.shape[-1], device=self.device)], dim=-1)
        if self.enable_camera and hasattr(self, "camera_history_buf"):
            image_flat = self.camera_history_buf[:, -1]
            if env_ids is not None:
                image_flat = image_flat[env_ids]
            image = image_flat.reshape(num_rows, self._wm_image_h, self._wm_image_w, self._wm_image_c)
        else:
            image = torch.zeros((num_rows, self._wm_image_h, self._wm_image_w, self._wm_image_c), device=self.device)
        return prop, image

    def _compute_student_obs_from_current(self, env_ids=None):
        prop, image = self._split_student_raw_obs(env_ids)
        if env_ids is None:
            actions_now = getattr(self, "actions", None)
            if actions_now is None:
                actions_now = torch.zeros((self.num_envs, self.num_actions), device=self.device)
            self.world_model_adapter.cache_prev_action(actions_now)
            wm_feature = self.world_model_adapter.step(prop, image)
            self.student_obs_buf[:] = torch.cat([prop, wm_feature], dim=-1)
        else:
            wm_feature = self.world_model_adapter.feature[env_ids]
            self.student_obs_buf[env_ids] = torch.cat([prop, wm_feature], dim=-1)
        self.obs_buf = self.student_obs_buf

    def _extract_wm_fullhead_targets(self):
        raw = getattr(self, "_curr_obs", None)
        if raw is None:
            return None
        num_bodies = int(self._rigid_body_pos.shape[1])
        dof_dim = int(self._dof_pos.shape[1])
        ts_dim = int(self._target_states.shape[1]) if hasattr(self, "_target_states") else 13
        idx = 3 + 4 + dof_dim + dof_dim + num_bodies * (3 + 4 + 3 + 3)
        ts_start = idx
        ig_start = ts_start + ts_dim
        contact_start = ig_start + num_bodies * 3
        tar_contact_start = contact_start + num_bodies
        object_state = raw[:, ts_start:ts_start + ts_dim].detach()
        ig = raw[:, ig_start:ig_start + num_bodies * 3].detach()
        contact = raw[:, tar_contact_start:tar_contact_start + 1].detach()
        return {
            "privileged": torch.cat([object_state, ig, contact], dim=-1),
            "contact": contact,
            "object_state": object_state,
        }

    def _wm_fullheads_enabled(self):
        return bool(getattr(self.world_model_adapter, "use_fullheads", False))

    def single_model_forward(self, params, obs, mean, var):
        curr_obs = torch.clamp((obs - mean) / torch.sqrt(var + 1e-5), min=-5.0, max=5.0)
        input_dict = {"is_train": False, "prev_actions": None, "obs": curr_obs, "rnn_states": None}
        res_dict = self.functional_models[0](params, input_dict)
        sigma = res_dict["sigmas"] if "sigmas" in res_dict else res_dict["log_std"].exp()
        return res_dict["mus"], sigma

    def _update_teacher_actions(self):
        with torch.no_grad():
            batched_forward = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            mus_all, sigma_all = batched_forward(
                self.stacked_params,
                self.obs_buf_teacher.unsqueeze(0).repeat(self.running_means_all.shape[0], 1, 1),
                self.running_means_all,
                self.running_vars_all,
            )
            distr = torch.distributions.Normal(mus_all, sigma_all)
            teacher_actions_all = torch.clamp(distr.sample(), min=-1.0, max=1.0)
            self.action_buf = teacher_actions_all[self.model_indices, self.sample_indices]
            self.mu_buf = mus_all[self.model_indices, self.sample_indices]

    def _update_teacher_indices(self):
        id_to_index = {subid: i for i, subid in enumerate(self.models_subid)}
        self.model_indices = torch.tensor([id_to_index[int(subid.item())] for subid in self.dataset_id], dtype=torch.long, device=self.device)
        self.sample_indices = torch.arange(self.dataset_id.shape[0], device=self.device)

    def post_physics_step(self):
        super().post_physics_step()
        if self.enable_camera:
            self._update_camera_history()
            self._compute_student_obs_from_current(None)
        prop, image = self._split_student_raw_obs(None)
        actions_now = getattr(self, "actions", None)
        if actions_now is None:
            actions_now = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        is_first = (self.progress_buf <= 1).float()
        targets = self._extract_wm_fullhead_targets() if self._wm_fullheads_enabled() else None
        self.world_model_adapter.append(prop, image, actions_now, self.rew_buf, is_first, targets=targets)

    def step(self, weights):
        self.pre_physics_step(weights)
        self.global_step_counter += 1
        self._physics_step()
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)
        self.post_physics_step()
        self._update_teacher_actions()

    def reset(self, env_ids=None):
        if hasattr(self, "world_model_adapter"):
            reset_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) if env_ids is None else env_ids
            self.world_model_adapter.reset_envs(reset_ids)
        InterMimic.reset(self, env_ids=env_ids)
        self._update_teacher_indices()
        self._compute_student_obs_from_current(env_ids)
        self._update_teacher_actions()
