import copy
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid_g1 import Humanoid_G1
from env.tasks.intermimic import InterMimic, compute_sdf
from env.tasks.intermimic_g1gmrv4 import InterMimicG1 as _InterMimicG1GMRv4
from enum import Enum
import numpy as np
import os
import joblib
from isaacgym import gymapi
from env.tasks.humanoid import *
import trimesh
import xml.etree.ElementTree as ET

contact_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,25,26,28,
              29,31,32,34,35,37,38,39,40,41,43,44,46,47,49,50,52,53,54]

class InterMimicG1(Humanoid_G1, InterMimic):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.motion_file = cfg['env']['motion_file']
        self.sub = cfg['env']['dataSub']
        self.dataset = cfg['env']['motion_file'].split('/')[-1].split('_')[0]
        self.motion_data = joblib.load(self.motion_file)
        if self.sub:
            self.motion_data = {key: value for key, value in self.motion_data.items() if any(s in key for s in self.sub)}
        
        self.object_name = [str(value['obj_name']) for key, value in self.motion_data.items()]
        object_name_set = sorted(list(set(self.object_name)))
        self.object_id = to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long).cuda()
        self.obj2motion = torch.stack([self.object_id == k for k in range(len(object_name_set))], dim=0)
        self.object_name = object_name_set
        self.num_motions = len(self.motion_data)
        #TODO now only fit for omomo dataset
        self.dataset_index = to_torch([int(key.split('_')[1][3:]) for key in self.motion_data.keys()], dtype=torch.long).cuda()

        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        # self.motion_file = sorted([os.path.join(self.motion_file, data_path) for data_path in motion_file if data_path.split('_')[0] in cfg['env']['dataSub']])
        
        # root trans/rot + dof pos/vel + body pos/pos vel/rot/rot vel + object pos/pos vel/rot/rot vel + ig + contact human + contact obj
        self.ref_hoi_obs_size = 7 + 43 * 2 + 44 * 13 + 13 + 44 * 3 + 44 + 1

        self.hoi_data = self._load_motion(self.motion_file, startk=1, initk=15)
        self.scaling = cfg['env']['scaling']
        self.init_root_height = cfg['env']['initRootHeight']
        self.init_dof = torch.cat([to_torch([-0.1, 0, 0.0, 0.3, -0.2, 0, -0.1, 0, 0.0, 0.3, -0.2, 0, 0, 0, 0, 
                                             0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float),
                                   to_torch([0, 0, 0, 0.5], device=self.device, dtype=torch.float),
                                   to_torch([0] * (self._num_actions_hand + self._num_actions_wrist), device=self.device, dtype=torch.float)])
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.kinematic_reset = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.contact_reset = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._curr_reward = torch.zeros([self.num_envs, cfg['env']['rolloutLength']], device=self.device, dtype=torch.float)
        self._sum_reward = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._curr_state = torch.zeros([self.num_envs, cfg['env']['rolloutLength'], 112], device=self.device, dtype=torch.float)
        self._build_target_tensors()

        self.reward_scales = cfg["env"]["reward"]["rewardscales"]
        self.penalty_scales = cfg["env"]["reward"]["penaltyscales"]
        self.soft_dof_pos_limit = cfg["env"]["reward"]["soft_dof_pos_limit"]
        self.soft_dof_vel_limit = cfg["env"]["reward"]["soft_dof_vel_limit"]
        self.soft_torque_limit = cfg["env"]["reward"]["soft_torque_limit"]
        self.base_height_target = cfg["env"]["reward"]["base_height_target"]
        self.max_contact_force = cfg["env"]["reward"]["max_contact_force"]
    
        self._prepare_penalty_function()
        
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self._dof_vel)
        self.last_root_vel = torch.zeros_like(self._humanoid_root_states[:, 7:13])
        self.last_root_pos = torch.zeros_like(self._humanoid_root_states[:, 0:3])

        self.left_contact_hand_ids = list(range(23,29)) 
        self.right_contact_hand_ids = list(range(37,43))
        self.feet_indices = torch.tensor([self._key_body_ids[i] for i in range(len(self.key_bodies)) if 'ankle' in self.key_bodies[i]], dtype=torch.long, device=self.device)        
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_quat = self._humanoid_root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self._humanoid_root_states[:, 7:10]) # normalization
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self._humanoid_root_states[:, 10:13])
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)


        return


    def _prepare_penalty_function(self):
        """ Prepares a list of penalty functions, whcih will be called to compute the total penalty.
            Looks for self._penalty_<PENALTY_NAME>, where <PENALTY_NAME> are names of all non zero penalty scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt

        for key in list(self.penalty_scales.keys()):
            scale = self.penalty_scales[key]
            if scale==0:
                self.penalty_scales.pop(key) 
            else:
                self.penalty_scales[key] *= self.dt
        # prepare list of functions
        self.penalty_functions = []
        self.penalty_names = []
        for name, scale in self.penalty_scales.items():
            if name=="termination":
                continue
            self.penalty_names.append(name)
            name = '_penalty_' + name
            self.penalty_functions.append(getattr(self, name))

        # penalty episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.penalty_scales.keys()}


    def _load_target_asset(self): # smplx
        asset_root = f"dreammimic/data/assets/objects/{self.dataset}/"
        self._target_asset = []
        points_num = []
        self.object_points = []
        for i, object_name in enumerate(self.object_name):

            asset_file = object_name + ".urdf"
            obj_name = object_name.split('_')[0]
            obj_file = asset_root + obj_name + '/' + obj_name + '.obj'
            max_convex_hulls = 64
            density = self.object_density
        
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01

            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000


            self._target_asset.append(self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))
    

            tree = ET.parse(os.path.join(asset_root, asset_file))
            root = tree.getroot()
            mesh = root.find(".//mesh")
            scale_str = mesh.attrib.get("scale", "1 1 1")
            scale = [float(s) for s in scale_str.strip().split()][0]

            mesh_obj = trimesh.load(obj_file, force='mesh')
            obj_verts = mesh_obj.vertices
            obj_verts = obj_verts * scale
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            object_points = to_torch(object_points - center)
            

            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))
        
        self.object_points = torch.stack(self.object_points, dim=0)
        return


    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):

        hoi_datas = []
        hoi_refs = []
    
        max_episode_length = []
        
        for idx, (key, value) in enumerate(self.motion_data.items()):
            loaded_dict = {}
            hoi_data = copy.deepcopy(value)
            root_pos = torch.from_numpy(hoi_data['root_trans_offset'][startk:]).to('cuda').float()
            T = root_pos.shape[0]
            root_rot = torch.from_numpy(hoi_data['root_rot'][startk:]).to('cuda').float()
            dof_pos = torch.from_numpy(hoi_data['dof'][startk:]).view(T, -1).to('cuda').float()
            body_pos = torch.from_numpy(hoi_data['body_pos'][startk:]).view(T, -1).to('cuda').float()
            obj_pos = torch.from_numpy(hoi_data['obj_trans'][startk:]).to('cuda').float()
            obj_rot = torch.from_numpy(hoi_data['obj_rot'][startk:]).to('cuda').float()
            contact_obj = torch.from_numpy(hoi_data['contact_obj'][startk:]).to('cuda').float()
            contact_human = torch.from_numpy(hoi_data['contact_human'][startk:, contact_id]).to('cuda').float()
            body_rot = torch.from_numpy(hoi_data['body_rot'][startk:]).view(T, -1).to('cuda').float()


            
            max_episode_length.append(dof_pos.shape[0])
            self.fps_data = 30.

            loaded_dict['root_pos'] = root_pos.clone()
            loaded_dict['root_pos_vel'] = (root_pos[1:,:].clone() - root_pos[:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to('cuda'),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = root_rot.clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(root_rot)
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to('cuda'),loaded_dict['root_rot_vel']),dim=0)

            loaded_dict['dof_pos'] = dof_pos.clone()

            loaded_dict['dof_vel'] = []

            loaded_dict['dof_vel'] = (dof_pos[1:,:].clone() - dof_pos[:-1,:].clone())*self.fps_data
            loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_vel'].shape[-1])).to('cuda'),loaded_dict['dof_vel']),dim=0)

            loaded_dict['body_pos'] = body_pos.clone()
            loaded_dict['body_pos_vel'] = (body_pos[1:,:].clone() - body_pos[:-1,:].clone())*self.fps_data
            loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_pos_vel'].shape[-1])).to('cuda'),loaded_dict['body_pos_vel']),dim=0)

            loaded_dict['obj_pos'] = obj_pos.clone()

            loaded_dict['obj_pos_vel'] = (obj_pos[1:,:].clone() - obj_pos[:-1,:].clone())*self.fps_data
            if self.init_vel:
                loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
            else:
                loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to('cuda'),loaded_dict['obj_pos_vel']),dim=0)


            loaded_dict['obj_rot'] = obj_rot.clone()
            obj_rot_exp_map = torch_utils.quat_to_exp_map(obj_rot)
            loaded_dict['obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to('cuda'),loaded_dict['obj_rot_vel']),dim=0)


            obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points[self.object_id[idx]].shape[0], 1).view(-1, 4).float()
            object_points_extend = self.object_points[self.object_id[idx]].unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points[self.object_id[idx]].shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)

            ref_ig = compute_sdf(loaded_dict['body_pos'].view(max_episode_length[-1],44,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, loaded_dict['body_pos'].shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
            loaded_dict['ig'] = ref_ig
            loaded_dict['contact_obj'] = torch.round(contact_obj.clone())
            loaded_dict['contact_human'] = torch.round(contact_human.clone())
            loaded_dict['body_rot'] = body_rot.clone()

            human_rot_exp_map = torch_utils.quat_to_exp_map(body_rot.view(-1, 4)).view(-1, 44*3)
            loaded_dict['body_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['body_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_rot_vel'].shape[-1])).to('cuda'),loaded_dict['body_rot_vel']),dim=0)

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(), 
                                                    loaded_dict['root_rot'].clone(), 
                                                    loaded_dict['dof_pos'].clone(), 
                                                    loaded_dict['dof_vel'].clone(),
                                                    loaded_dict['body_pos'].clone(),
                                                    loaded_dict['body_rot'].clone(),
                                                    loaded_dict['body_pos_vel'].clone(),
                                                    loaded_dict['body_rot_vel'].clone(),
                                                    loaded_dict['obj_pos'].clone(),
                                                    loaded_dict['obj_rot'].clone(),
                                                    loaded_dict['obj_pos_vel'].clone(), 
                                                    loaded_dict['obj_rot_vel'].clone(),
                                                    loaded_dict['ig'].clone(),
                                                    loaded_dict['contact_human'].clone(),
                                                    loaded_dict['contact_obj'].clone(),
                                                    ),dim=-1)
            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(initk)]+[loaded_dict['hoi_data']], dim=0)
            hoi_datas.append(loaded_dict['hoi_data'])

            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(), 
                                loaded_dict['root_rot'].clone(), 
                                loaded_dict['root_pos_vel'].clone(),
                                loaded_dict['root_rot_vel'].clone(), 
                                loaded_dict['dof_pos'].clone(), 
                                loaded_dict['dof_vel'].clone(), 
                                loaded_dict['obj_pos'].clone(),
                                loaded_dict['obj_rot'].clone(),
                                loaded_dict['obj_pos_vel'].clone(),
                                loaded_dict['obj_rot_vel'].clone(),
                                ),dim=-1)
            hoi_refs.append(hoi_ref)
        max_length = max(max_episode_length) + initk
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(max_episode_length, dtype=torch.long) + initk
        hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        hoi_data = torch.stack(hoi_data, dim=0)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, topk, 1, 1)

        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])).to(self.hoi_refs.device)
        self.ref_reward[:, 0, :] = 1.0

        self.ref_index = torch.zeros((self.num_envs, )).long().to(self.hoi_refs.device)
        if not hasattr(self, 'data_component_order'):
            self.create_component_stat(loaded_dict)
        return hoi_data

    def _compute_reset(self):
        super()._compute_reset()

    def _compute_torques(self, action_target):
        torques = self.p_gains*(action_target - self._dof_pos) - self.d_gains*self._dof_vel
        return torques

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            self.torques = self._compute_torques(pd_tar)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)       

        return

    def post_physics_step(self):
        self.progress_buf += 1
                
        self._refresh_sim_tensors()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._update_hist_hoi_obs()
        self._compute_hoi_observations()
        self._compute_observations(env_ids)
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self._dof_vel[:]
        self.last_root_vel[:] = self._humanoid_root_states[:, 7:13]
        self.last_root_pos[:] = self._humanoid_root_states[:, 0:3]

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self._humanoid_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self._rigid_body_ang_vel[:, 11, :])
        self.projected_gravity[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self.gravity_vec)


        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return



    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return


    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids, key_body_ids_gt, contact_body_ids_gt):
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        len_keypos = len(key_body_ids)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])
        
        # _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids_gt, :]
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids_gt, :]
        _body_pos = body_pos[:, key_body_ids, :]

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos[:, key_body_ids, :] - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = body_rot[:, key_body_ids, :].reshape(body_rot.shape[0] * len_keypos, body_rot.shape[2])
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], len_keypos * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs).view(ref_obs.shape[0], -1, 4)
        # ref_body_rot_no_hand = ref_body_rot[:, key_body_ids_gt, :]
        ref_body_rot_no_hand = ref_body_rot[:, key_body_ids_gt, :]
        body_rot_no_hand = body_rot[:, key_body_ids]

        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        # ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids_gt, :]
        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids_gt, :]
        _body_vel = body_vel[:, key_body_ids, :]
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        # ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 52, 3)[:, key_body_ids_gt]
        ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 44, 3)[:, key_body_ids_gt]
        body_ang_vel_no_hand = body_ang_vel[:, key_body_ids]
        diff_global_ang_vel = ref_body_ang_vel_no_hand - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, len_keypos * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel[:, key_body_ids, :].reshape(body_vel.shape[0] * len_keypos, body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], len_keypos * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel[:, key_body_ids, :].reshape(body_ang_vel.shape[0] * len_keypos, body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], len_keypos * body_ang_vel.shape[2])

        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        # ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids_gt]
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids_gt]
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

        obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
        return obs

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return   

    def _reset_target(self, env_ids):
        super()._reset_target(env_ids)
        self._target_states[env_ids, 0:2] = self._target_states[env_ids, 0:2] * self.scaling
        return


    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

        return    

    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos * self.scaling
        self._humanoid_root_states[env_ids, 2:3] = self.init_root_height
        self._humanoid_root_states[env_ids, 3:5] = 0
        self._humanoid_root_states[env_ids, 5:6] = 1
        self._humanoid_root_states[env_ids, 6:7] = -1
        self._humanoid_root_states[env_ids, 7:10] = 0
        self._humanoid_root_states[env_ids, 10:13] = 0
        
        self._dof_pos[env_ids] = self.init_dof
        self._dof_vel[env_ids] = 0
        return
    
    def _compute_ig_obs(self, env_ids, ref_obs):
        ig = self.ig[env_ids]
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        # ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
        ig = ig_all[:, self._key_body_ids_gt, :].view(env_ids.shape[0], -1)
        ig_all = ig_all.view(env_ids.shape[0], -1)    
        ref_ig = self.extract_data_component('ig', obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids_gt, :]
        # ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(env_ids.shape[0], -1)
        return ig_all, ig, ref_ig
    
    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            self._curr_ref_obs[:] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(self.hoi_data, None, 1), self._compute_observations_iter(self.hoi_data, None, 16), (self.progress_buf >= 5).float().unsqueeze(1)), dim=-1)

        else:
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data, env_ids, 1), self._compute_observations_iter(self.hoi_data, env_ids, 16), (self.progress_buf[env_ids] >= 5).float().unsqueeze(1)), dim=-1)
            
        return
    
    def _compute_hoi_observations(self, env_ids=None):
        self._curr_obs[:] = self.build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                        self._rigid_body_rot[:, 0, :],
                                                        self._rigid_body_vel[:, 0, :],
                                                        self._rigid_body_ang_vel[:, 0, :],
                                                        self._dof_pos, self._dof_vel, self._rigid_body_pos,
                                                        self._local_root_obs, self._root_height_obs, 
                                                        self._dof_obs_size, self._target_states,
                                                        self._tar_contact_forces,
                                                        self._contact_forces,
                                                        self.object_points[self.object_id[self.data_id]],
                                                        self._rigid_body_rot,
                                                        self._rigid_body_vel,
                                                        self._rigid_body_ang_vel,
                                                        self._key_body_ids,
                                                        self._contact_body_ids,
                                                        self._key_body_ids_gt,
                                                        self._contact_body_ids_gt,
                                                        )
        return

    
    def build_hoi_observations(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, 
                               local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel, _key_body_ids, _contact_body_ids, _key_body_ids_gt=None, _contact_body_ids_gt=None):

        contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
        target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

        tar_pos = target_states[:, 0:3]
        tar_rot = target_states[:, 3:7]
        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        ig = compute_sdf(body_pos, obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1, 3)    
        self.ig = ig
        dof_pos_new = torch.zeros((root_pos.shape[0], 43), device=root_pos.device)
        dof_vel_new = torch.zeros((root_pos.shape[0], 43), device=root_pos.device)    
        dof_pos_new[:, :dof_pos.shape[1]] = dof_pos        
        dof_vel_new[:, :dof_vel.shape[1]] = dof_vel  
       
        contact_new = torch.zeros((root_pos.shape[0], 44), device=root_pos.device) 
        contact_new[:, _contact_body_ids_gt] = contact[:, _contact_body_ids]
        body_pos_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device) 
        body_vel_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device)
        ig_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device)
        body_pos_new[:, _key_body_ids_gt] = body_pos[:, _key_body_ids]
        body_vel_new[:, _key_body_ids_gt] = body_vel[:, _key_body_ids]
        ig_new[:, _key_body_ids_gt] = ig[:, _key_body_ids]
        body_rot_new = torch.zeros((root_pos.shape[0], 44, 4), device=root_pos.device) 
        body_rot_vel_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device) 
        body_rot_new[:, _key_body_ids_gt] = body_rot[:, _key_body_ids]
        body_rot_vel_new[:, _key_body_ids_gt] = body_rot_vel[:, _key_body_ids]
        
        # contact_new = torch.zeros((root_pos.shape[0], 44), device=root_pos.device) 
        # contact_new = contact[:, _contact_body_ids]
        # body_pos_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device) 
        # body_vel_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device)
        # ig_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device)
        # body_pos_new = body_pos[:, _key_body_ids]
        # body_vel_new = body_vel[:, _key_body_ids]
        # ig_new = ig[:, _key_body_ids]
        # body_rot_new = torch.zeros((root_pos.shape[0], 44, 4), device=root_pos.device) 
        # body_rot_vel_new = torch.zeros((root_pos.shape[0], 44, 3), device=root_pos.device) 
        # body_rot_new = body_rot[:, _key_body_ids]
        # body_rot_vel_new = body_rot_vel[:, _key_body_ids]

        obs = torch.cat((root_pos, root_rot, dof_pos_new, dof_vel_new, 
                         body_pos_new.view(body_pos_new.shape[0],-1), 
                         body_rot_new.view(body_rot_new.shape[0],-1), 
                         body_vel_new.view(body_vel_new.shape[0],-1), 
                         body_rot_vel_new.view(body_rot_vel_new.shape[0],-1),
                         target_states, ig_new.view(ig_new.shape[0],-1), contact_new, target_contact), dim=-1)        
        return obs
    
    def play_dataset_step(self, time):
        return
    
    def _compute_reward(self, actions):
        # super()._compute_reward(actions)
        # self.rew_buf = 0
        rb, human_reset, key_pos, ref_key_pos = self.compute_humanoid_reward(self.reward_scales)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward(self.reward_scales)
        rig, ig_reset = self.compute_ig_reward(self.reward_scales, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset = self.compute_cg_reward(self.reward_scales)
        self.rew_buf[:] = rb + ro + rig + rcg
        # self.rew_buf[:] = rb * ro * rig

        for i in range(len(self.penalty_functions)):
            name = self.penalty_names[i]
            rew = self.penalty_functions[i]() * self.penalty_scales[name]  
            self.rew_buf += rew
            self.episode_sums[name] += rew


        kinematic_reset = torch.logical_or(human_reset, object_reset)
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        index = torch.arange(self._curr_reward.shape[0])
        # # print(self._humanoid_root_states.dtype)
        self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
        self._sum_reward[index] += self.rew_buf
        self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
            self._humanoid_root_states,
            self._dof_pos,
            self._dof_vel,
            self._target_states,
        ], dim=1)
        return
    
    def compute_humanoid_reward(self, reward_scales):
        # body pos reward
        len_keypos = len(self._key_body_ids_gt)
        key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._key_body_ids_gt]
        # key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)
        
        ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._key_body_ids_gt]
        # ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1)
        weight_h = (-5 * ref_ig_norm).exp()
        weight_hp = weight_h.clone().detach()  
        ancle_toe_ids = [i for i in range(len_keypos) if 'ankle' in self.key_bodies[i] or 'toe' in self.key_bodies[i]]
        # ancle_toe_ids = [self._key_body_ids[i] for i in range(len(self.key_bodies)) if 'ankle' in self.key_bodies[i]]

        weight_hp[:, ancle_toe_ids] = 1

        # ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, self._key_body_ids_gt],dim=-1)
        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1), dim=-1) 
        # ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp,dim=-1)
        rp = torch.exp(-ep/reward_scales['humanoid_pos_sigma'])
        self.extras['humanoid_pos'] = rp.detach().cpu().numpy()
        rp = reward_scales['humanoid_pos'] * rp
        self.extras['reward_humanoid_pos'] = rp.detach().cpu().numpy()

        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)[:, self._key_body_ids_gt]
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)[:, self._key_body_ids_gt]
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        # diff = diff_angle.view(-1, 52)
        diff = diff_angle.view(-1, len_keypos)
        weight_hr = 1 - weight_h
        
        # er = torch.mean(diff[:, :] * weight_hr[:, self._key_body_ids_gt], dim=-1)
        er = torch.mean(diff[:, :], dim=-1)  
        rr = torch.exp(-er*reward_scales['humanoid_rot_sigma'])
        self.extras['humanoid_rot'] = rr.detach().cpu().numpy()
        rr = reward_scales['humanoid_rot'] * rr
        self.extras['reward_humanoid_rot'] = rr.detach().cpu().numpy()
        
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)[:, self._key_body_ids_gt]
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)[:, self._key_body_ids_gt]
        # body pos vel reward
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
        # epv = torch.mean(pos_vel ,dim=-1) # torch.zeros_like(ep)
        rpv = torch.exp(-epv*reward_scales['humanoid_pos_vel_sigma'])
        self.extras['humanoid_pos_vel'] = rpv.detach().cpu().numpy()
        rpv = reward_scales['humanoid_pos_vel'] * rpv
        self.extras['reward_humanoid_pos_vel'] = rpv.detach().cpu().numpy()

        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        # body rot vel reward
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
        rrv = torch.exp(-erv*reward_scales['humanoid_rot_vel_sigma'])
        self.extras['humanoid_rot_vel'] = rrv.detach().cpu().numpy()
        rrv = reward_scales['humanoid_rot_vel'] * rrv   
        self.extras['reward_humanoid_rot_vel'] = rrv.detach().cpu().numpy()

        # energy penalty
        # hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        # local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel)*self.fps_data
        # # dof_diffacc = (local_vel.view(-1, 51*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        # dof_diffacc = (local_vel.view(-1, 43)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        # energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()
        dof_pos = self.extract_data_component('dof_pos', obs=self._curr_obs)
        ref_dof_pos = self.extract_data_component('dof_pos', obs=self._curr_ref_obs)
        edp = torch.mean((ref_dof_pos - dof_pos)**2,dim=-1)
        rdp = torch.exp(-edp * reward_scales['humanoid_dof_pos_sigma'])
        self.extras['humanoid_dof_pos'] = rdp.detach().cpu().numpy()
        rdp = reward_scales['humanoid_dof_pos'] * rdp
        self.extras['reward_humanoid_dof_pos'] = rdp.detach().cpu().numpy()

        dof_vel = self.extract_data_component('dof_vel', obs=self._curr_obs)
        ref_dof_vel = self.extract_data_component('dof_vel', obs=self._curr_ref_obs)
        edv = torch.mean((ref_dof_vel - dof_vel)**2,dim=-1)
        rdv = torch.exp(-edv * reward_scales['humanoid_dof_vel_sigma'])
        self.extras['humanoid_dof_vel'] = rdv.detach().cpu().numpy()
        rdv = reward_scales['humanoid_dof_vel'] * rdv
        self.extras['reward_humanoid_dof_vel'] = rdv.detach().cpu().numpy()

        # rb = rp*rr*rpv*rrv*energy
        rb = rp+rr+rpv+rrv+rdp+rdv
        human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        
        return rb, human_reset, key_pos, ref_key_pos

    def compute_obj_reward(self, reward_scales):
        # object pos reward
        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        root_rot = self.extract_data_component('root_rot', obs=self._curr_obs)

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        
        obj_pos = self.extract_data_component('obj_pos', obs=self._curr_obs)
        obj_rot = self.extract_data_component('obj_rot', obs=self._curr_obs)
        local_obj_pos = obj_pos - root_pos
        local_obj_pos[..., -1] = obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

        local_obj_rot = quat_mul(heading_rot, obj_rot)

        object_points = self.object_points[self.object_id[self.data_id]]
        obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_root_rot = self.extract_data_component('root_rot', obs=self._curr_ref_obs)

        ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)

        ref_obj_pos = self.extract_data_component('obj_pos', obs=self._curr_ref_obs)
        ref_obj_rot = self.extract_data_component('obj_rot', obs=self._curr_ref_obs)

        ref_local_obj_pos = ref_obj_pos - ref_root_pos
        ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
        ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

        ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)

        ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

        eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2),dim=-1) # * (1 - weight_h.max(dim=-1)[0])
        rop = torch.exp(-eop*reward_scales['object_pos_sigma'])
        self.extras['object_pos'] = rop.detach().cpu().numpy()
        rop = reward_scales['object_pos'] * rop
        self.extras['reward_object_pos'] = rop.detach().cpu().numpy()

        # object rot reward
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 1)
        
        eor = torch.mean(diff,dim=-1)
        ror = torch.exp(-eor*reward_scales['object_rot_sigma'])
        self.extras['object_rot'] = ror.detach().cpu().numpy()
        ror = reward_scales['object_rot'] * ror
        self.extras['reward_object_rot'] = ror.detach().cpu().numpy()



        obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_obs)
        ref_obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_ref_obs)
        # object pos vel reward
        eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
        ropv = torch.exp(-eopv*reward_scales['object_pos_vel_sigma'])
        self.extras['object_pos_vel'] = ropv.detach().cpu().numpy()
        ropv = reward_scales['object_pos_vel'] * ropv
        self.extras['reward_object_pos_vel'] = ropv.detach().cpu().numpy()
        
        obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_obs)
        ref_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_ref_obs)
        # object rot vel reward
        eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
        rorv = torch.exp(-eorv*reward_scales['object_rot_vel_sigma'])
        self.extras['object_rot_vel'] = rorv.detach().cpu().numpy()
        rorv = reward_scales['object_rot_vel'] * rorv
        self.extras['reward_object_rot_vel'] = rorv.detach().cpu().numpy()
        
        hist_obj_vel = self.extract_data_component('obj_pos_vel', obs=self._hist_obs)
        obj_diffacc = (self.extract_data_component('obj_pos_vel', obs=self._curr_obs) - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('obj_rot_vel', obs=self._curr_obs) - hist_obj_rot_vel)*self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        obj_energy = (obj_diffacc.pow(2).mean(dim=-1).mul(-reward_scales['object_energy']).exp()) * (obj_rot_diffacc.pow(2).mean(dim=-1).mul(-reward_scales['object_energy']).exp())
        self.extras['object_energy'] = obj_energy.detach().cpu().numpy()
        # ro = rop*ror*ropv*rorv*obj_energy
        ro = rop + ror + ropv + rorv + obj_energy
        object_reset = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1) > 0.5
        return ro, object_reset, obj_points, ref_obj_points

    def compute_ig_reward(self, reward_scales, key_pos, ref_key_pos, obj_points, ref_obj_points):
        len_keypos = len(self._key_body_ids)
        ig = key_pos.view(-1,len_keypos,3).unsqueeze(2) - obj_points.unsqueeze(1)
        ref_ig = ref_key_pos.view(-1,len_keypos,3).unsqueeze(2) - ref_obj_points.unsqueeze(1)
        ### interaction graph reward ###
        weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.01))
        weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.01))
        weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)  

        rig = torch.exp(-reward_scales['interaction_graph'] * (eig.sum(dim=-1).sum(dim=-1) * 0.5))
        self.extras['interaction_graph'] = rig.detach().cpu().numpy()
        reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
        return rig, reset_ig
    
    def compute_cg_reward(self, reward_scales):    
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        
        ref_left_contact_hand = ref_human_contact[:, self.left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, self.left_contact_hand_ids].clone()
        left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left*reward_scales['contact_hand'])) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)
        
        ref_right_contact_hand = ref_human_contact[:, self.right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, self.right_contact_hand_ids].clone()
        right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        contact_reset = torch.cat([ 
                                torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(-1), 
                                torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1),
                                ], dim=-1)
        
        ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right*reward_scales['contact_hand'])) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)

        rcg_hand = rcg_left * rcg_right
        # rcg_hand = rcg_hand * reward_scales['contact_hand']
        self.extras['contact_hand'] = rcg_hand.detach().cpu().numpy()
        
        other_ids = [i for i in range(len(self.contact_bodies)) if i not in self.left_contact_hand_ids and i not in self.right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other*reward_scales['contact_other'])
        self.extras['contact_other'] = rcg_other.detach().cpu().numpy()
        
        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all*reward_scales['contact_all'])
        self.extras['contact_all'] = rcg_all.detach().cpu().numpy()

        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-reward_scales['contact_energy']).exp()
        self.extras['contact_energy'] = contact_energy.detach().cpu().numpy()

        # rcg = rcg_hand*rcg_other*rcg_all*contact_energy
        rcg = rcg_hand + rcg_other + rcg_all + contact_energy
        return rcg, contact_reset
    
    #------------ penalty functions----------------
    def _penalty_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.soft_torque_limit).clip(min=0.), dim=1)

    def _penalty_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self._dof_pos - self.dof_limits_lower[:]).clip(max=0.) # lower limit
        out_of_limits += (self._dof_pos - self.dof_limits_upper[:]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _penalty_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self._dof_vel) - self.dof_vel_limits*self.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _penalty_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self._dof_vel) / self.dt), dim=1)

    def _penalty_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self._dof_vel), dim=1)
    
    # TODO: make sure whether is 11
    def _penalty_lower_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, :11] - self.actions[:, :11]), dim=1)
    
    def _penalty_upper_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, 11:] - self.actions[:, 11:]), dim=1)
        
    def _penalty_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _penalty_feet_air_time_teleop(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        ref_body_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        
        contact = self._contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(ref_body_vel[:, :2], dim=1) > 0.1 #no reward for low ref motion velocity (root xy velocity)
        self.feet_air_time *= ~contact_filt
        return rew_airTime    
    
    def _penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self._contact_forces[:, self.feet_indices, :], dim=-1) -  self.max_contact_force).clip(min=0.), dim=1)

    def _penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self._contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self._contact_forces[:, self.feet_indices, 2]), dim=1)

    def _penalty_slippage(self):
        foot_vel = self._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self._contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)

    def _penalty_feet_ori(self):
        left_quat = self._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _penalty_in_the_air(self):
        contact = self._contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward

    def _penalty_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _penalty_base_height(self):
        # Penalize base height away from target
        base_height = self._humanoid_root_states[:, 2]
        dif = torch.abs(base_height - self.base_height_target)
        return torch.clip(dif - 0.15, min=0.)
      
    def _penalty_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # TODO
    def _penalty_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self._contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _penalty_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _penalty_close_feet(self):
        # returns 1 if two feet are too close
        return (self.feet_distance < 0.24).squeeze(-1) * 1.0
               
                
    def _penalty_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self._dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    
    def _penalty_no_fly(self):
        contacts = self._contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact


# ---------------------------------------------------------------------------
# OMOMO_new (InterAct/OMOMO_new) training entrypoint
# ---------------------------------------------------------------------------
#
# NOTE:
# - The legacy class above (`InterMimicG1`) is a single-file / joblib(.pkl) loader
#   that was used for "single-trajectory" style experiments.
# - For training on the official multi-trajectory dataset folder `InterAct/OMOMO_new`
#   (each sequence is a `.pt` tensor), we reuse the G1 task that already supports
#   this dataset format.
#
# Use via:
#   python dreammimic/run.py --task InterMimicG1Hand --cfg_env ... --sub sub8
#
class InterMimicG1Hand(_InterMimicG1GMRv4):
    """G1-with-hand task that trains directly on `InterAct/OMOMO_new`."""
    pass
    