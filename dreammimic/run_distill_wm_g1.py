# Copyright (c) 2018-2022, NVIDIA Corporation
# Run script for G1 World Model integrated distillation

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task_distill

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np
import copy
import torch

from learning import intermimic_agent_distill_wm_g1
from learning import intermimic_players_distill
from learning import intermimic_models
from learning import intermimic_network_builder_wm_g1

args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    print("[G1 WM] Step 1/5: Parsing simulation parameters...")
    sim_params = parse_sim_params(args, cfg, cfg_train)
    
    print("[G1 WM] Step 2/5: Creating task and environment...")
    task, env = parse_task_distill(args, cfg, cfg_train, sim_params)

    print('[G1 WM] Step 3/5: Environment created successfully!')
    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    print('[G1 WM] Step 4/5: Environment info printed, proceeding to RLGPUEnv initialization...')
    
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        print(f"[G1 WM] RLGPUEnv: Creating environment '{config_name}' with {num_actors} actors...")
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        print("[G1 WM] RLGPUEnv: Calling reset() to initialize environment...")
        self.full_state = {}
        self.full_state["obs"], expert = self.reset()
        print("[G1 WM] RLGPUEnv: Reset completed successfully!")
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        print("[G1 WM] RLGPUEnv: Initialization complete!")
        return

    def step(self, action):
        next_obs, reward, is_done, info, expert = self.env.step(action)

        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info, expert
        else:
            return self.full_state["obs"], reward, is_done, info, expert

    def reset(self, env_ids=None):
        if env_ids is None:
            print("[G1 WM] RLGPUEnv: Resetting all environments...")
        else:
            print(f"[G1 WM] RLGPUEnv: Resetting {len(env_ids)} environments...")
        self.full_state["obs"], expert = self.env.reset(env_ids)
        print("[G1 WM] RLGPUEnv: Environment reset completed!")
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, expert
        else:
            return self.full_state["obs"], expert

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    print("[G1 WM] Step 5/5: Building algorithm runner...")
    runner = Runner(algo_observer)

    print("[G1 WM] Registering G1 World Model components...")
    # Register G1 World Model agent and network builder
    runner.algo_factory.register_builder('intermimic_wm_g1', lambda **kwargs : intermimic_agent_distill_wm_g1.InterMimicAgentDistillWMG1(**kwargs))
    runner.player_factory.register_builder('intermimic', lambda **kwargs : intermimic_players_distill.InterMimicPlayerContinuousDistill(**kwargs))
    runner.player_factory.register_builder('intermimic_wm_g1', lambda **kwargs : intermimic_players_distill.InterMimicPlayerContinuousDistill(**kwargs))
    runner.model_builder.model_factory.register_builder('intermimic_wm_g1', lambda network, **kwargs : intermimic_models.ModelInterMimicContinuous(network))  
    runner.model_builder.network_factory.register_builder('intermimic_wm_g1', lambda **kwargs : intermimic_network_builder_wm_g1.InterMimicBuilderWMG1(**kwargs))
    print("[G1 WM] Components registered successfully!")

    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
        
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    if args.play_dataset:
        cfg['env']['playdataset'] = True

    if args.projtype:
        cfg['env']['projtype'] = args.projtype

    if args.cg1 != -1.:
        cfg['env']['rewardWeights']['cg1'] = args.cg1

    if args.cg2 != -1.:
        cfg['env']['rewardWeights']['cg2'] = args.cg2

    if args.ig != -1.:
        cfg['env']['rewardWeights']['ig'] = args.ig

    if args.op != -1.:
        cfg['env']['rewardWeights']['op'] = args.op

    if args.save_images:
        cfg['env']['saveImages'] = True
    
    if args.init_vel:
        cfg['env']['initVel'] = True

    if args.frames_scale!= 0.:
        cfg['env']['dataFramesScale'] = args.frames_scale

    if args.ball_size!= 0.:
        cfg['env']['ballSize'] = args.ball_size
    
    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path
    
    vargs = vars(args)

    print("[G1 WM] Creating algorithm observer...")
    algo_observer = RLGPUAlgoObserver()

    print("[G1 WM] Building algorithm runner...")
    runner = build_alg_runner(algo_observer)
    
    print("[G1 WM] Loading configuration...")
    runner.load(cfg_train)
    
    print("[G1 WM] Resetting runner...")
    runner.reset()
    
    print("[G1 WM] Starting training...")
    runner.run(vargs)

    return

if __name__ == '__main__':
    main()
