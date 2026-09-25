import os

from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed

import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from env.tasks.intermimic_g1_vis_student_obs_wm import InterMimicG1_Vis_StudentObs_WM
from env.tasks.vec_task_student_obs_wrappers import VecTaskDAggerStudentObsWrapper
from learning import intermimic_agent_distill_student_wm
from learning import intermimic_models
from learning import intermimic_network_builder_student_wm
from learning import intermimic_players_student_wm


args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task = InterMimicG1_Vis_StudentObs_WM(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
    )
    env = VecTaskDAggerStudentObsWrapper(
        task,
        args.rl_device,
        cfg_train.get("clip_observations", np.inf),
        cfg_train.get("clip_actions", 1.0),
    )
    print("num_envs: {:d}".format(env.num_envs))
    print("num_actions: {:d}".format(env.num_actions))
    print("num_obs: {:d}".format(env.num_obs))
    print("num_states: {:d}".format(env.num_states))
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict) and self.use_successes and "successes" in infos:
            self.consecutive_successes.update(infos["successes"].clone()[done_indices].to(self.algo.ppo_device))

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar("successes/consecutive_successes/mean", mean_successes, frame)
            self.writer.add_scalar("successes/consecutive_successes/iter", mean_successes, epoch_num)
            self.writer.add_scalar("successes/consecutive_successes/time", mean_successes, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.use_global_obs = self.env.num_states > 0
        self.full_state = {}
        self.full_state["obs"], _ = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    def step(self, action):
        next_obs, reward, is_done, info, expert = self.env.step(action)
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info, expert
        return self.full_state["obs"], reward, is_done, info, expert

    def reset(self, env_ids=None):
        self.full_state["obs"], expert = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, expert
        return self.full_state["obs"], expert

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {
            "action_space": self.env.action_space,
            "observation_space": self.env.observation_space,
            "amp_observation_space": self.env.amp_observation_space,
        }
        if self.use_global_obs:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])
        return info


vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register("rlgpu", {"env_creator": lambda **kwargs: create_rlgpu_env(**kwargs), "vecenv_type": "RLGPU"})


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder(
        "intermimic_student_wm",
        lambda **kwargs: intermimic_agent_distill_student_wm.InterMimicAgentDistillStudentWM(**kwargs),
    )
    runner.player_factory.register_builder(
        "intermimic_student_wm",
        lambda **kwargs: intermimic_players_student_wm.InterMimicPlayerContinuousStudentWM(**kwargs),
    )
    runner.model_builder.model_factory.register_builder(
        "intermimic_student_wm",
        lambda network, **kwargs: intermimic_models.ModelInterMimicContinuous(network),
    )
    runner.model_builder.network_factory.register_builder(
        "intermimic_student_wm",
        lambda **kwargs: intermimic_network_builder_student_wm.InterMimicBuilderStudentWM(**kwargs),
    )
    return runner


def main():
    global args, cfg, cfg_train
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    cfg_train["params"]["seed"] = set_seed(
        cfg_train["params"].get("seed", -1),
        cfg_train["params"].get("torch_deterministic", False),
    )
    if args.horizon_length != -1:
        cfg_train["params"]["config"]["horizon_length"] = args.horizon_length
    if args.minibatch_size != -1:
        cfg_train["params"]["config"]["minibatch_size"] = args.minibatch_size
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    if args.save_images:
        cfg["env"]["saveImages"] = True
    cfg_train["params"]["config"]["train_dir"] = args.output_path
    os.makedirs(args.output_path, exist_ok=True)

    runner = build_alg_runner(RLGPUAlgoObserver())
    runner.load(cfg_train)
    runner.reset()
    runner.run(vars(args))


if __name__ == "__main__":
    main()
