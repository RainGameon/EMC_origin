import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.episode_buffer import Prioritized_ReplayBuffer
from components.transforms import OneHot
from utils.torch_utils import to_cuda
from modules.agents.LRN_KNN import LRU_KNN
from components.episodic_memory_buffer import Episodic_memory_buffer
from components.episodic_memory_buffer_vp import Episodic_memory_buffer_vp
import numpy as np
import copy as cp
import random

def run(_run, _config, _log):
    print('SUNMENGYAO________________________starting run')
    os.environ['SC2PATH'] = '/home/EMC/pymarl/3rdparty/StarCraftII'
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    os.environ['SET_DEVICE'] = '7'
    set_device = os.getenv('SET_DEVICE')
    print('SUNMENGYAO________________________')
    print('cuda', th.version.cuda)
    if args.use_cuda and set_device != '-1':
        if set_device is None:
            args.device = "cuda"
        else:
            args.device = f"cuda:{set_device}"
    else:
        args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env,
                                     args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
        tb_info_get = os.path.join("results", "tb_logs", args.env, args.env_args['map_name'], "{}").format(unique_token)
        _log.info("saving tb_logs to " + tb_info_get)

    # sacred is on by default
    logger.setup_sacred(_run)

    # ❗️Run and train！！！！！！！！
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

# 该函数用于按顺序运行一系列测试样例，以测试模式评估强化学习算法的性能。它还提供了保存回放数据和关闭环境的选项。
def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'
    path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    if os.path.exists(path_name):
        random_name = '../../buffer/' + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"] # 表示在强化学习环境中的智能体数量
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]  # 这个值表示环境中单位的特征维度或状态维度。
    args.obs_shape = env_info["obs_shape"]

    # scheme字典包含6个键值对  Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8}, # 表示终止状态的形状为一个标量
        "flag_win": {"vshape": (1,), "dtype": th.uint8},
        # "return_H":{"vshape": (1,),"dtype": th.long},   # 估计的最高return
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    if args.is_prioritized_buffer:
        buffer = Prioritized_ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                          args.prioritized_buffer_alpha,
                                          preprocess=preprocess,
                                          device="cpu" if args.buffer_cpu_only else args.device)
    else:
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              args.burn_in_period,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_save_buffer:
        save_buffer = ReplayBuffer(scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                   args.burn_in_period,
                                   preprocess=preprocess,
                                   device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_batch_rl:
        assert (args.is_save_buffer == False)
        x_env_name = env_name
        if args.is_from_start:
            x_env_name += '_from_start/'
        path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
        assert (os.path.exists(path_name) == True)
        buffer.load(path_name)

    # 创建episodic memory buffer
    if getattr(args, "use_emdqn", False):
        ec_buffer=Episodic_memory_buffer_vp(args,scheme)
        # memory_buffer = EpisodicMemoryTBP(args,self.buffer_size, state_dim = args.state_shape,
        #                                     obs_shape=scheme["obs"]["vshape"],
        #                                     action_shape=scheme["actions"]["vshape"],
        #                                     avail_actions_shape = scheme["avail_actions"]["vshape"],
        #                                     gamma=self.gamma,
        #                                     max_step=self.max_step)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    if args.runner != 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.learner=="fast_QLearner" or args.learner=="qplex_curiosity_vdn_learner" or args.learner=="qplex_curiosity_vdn_learner_COIN" or args.learner=="qplex_curiosity_vdn_learner_inf" or args.learner=="qplex_curiosity_vdn_learner_ps":
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, groups=groups)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.runner == 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, test_mac=learner.extrinsic_mac)

    if hasattr(args, "save_buffer") and args.save_buffer:
        learner.buffer = buffer

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))  # 存放检查点的时间步？
        # 根据load_step的值来决定【加载】哪一个时间步的【模型】
        if args.load_step == 0:
            #  选择timesteps列表中的最大值（即最新的检查点）
            timestep_to_load = max(timesteps)
        else:
            # 选择与load_step最接近的时间步
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path) # 加载模型
        runner.t_env = timestep_to_load # 将runner.t_env设置为已加载模型的时间步，以便后续使用

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner) # 进行评估或保存回放
            return

    ''' ———————————————————————————————————————————————start training—————————————————————————————————————— '''
    episode = 0
    last_test_T = -args.test_interval - 1  # test_interval: 2000
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    if args.env == 'matrix_game_1' or args.env == 'matrix_game_2' or args.env == 'matrix_game_3' \
            or args.env == 'mmdp_game_1':
        last_demo_T = -args.demo_interval - 1

    while runner.t_env <= args.t_max: # runner.t_env为已加载模型的时间步
        # （1） 更新replay buffer
        if not args.is_batch_rl:
            # ① 交互获得一个完整的episode(不是测试模式)   Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            # ② 更新episodic memory buffer
            if getattr(args, "use_emdqn", False):
                ec_buffer.update_ec(episode_batch)
            # ③ 更新普通buffer
            buffer.insert_episode_batch(episode_batch)
            # ④ 如果设置了保存缓冲区的标志，则将episode_batch插入到另一个save_buffer缓冲区中
            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    save_buffer.is_from_start = False
                    save_one_buffer(args, save_buffer, env_name, from_start=True)
                    break
                if save_buffer.buffer_index % args.save_buffer_interval == 0:
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)
        # （2） 【进行模型训练】，【从普通replay_buffer中采样】并根据训练结果（如TD Error）动态调整样本的优先级
        for _ in range(args.num_circle): # args.num_circle 是预先设定的训练循环次数
            # 检查buffer是否有足够样本构成一个batch
            if buffer.can_sample(args.batch_size):
                # ① 采样
                if args.is_prioritized_buffer:  # 以优先级方式采样
                    sample_indices, episode_sample = buffer.sample(args.batch_size)
                else:                           # 以普通方式采样
                    episode_sample = buffer.sample(args.batch_size)
                    # episode_sample = self.memory_buffer.sample(args.batch_size, mix=False)
                # 如果是批量模式，根据采样的样本中的时间步来【更新环境时间（runner.t_env）】
                if args.is_batch_rl:
                    runner.t_env += int(th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size
                # 记录动作和奖励
                if args.env == 'matrix_game_2' or args.env == 'matrix_game_3':
                    for t in range(args.batch_size):
                        i = t % (args.n_actions ** 2) // args.n_actions
                        j = t % args.n_actions
                        new_actions = th.Tensor([i, j]).unsqueeze(0).repeat(args.episode_limit + 1, 1)
                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]
                        # 更新动作数据
                        episode_sample['actions'][t, :, :, 0] = new_actions
                        # 根据动作索引i和j计算奖励
                        if i == 0 and j == 0:
                            rew = th.Tensor([8, ])   # 如果i和j都为0，奖励为8
                        elif i == 0 or j == 0:
                            rew = th.Tensor([-12, ]) # 如果i或j为0，奖励为-12
                        else:
                            rew = th.Tensor([0, ])
                        # 如果环境是matrix_game_3，还需考虑特殊奖励规则
                        if args.env == 'matrix_game_3':
                            if i == 1 and j == 1 or i == 2 and j == 2:
                                rew = th.Tensor([6, ])
                        # 更新奖励数据
                        episode_sample['reward'][t, 0, 0] = rew
                    # 将动作数据转换为one-hot编码
                    new_actions_onehot = to_cuda(th.zeros(
                        episode_sample['actions'].squeeze(3).shape + (args.n_actions,)), self.args.device)
                    new_actions_onehot = new_actions_onehot.scatter_(3, to_cuda(episode_sample['actions'], self.args.device), 1)
                    episode_sample['actions_onehot'][:] = new_actions_onehot

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                else:
                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()  # 得到所有样本中的最大时间步长，用于后续截断操作，确保所有样本的时间长度一致。
                    episode_sample = episode_sample[:, :max_ep_t] # 进行截断

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                # ③ 训练learner
                if args.is_prioritized_buffer:  # false
                    # 如果使用EMDQN，则使用episodic memory buffer进行训练
                    if getattr(args, "use_emdqn", False):
                        td_error = learner.train(episode_sample, runner.t_env, episode,ec_buffer=ec_buffer)
                    else:
                        td_error = learner.train(episode_sample, runner.t_env, episode)
                        buffer.update_priority(sample_indices, td_error)  # # 基于TDerror更新样本优先级
                else:
                    if getattr(args, "use_emdqn", False):
                        td_error = learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                    else:
                        learner.train(episode_sample, runner.t_env, episode,replay_buffer=buffer)
                # 训练学习器learner 【如果环境是'mmdp_game_1'且学习器设置为"q_learner_exp"】
                if args.env == 'mmdp_game_1' and args.learner == "q_learner_exp":
                    for i in range(int(learner.target_gap) - 1):
                        # 从缓冲区中采样一批数据
                        episode_sample = buffer.sample(args.batch_size)

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()      # 获取最大填充的时间步长
                        episode_sample = episode_sample[:, :max_ep_t] # 根据最大时间步长截断样本

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)
                        # 使用采样的数据训练学习器
                        learner.train(episode_sample, runner.t_env, episode)

        # （3） 定期测试 Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)  # 计算需要执行的【测试运行次数】
        # 判断是否已经到了执行下一组测试运行的时间
        # runner.t_env 表示当前的环境时间（或步数），last_test_T 是上次测试运行的时间，args.test_interval 是两次测试之间的间隔时间
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0 :
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            logger.log_stat("num_circle", args.num_circle, runner.t_env)

            last_test_T = runner.t_env # 更新上次测试的时间为当前环境时间。
            for _ in range(n_test_runs):
                episode_sample = runner.run(test_mode=True) # 收集一批测试样本
                if args.mac == "offline_mac":
                    max_ep_t = episode_sample.max_t_filled() # 【截断批次样本】，仅保留至最大填充时间步的数据
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    # 进行测试，记录
                    learner.train(episode_sample, runner.t_env, episode, show_v=True)

        if args.env == 'mmdp_game_1' and \
                (runner.t_env - last_demo_T) / args.demo_interval >= 1.0 and buffer.can_sample(args.batch_size):
            ### demo
            episode_sample = cp.deepcopy(buffer.sample(1))
            for i in range(args.n_actions):
                for j in range(args.n_actions):
                    if args.env == 'mmdp_game_1' and args.joint_random_policy_eps > 0:
                        logger.log_stat("joint_prob_%d_%d" % (i, j), runner.mac.action_selector.joint_action_seeds[i * 2 + j], runner.t_env)
                    new_actions = th.Tensor([i, j]).unsqueeze(0).repeat(args.episode_limit + 1, 1)
                    if i == 0 and j == 0:
                        rew = th.Tensor([1, ])
                    else:
                        rew = th.Tensor([0, ])
                    if i == 1 and j == 1:
                        new_obs = th.Tensor([1, 0]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                    else:
                        new_obs = th.Tensor([0, 1]).unsqueeze(0).unsqueeze(0).repeat(args.episode_limit, args.n_agents, 1)
                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    episode_sample['actions'][0, :, :, 0] = new_actions
                    episode_sample['obs'][0, 1:, :, :] = new_obs
                    episode_sample['reward'][0, 0, 0] = rew
                    new_actions_onehot = th.zeros(episode_sample['actions'].squeeze(3).shape + (args.n_actions,))
                    new_actions_onehot = new_actions_onehot.scatter_(3, episode_sample['actions'].cpu(), 1)
                    episode_sample['actions_onehot'][:] = new_actions_onehot

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    #print("action pair: %d, %d" % (i, j))
                    learner.train(episode_sample, runner.t_env, episode, show_demo=True, save_data=(i, j))
            last_demo_T = runner.t_env
            #time.sleep(1)

        if (args.env == 'matrix_game_1' or args.env == 'matrix_game_2' or args.env == 'matrix_game_3') and \
                (runner.t_env - last_demo_T) / args.demo_interval >= 1.0 and buffer.can_sample(args.batch_size):
            ### demo
            episode_sample = cp.deepcopy(buffer.sample(1))
            for i in range(args.n_actions):
                for j in range(args.n_actions):
                    new_actions = th.Tensor([i, j]).unsqueeze(0).repeat(args.episode_limit + 1, 1)
                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    episode_sample['actions'][0, :, :, 0] = new_actions
                    new_actions_onehot = to_cuda(th.zeros(episode_sample['actions'].squeeze(3).shape + (args.n_actions,)), self.args.device)
                    new_actions_onehot = new_actions_onehot.scatter_(3, to_cuda(episode_sample['actions'], self.args.device), 1)
                    episode_sample['actions_onehot'][:] = new_actions_onehot
                    if i == 0 and j == 0:
                        rew = th.Tensor([8, ])
                    elif i == 0 or j == 0:
                        rew = th.Tensor([-12, ])
                    else:
                        rew = th.Tensor([0, ])
                    if args.env == 'matrix_game_3':
                        if i == 1 and j == 1 or i == 2 and j == 2:
                            rew = th.Tensor([6, ])
                    episode_sample['reward'][0, 0, 0] = rew

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    #print("action pair: %d, %d" % (i, j))
                    learner.train(episode_sample, runner.t_env, episode, show_demo=True, save_data=(i, j))
            last_demo_T = runner.t_env
            #time.sleep(1)
        # （4） 保存模型
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            if args.double_q:
                os.makedirs(save_path + '_x', exist_ok=True)

            if args.learner == 'curiosity_learner' or args.learner == 'curiosity_learner_new' or args.learner == 'qplex_curiosity_learner'\
                    or args.learner == 'qplex_curiosity_rnd_learner' or args.learner =='qplex_rnd_history_curiosity_learner':
                os.makedirs(save_path + '/mac/', exist_ok=True)
                os.makedirs(save_path + '/extrinsic_mac/', exist_ok=True)
                os.makedirs(save_path + '/predict_mac/', exist_ok=True)
                if args.learner == 'curiosity_learner_new' or args.learner == 'qplex_curiosity_rnd_learner'or args.learner =='qplex_rnd_history_curiosity_learner':
                    os.makedirs(save_path + '/rnd_predict_mac/', exist_ok=True)
                    os.makedirs(save_path + '/rnd_target_mac/', exist_ok=True)

            if args.learner == 'rnd_learner' or args.learner == 'rnd_learner2' or args.learner =='qplex_rnd_learner'\
                    or args.learner =='qplex_rnd_history_learner' or args.learner =='qplex_rnd_emdqn_learner' :
                os.makedirs(save_path + '/mac/', exist_ok=True)
                os.makedirs(save_path + '/rnd_predict_mac/', exist_ok=True)
                os.makedirs(save_path + '/rnd_target_mac/', exist_ok=True)
            if args.learner == 'qplex_curiosity_single_learner' or "qplex_curiosity_single_fast_learner":
                os.makedirs(save_path + '/mac/', exist_ok=True)
                os.makedirs(save_path + '/predict_mac/', exist_ok=True)
                os.makedirs(save_path + '/soft_update_target_mac/', exist_ok=True)


            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run * args.num_circle

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    # ————————————————————————————————————————————while止——————————————————————————————————————
    if args.is_save_buffer and save_buffer.is_from_start:
        save_buffer.is_from_start = False
        save_one_buffer(args, save_buffer, env_name, from_start=True)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
