from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run # 1
        assert self.batch_size == 1

        if 'stag_hunt' in self.args.env:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)
        else:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0  # 经常用到

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # 创建一个函数对象new_batch，存储和处理多个episodes的数据，
        # 一般包含：状态（state）、可用动作（avail_actions）、观测（obs）    //交互前可知
        #         动作（actions）、奖励（reward）、终止标志（terminated）  //交互后才知
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac # mac是一个多智能体控制器（Multi-Agent Controller），用于实现智能体的决策和交互。

    def get_env_info(self):
        return self.env.get_env_info() # 可以获取例如环境的规模、可用的动作空间、奖励函数等

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    '''————进行交互并保存episode数据，包括：state、avail_actions、obs、actions、reward、terminated————'''
    def run(self, test_mode=False):
        self.reset()       # 重置环境和状态
        terminated = False
        episode_return = 0 # 记录回报值
        self.sequence = []
        self.mac.init_hidden(batch_size=self.batch_size)
        # 进入一个循环，直到环境终止
        rewards = [] # 存放这条trajectory的奖励
        start_time = self.t
        while not terminated:
            # 收集先前的状态、可能的动作、观测
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            rewards.append(reward)
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "flag_win": [(bool(env_info['battle_won']),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1
        # 收集最后一个时间步的【状态、可用动作、观测、所选动作】数据，并将其更新到批次数据中
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        # # 计算Rtd并将return信息更新到episodebatch中
        # Rtd = 0.0
        # end_time = self.t 
        # # print('SUNMENGYAO___________________start_t={} end_t={}'.format(start_time,end_time))
        # # print('SUNMENGYAO___________________rewards.len={}'.format(len(rewards)))
        # for r in rewards[::-1]:
        #     Rtd = Rtd * 0.99 + r
        #     self.batch.update({"return_H":  [(Rtd,)],}, ts=end_time-1)
        #     end_time -= 1

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        # 【日志前缀】用于区分训练日志和测试日志
        log_prefix = "test_" if test_mode else ""
        # 将当前的统计信息（cur_stats）与环境信息（env_info）进行合并更新
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        # 【记录已经进行的回合数】将当前统计信息中的"n_episodes"键对应的值加1，并将结果更新回统计信息中。
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        # 【记录已经进行的总时间步数】
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
        # 更新回报值列表
        cur_returns.append(episode_return)

        # 如果处于测试模式且已经收集了指定数量的测试回合，则调用self._log()方法记录当前的回报值和统计信息。
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        # 否则，如果距离上次记录训练统计信息的时间超过指定的时间间隔，则调用self._log()方法记录当前的回报值和统计信息
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
