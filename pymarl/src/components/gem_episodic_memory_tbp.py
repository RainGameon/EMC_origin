from stable_baselines.td3.episodic_memory import EpisodicMemory
import numpy as np
import time


class EpisodicMemoryTBP(EpisodicMemory):
    def __init__(self,args, buffer_size, state_dim, action_shape, obs_shape,
                 gamma=0.99,
                 alpha=0.6,max_step=1000):
        super(EpisodicMemoryTBP, self).__init__(args,buffer_size, state_dim, action_shape, obs_shape,
                                                gamma, alpha,max_step)
        del self._q_values
        self._q_values = -np.inf * np.ones((buffer_size + 1, 2))
        # self.max_step = max_step

    # def compute_approximate_return_double(self, obses, actions=None):
    #     return np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses}))

    def update_memory(self, q_base=0, use_knn=False, beta=-1):
        discount_beta = beta ** np.arange(self.max_step) # 折扣因子
        trajs = self.retrieve_trajectories()
        for traj in trajs:
            # 对于一条trajectory，计算4个Q(s,a)
            approximate_qs = self.compute_approximate_return_double(self.obs_buffer[traj], self.action_buffer[traj]) # obs_buffer[traj]为obs
            num_q = len(approximate_qs)
            if num_q >= 4:
                approximate_qs = approximate_qs.reshape((2, num_q//2, -1)) # 【2，2，T】分成两份分别用于计算Q1，Q2
                approximate_qs = np.min(approximate_qs, axis=1)  # clip double q  【2，T】？

            else:
                assert num_q == 2
                approximate_qs = approximate_qs.reshape(2, -1)
            approximate_qs = np.concatenate([np.zeros((2, 1)), approximate_qs], axis=1)  # 【2，T+1】？
            self.q_values[traj] = 0
            #————————————————————————————————从这开始往下————————————————————————————
            rtn_1 = np.zeros((len(traj), len(traj))) # 【T,T】
            rtn_2 = np.zeros((len(traj), len(traj))) # 【T,T】
            # 遍历每一步，计算1步return，即h=0
            for i, s in enumerate(traj):
                rtn_1[i, 0], rtn_2[i, 0] = self.reward_buffer[s] + self.gamma * (1 - self.truly_done_buffer[s]) * (approximate_qs[:, i] - q_base)

            # 遍历每一步，计算1到T步return（公式7）
            for i, s in enumerate(traj):
                rtn_1[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_1[i - 1, :-1]
                rtn_2[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_2[i - 1, :-1]
            if beta > 0:
                double_rtn = [
                    [np.dot(rtn_2[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                        discount_beta[:min(i + 1, self.max_step)]),
                     np.dot(rtn_1[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                         discount_beta[:min(i + 1, self.max_step)])]
                    for i in range(len(traj))]
            else: # 计算R1和R2预估最高回报 公式78
                double_rtn = [
                    [rtn_2[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])],    # rtn_1用来得到最高估计回报对应的step
                     rtn_1[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])]] for i
                    in
                    range(len(traj))]
                # double_rtn = [
                #     [rtn_1[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])],
                #      rtn_2[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])]] for i
                #     in
                #     range(len(traj))]
                # double_rtn = np.min(np.array(double_rtn),axis=1,keepdims=True)
                # double_rtn = np.repeat(double_rtn,2,axis=1)
            # self.q_values[traj] = np.maximum(np.array(double_rtn),np.minimum(rtn_1[:,0],rtn_2[:,0]))
            one_step_q = np.array([rtn_1[:, 0], rtn_2[:, 0]]).transpose() # 一步return
            # 选择计算得到的R值和一步TD return的最大值作为预估最高回报q_values
            self.q_values[traj] = np.maximum(np.array(double_rtn),one_step_q)
