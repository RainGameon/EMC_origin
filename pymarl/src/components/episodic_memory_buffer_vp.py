import numpy as np
from modules.agents.LRN_KNN import LRU_KNN

# 可以进行一步value propagation的episodic memory
class Episodic_memory_buffer_vp:
    def __init__(self, args,scheme):
        self.ec_buffer = LRU_KNN(args.emdqn_buffer_size, args.emdqn_latent_dim, 'game')
        self.rng = np.random.RandomState(123456)  # 随机数生成器对象 deterministic, erase 123456 for stochastic
        self.random_projection = self.rng.normal(loc=0, scale=1. / np.sqrt(args.emdqn_latent_dim),
                                       size=(args.emdqn_latent_dim, scheme['state']['vshape']))
        self.q_episodic_memeory_cwatch = []
        self.args=args
        self.update_counter =0 # learner.train中会+1
        self.qecwatch=[]
        self.qec_found=0 # 记录成功找到 Q 值的次数


    def update_kdtree(self):
        self.ec_buffer.update_kdtree()

    def peek(self, key, value_decay, modify,propagate=False):
        return self.ec_buffer.peek(key, value_decay, modify,propagate)

    def update_ec(self, episode_batch):
        ep_state = episode_batch['state'][0, :]     #(T, state_shape:216)   # batchsize=1（因为run一条trajectory就update）
        ep_action = episode_batch['actions'][0, :]  #(T, n_agents, 1)
        ep_reward = episode_batch['reward'][0, :]   #(T, 1)
        Rtd = 0.
        # print('SMY: max_seq_length={}'.format(episode_batch.max_seq_length))
        # print('SMY: ep_state.shape={},ep_action.shape={},ep_reward.shape={}'.format(ep_state.shape,ep_action.shape,ep_reward.shape))

        # 从T-1开始遍历，到0
        z_next, z = [], None # 下一时刻即s'的embedding
        for t in range(episode_batch.max_seq_length - 1, -1, -1):
            s = ep_state[t]  # 1,1,216
            a = ep_action[t]
            r = ep_reward[t]
            z = np.dot(self.random_projection, s.flatten().cpu())  # 标量emdqn_latent_dim
            Rtd = r + self.args.gamma * Rtd
            z = z.reshape((self.args.emdqn_latent_dim))            # 1D 的 NumPy 数组
            qd = self.ec_buffer.peek(z, Rtd, True, r.cpu().numpy() , z_next) # propagate = self.args.is_value_propagation)
            if qd == None:  # new action
                self.ec_buffer.add(z, Rtd, r.cpu().numpy() , z_next)
            z_next = z

    def hit_probability(self):
        return (1.0 * self.qec_found / self.args.batch_size / self.update_counter)
    def value_propagation(self):
        # 遍历所有state_embedding，基于后继节点进行1步TD更新
        updata_rate = self.ec_buffer.value_propagation_one_step()
        return updata_rate
    # def value_propagation_a_item(self,z):
    #     updata_rate = self.ec_buffer.value_propagation_a_item(z)
        
