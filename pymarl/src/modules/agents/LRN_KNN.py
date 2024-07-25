import numpy as np
from sklearn.neighbors import BallTree,KDTree
import os
import torch as th
import gc
import datetime


#each action -> a lru_knn buffer
class LRU_KNN:
    def __init__(self, capacity, z_dim, env_name):
        self.env_name = env_name
        self.capacity = capacity
        self.states = np.empty((capacity, z_dim), dtype = np.float32)   # 需要保存的
        self.q_values_decay = np.zeros(capacity)                        # 需要保存的
        # 用来存放reward和下一时刻s'的embedding
        self.reward_and_nextState = np.empty(capacity, dtype = object)

        self.lru = np.zeros(capacity)                                   # 需要保存的
        self.curr_capacity = 0
        self.tm = 0.0
        self.tree = None
        self.addnum = 0
        self.buildnum = 256
        self.buildnum_max = 256
        self.bufpath = './buffer/%s'%self.env_name
        self.build_tree_times = 0
        self.build_tree = False

    def load(self, action):
        try:
            assert(os.path.exists(self.bufpath))
            lru = np.load(os.path.join(self.bufpath, 'lru_%d.npy'%action))
            cap = lru.shape[0]
            self.curr_capacity = cap
            self.tm = np.max(lru) + 0.01
            self.buildnum = self.buildnum_max

            self.states[:cap] = np.load(os.path.join(self.bufpath, 'states_%d.npy'%action))
            self.q_values_decay[:cap] = np.load(os.path.join(self.bufpath, 'q_values_decay_%d.npy'%action))
            self.lru[:cap] = lru
            self.tree = KDTree(self.states[:self.curr_capacity]) # 状态的二叉查找树
            print ("load %d-th buffer success, cap=%d" % (action, cap))
        except:
            print ("load %d-th buffer failed" % action)

    def save(self, action):
        if not os.path.exists('buffer'):
            os.makedirs('buffer')
        if not os.path.exists(self.bufpath):
            os.makedirs(self.bufpath)
        np.save(os.path.join(self.bufpath, 'states_%d'%action), self.states[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'q_values_decay_%d'%action), self.q_values_decay[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'lru_%d'%action), self.lru[:self.curr_capacity])

    #找最近邻点并更新节点值
    def peek(self, key, value_decay, modify, reward=[], z_next=[], propagate=False, is_vp=False): # key为状态的embedding表示z
        if modify == False:
            x = 1
        if self.curr_capacity==0 or self.build_tree == False:
            return None
        # 在tree中进行最近邻搜索
        # dist查询点到最近邻点的距离，ind为最近邻点在原始数据集中的索引
        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]

        # peek到了
        if np.allclose(self.states[ind], key, atol=1e-8):
            if not is_vp:
                self.lru[ind] = self.tm
                self.tm +=0.01
            # 更新Q值,添加reward_and_nextState
            if modify:
                if value_decay > self.q_values_decay[ind]:
                    self.q_values_decay[ind] = value_decay
                if len(z_next) != 0: # 不是最后一刻，即说明有s‘
                    temp = [reward, z_next]
                    # 如果没有过后继节点
                    if self.reward_and_nextState[ind] == None:
                        self.reward_and_nextState[ind] = [temp]
                    else:
                        self.reward_and_nextState[ind].append(temp)
            # 如果有后继节点则进行value propagation
            if propagate and self.reward_and_nextState[ind] != None:
                reward_and_nextState = self.reward_and_nextState[ind]  # 所有后继结点
                num_next = len(self.reward_and_nextState[ind])
                nextState_q_values = np.zeros(num_next, dtype = np.float32)
                rewards = np.zeros(num_next, dtype = np.float32)
                # peek到 [后继节点] 的q值
                for j in range(num_next):
                    q_value = self.peek(reward_and_nextState[j][1], None, False)
                    q_value = 0 if q_value == None else q_value
                    nextState_q_values[j] = q_value
                    rewards[j] = reward_and_nextState[j][0][0]
                # print('smy:rewards.shape={},nextState_q_values.shape={}'.format(rewards.shape,nextState_q_values.shape))

                one_step_TD = max(rewards + 0.99 * nextState_q_values)
                if one_step_TD > self.q_values_decay[ind]:
                    self.q_values_decay[ind] = one_step_TD

            return self.q_values_decay[ind]
        return None

    # 找knn个最近邻点，然后value取平均
    def knn_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity==0 or self.build_tree == False:
            return 0.0, 0.0

        dist, ind = self.tree.query([key], k=knn)

        value = 0.0
        value_decay = 0.0
        for index in ind[0]:
            value_decay += self.q_values_decay[index]
            self.lru[index] = self.tm
            self.tm+=0.01

        q_decay = value_decay / knn

        return q_decay

    # 添加一个新条目
    def add(self, key, value_decay,  reward, z_next):
        # print('KNN-SMY:z_next={}',z_next)
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.q_values_decay[old_index] = value_decay
            if len(z_next) != 0:  # 不是最后一刻，即说明有s‘
                temp = [reward, z_next]
                self.reward_and_nextState[old_index] = [temp]
            self.lru[old_index] = self.tm
        else:
            self.states[self.curr_capacity] = key
            self.q_values_decay[self.curr_capacity] = value_decay
            if len(z_next) != 0:  # 不是最后一刻，即说明有s‘
                temp = [reward, z_next]
                self.reward_and_nextState[self.curr_capacity] = [temp]
            self.lru[self.curr_capacity] = self.tm
            self.curr_capacity+=1
        self.tm += 0.01

    def update_kdtree(self):
        if self.build_tree:
            del self.tree
        self.tree = KDTree(self.states[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()
    # 对memory中所有条目进行propagation
    def value_propagation_one_step(self):
        # 遍历所有state_embedding，基于后继节点进行1步TD更新
        updata_num = 0 # 统计value propagation的时候会有多少次one step return值会大于memory中存储的
        num_next_sum, peek_total_time, update_total_time = 0, 0.0, 0.0
        print("start value propagation...........")
        for i in range(self.curr_capacity):  # 1w step时才2k多
            # 如果存在后继节点：得到当前时刻reward和后继节点embedding
            if self.reward_and_nextState[i] != None:
                reward_and_nextState_i = self.reward_and_nextState[i]  #[[array([0.], dtype=float32), array([0., 0., 0., 0.])]]
                num_next = len(reward_and_nextState_i)
                num_next_sum += num_next
                nextState_q_values = np.zeros(num_next, dtype=np.float32 )
                rewards = np.zeros(num_next, dtype=np.float32)
                # peek到后继节点的q值
                timea = datetime.datetime.now()
                for j in range(num_next):
                    # print(reward_and_nextState_i)
                    q_value = self.peek(reward_and_nextState_i[j][1], None,False, is_vp=True)
                    q_value = 0 if q_value == None else q_value
                    nextState_q_values[j] = q_value
                    rewards[j] = reward_and_nextState_i[j][0][0]
                # print('smy:rewards.shape={},nextState_q_values.shape={}'.format(rewards.shape,nextState_q_values.shape))
                timeb = datetime.datetime.now()
                peek_total_time += (timeb - timea).total_seconds()
                timea = datetime.datetime.now()
                one_step_TD = np.max(rewards + 0.99 * nextState_q_values)
                if one_step_TD > self.q_values_decay[i]:
                    updata_num += 1
                    self.q_values_decay[i] = one_step_TD
                timeb = datetime.datetime.now()
                update_total_time += (timeb - timea).total_seconds()
        print('now curr_capacity = {}, 平均后继节点个数 = {}, 替换one-step return的次数 = {}, peek操作总时间 = {}, 更新操作总时间 = {}'.format(self.curr_capacity, num_next_sum/self.curr_capacity, updata_num, peek_total_time,update_total_time))
        print("value propagation end...........")
        return updata_num / self.curr_capacity  # 返回更新率
    
    
    # 对memory中一个条目进行propagation并返回更新后对q_value
    # def value_propagation_a_item(self, z):
    #     q = self.peek(z, None, modify=False)  # H(Φ(s[j]))
    #     # 遍历所有后继节点进行propagation
    #     if q != None:

