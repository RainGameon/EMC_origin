import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop, Adam
from utils.torch_utils import to_cuda
import numpy as np
from .vdn_Qlearner import vdn_QLearner
from .vdn_Qlearner_Curiosity import vdn_QLearner_Curiosity
from .vdn_Qlearner_Curiosity_individual import vdn_QLearner_Curiosity_individual
import os
from modules.w_predictor.predict_net import Predict_Network_OS, Predict_Network
from modules.w_predictor.predict_vae import Predict_Network_vae


class QPLEX_curiosity_vdn_Learner_ps:
    def __init__(self, mac, scheme, logger, args,groups=None):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.save_buffer_cnt = 0
        if self.args.save_buffer:
            self.args.save_buffer_path = os.path.join(self.args.save_buffer_path, str(self.args.seed))

        self.mixer = None
        self.vdn_learner = vdn_QLearner_Curiosity_individual(mac, scheme, logger, args)
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        # 与influence reward有关
        self.eval_predict_with_i = Predict_Network(args.rnn_hidden_dim + args.n_actions, args.predict_net_dim, args.n_actions)
        self.target_predict_with_i = copy.deepcopy(self.eval_predict_with_i)
        # self.eval_predict_without_i = Predict_Network(args.obs_shape + args.n_actions, args.predict_net_dim, args.obs_shape)
        # self.target_predict_without_i = copy.deepcopy(self.eval_predict_without_i)
        # 与influence reward有关参数与Adam
        self.predictor_params = list(self.eval_predict_with_i.parameters())
        # self.predictor_params += list(self.eval_predict_without_i.parameters())
        self.predictor_optimiser = Adam(params=self.predictor_params, lr=args.predictor_lr)
        self.decay_stats_cur_t = 0
        self.decay_stats_inf_t = 0
        self.beta1 = self.args.beta1
        self.beta2 = self.args.beta2
        self.beta = self.args.beta
        # ___________________________________
        # 对p(s)进行建模
        self.predict_state_win = Predict_Network(args.state_shape, args.predict_net_dim, 2)
        self.pre_state_params = list(self.predict_state_win.parameters())
        self.pre_state_optimiser = Adam(params=self.pre_state_params, lr=args.predictor_lr)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_update_memory_t = 0
        self.save_buffer_cnt = 0
        self.n_actions = self.args.n_actions
        self.start_value_propagation = False

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,intrinsic_rewards,
                  show_demo=False, save_data=None, show_v=False, save_buffer=False,ec_buffer=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]

        terminated = batch["terminated"][:, :-1].float() # 只有终止时间步之前的时间步对应的元素为1，其他时间步对应的元素为0,则terminated全为0
        mask = batch["filled"][:, :-1].float()  # 获取所有时间步的填充信息
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # mask终止状态之前的时间步对应的元素为填充信息，终止状态后为0
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        # for influence reward
        state = batch["state"][:, :-1]
        obs = batch["obs"][:, :-1]
        obs_clone = obs.clone()
        obs_next = batch["obs"][:, 1:]
        obs_next_clone = obs_next.clone()
        mask_next = batch["filled"][:, 1:].float()  # [bs, T-1, 1]
        mask_next[:, 1:] = mask_next[:, 1:] * (1 - terminated[:, 1:])
        # Calculate estimated Q-Values
        mac.init_hidden(batch.batch_size)
        mac_out, hidden_out = mac.forward(batch, batch.max_seq_length, batch_inf=True) # 选择的动作
        # 如果需要保存缓冲区，那么保存
        if save_buffer:
            # 将内在奖励转换为NumPy数组并保存
            curiosity_r=intrinsic_rewards.clone().detach().cpu().numpy()
            # rnd_r = rnd_intrinsic_rewards.clone().detach().cpu().numpy()
            # extrinsic_mac_out_save=extrinsic_mac_out.clone().detach().cpu().numpy()
            # 将其他相关变量转换为NumPy数组并保存
            mac_out_save = mac_out.clone().detach().cpu().numpy()
            actions_save=actions.clone().detach().cpu().numpy()
            terminated_save=terminated.clone().detach().cpu().numpy()
            state_save=batch["state"][:, :-1].clone().detach().cpu().numpy()
            # 创建一个字典，保存所有需要的数据
            data_dic={'curiosity_r':curiosity_r,
                                 # 'extrinsic_Q':extrinsic_mac_out_save,
                        'control_Q':mac_out_save,'actions':actions_save,'terminated':terminated_save,
                        'state':state_save}
            # 保存缓冲区数据data_dic到文件
            self.save_buffer_cnt += self.args.save_buffer_cycle
            if not os.path.exists(self.args.save_buffer_path):
                os.makedirs(self.args.save_buffer_path)
            np.save(self.args.save_buffer_path +"/"+ 'data_{}'.format(self.save_buffer_cnt), data_dic)
            print('save buffer ({}) at time{}'.format(batch.batch_size, self.save_buffer_cnt))
            return

        # ！！！！【实际选的动作对应的Q值】根据样本中action选择每个智能体采取这个action的Q值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        x_mac_out = mac_out.clone().detach()
        # 并将不可行动作的Q值设置为一个极小值，以便在下一步找到最大动作
        x_mac_out[avail_actions == 0] = -9999999
        # ！！！！【最大Q值】找到每个时间步的最大动作的Q值和对应的动作索引
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
        # 分离动作索引，并添加一个维度
        max_action_index = max_action_index.detach().unsqueeze(3)
        # 判断当前动作是否为最大动作，并将结果转换为浮点型
        is_max_action = (max_action_index == actions).int().float()
        # 展示演示信息
        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

        # 初始化目标网络的隐藏状态
        self.target_mac.init_hidden(batch.batch_size)
        # 在目标网络上进行前向传播，获取目标网络的输出，！！注意这里是从t=1开始
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[0][:, 1:, ...]
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999    # 屏蔽不可行动作

        # ————————————————————————计算influence-based intrinsic reward——————————————————————
        if self.args.is_use_influence:
            with th.no_grad():
                actions_one_hot = batch["actions_onehot"][:, :-1]  # bs,t, n_agents, action_hot/n_actions
                actions_one_hot_clone = actions_one_hot.clone()  # （bs,T-1,n_agent, action_hot/n_actions)
                state_clone = state.clone()                      # （bs,T-1,state_shape)
                state_input = state_clone.unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)  # （bs,T-1,n_agents - 1,state_shape)
                obs_diverge = []  # 存放当前agent动作对其他agent观测变化对影响
                diverge = []
                epsilon = 1e-8
                
                # influence reward B(infb):将policy变化作为影响
                for i in range(self.args.n_agents):
                    # ①基于Qj进行softmax，得到Q-network对应的policy_Q (bs,T-1, n_agent-1, n_action)
                    x_mac_out_clone = x_mac_out.clone().detach()[:, :-1]
                    x_mac_out_clone_without_i = th.cat((x_mac_out_clone[:,:,:i], x_mac_out_clone[:,:,i+1:]), dim = -2)
                    probs = F.softmax(x_mac_out_clone_without_i / self.args.temperature, dim=-1)

                    # ②基于predict_with_action网络得：with ai情况下对应的 porobs_with_i
                    hidden_out_clone = hidden_out.clone().detach()[:,:-1]
                    hidden_out_clone_without_i = th.cat((hidden_out_clone[:,:,:i],hidden_out_clone[:,:,i+1:]), dim = -2)
                    actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1,self.args.n_agents - 1,1)  # (bs,T-1,n_agent-1,n_actions) ,a[t][i]   得到当前 agent i 动作
                    influence_inputs_with_i = th.cat((hidden_out_clone_without_i, actions_one_hot_i), dim=-1)  # (bs,T-1,n_agent-1,(rnn_hidden_dim+n_actions)  # 考虑了agent i 在时刻t的动作
                    porobs_with_i = self.target_predict_with_i(influence_inputs_with_i, pre_action = True)    # (bs,T-1,n_agent-1,n_action)

                    # ③对 policy_Q 和 porobs_with_i 求JS散度
                    even_probs = (probs + porobs_with_i) / 2 + epsilon
                    probs_epsilon, porobs_with_i_epsilon = probs + epsilon, porobs_with_i + epsilon
                    diverge_i = 0.5 * th.sum((probs_epsilon * th.log(probs_epsilon / even_probs + porobs_with_i_epsilon * th.log(porobs_with_i_epsilon / even_probs) )), dim = -1)  #(bs,T-1,n_agent-1)
                    diverge_i = th.mean(diverge_i, dim = -1).unsqueeze(-1)                #(bs,T-1,1)
                    diverge.append(diverge_i * mask * self.args.beta1)
                influence_intrinsic_reward = th.stack(diverge, dim=2).squeeze(-1).detach()  # bs, T-1, n_agents
                '''
                # influence reward C(infc):将policy变化作为影响(agent i影响的是下一个时刻的j的动作）
                for i in range(self.args.n_agents):
                    # ①基于Qj进行softmax，得到Q-network对应的policy_Q (bs,T-1, n_agent-1, n_action)
                    x_mac_out_clone = x_mac_out.clone().detach()[:, 1:]
                    x_mac_out_clone_without_i = th.cat((x_mac_out_clone[:, :, :i], x_mac_out_clone[:, :, i + 1:]),
                                                       dim=-2)
                    probs = F.softmax(x_mac_out_clone_without_i / self.args.temperature, dim=-1)

                    # ②基于predict_with_action网络得：with ai情况下对应的 porobs_with_i
                    hidden_out_clone = hidden_out.clone().detach()[:, 1:]
                    hidden_out_clone_without_i = th.cat((hidden_out_clone[:, :, :i], hidden_out_clone[:, :, i + 1:]),dim=-2)
                    actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)  # (bs,T-1,n_agent-1,n_actions) ,a[t][i]   得到当前 agent i 动作
                    influence_inputs_with_i = th.cat((hidden_out_clone_without_i, actions_one_hot_i), dim=-1)  # (bs,T-1,n_agent-1,(rnn_hidden_dim+n_actions)  # 考虑了agent i 在时刻t的动作
                    porobs_with_i = self.target_predict_with_i(influence_inputs_with_i,  pre_action=True)  # (bs,T-1,n_agent-1,n_action)

                    # ③对 policy_Q 和 porobs_with_i 求JS散度
                    even_probs = (probs + porobs_with_i) / 2 + epsilon
                    probs_epsilon, porobs_with_i_epsilon = probs + epsilon, porobs_with_i + epsilon
                    diverge_i = 0.5 * th.sum((probs_epsilon * th.log(
                        probs_epsilon / even_probs + porobs_with_i_epsilon * th.log(
                            porobs_with_i_epsilon / even_probs))), dim=-1)  # (bs,T-1,n_agent-1)
                    diverge_i = th.mean(diverge_i, dim=-1).unsqueeze(-1)  # (bs,T-1,1)
                    diverge.append(diverge_i * mask * self.args.beta1)
                influence_intrinsic_reward = th.stack(diverge, dim=2).squeeze(-1).detach()  # bs, T-1, n_agents
                '''
                '''
                # influence reward A(infa):将预测的下一时刻的观测变化作为influence reward（based on COIN论文）
                for i in range(self.args.n_agents):
                    #【action without i】
                    actions_one_hot_i_pre = th.cat((actions_one_hot_clone[:, :, :i], actions_one_hot_clone[:, :, i + 1:]), dim=2)  # (bs,T-1,n_agent-1,n_actions) ,a[t][-i]  得到除了当前 agent i 其他智能体对动作
                    #【action i】
                    actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)  # (bs,T-1,n_agent-1,n_actions) ,a[t][i]   得到当前 agent i 动作
                    obs_i = obs_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)

                    obs_next_i = th.cat((obs_next_clone[:, :, :i], obs_next_clone[:, :, i + 1:]), dim=2)  # (bs,T-1,n_agent-1,obs_dim)    o[t+1][-i]得到除了当前 agent i 其他智能体下时刻的观测

                    obs_without_i = th.cat((obs_clone[:, :, :i], obs_clone[:, :, i + 1:]), dim=2)
                    hidden_out_clone = hidden_out.clone().detach()[:, :-1]
                    hidden_out_clone_without_i = th.cat((hidden_out_clone[:, :, :i], hidden_out_clone[:, :, i + 1:]), dim=-2)

                    # 构造输入（obs+action）
                    influence_inputs_without_i = th.cat((obs_without_i, actions_one_hot_i_pre), dim=-1)  # (bs,T-1,n_agent-1,rnn_dim+n_action)  # 不考虑agent i 在时刻t的动作
                    influence_inputs_with_i = th.cat((obs_without_i, actions_one_hot_i_pre, actions_one_hot_i), dim=-1)  # (bs,T-1,n_agent-1,2*(obs_shape+n_actions)+state_shape)  # 考虑了agent i 在时刻t的动作

                    log_p_o_i = self.target_predict_without_i.get_log_pi(influence_inputs_without_i, obs_next_i)  # (bs,T-1,n_agent-1)
                    log_q_o_i = self.target_predict_with_i.get_log_pi(influence_inputs_with_i, obs_next_i)  # (bs,T-1,n_agent-1,obs_shape)

                    # 计算agent i获得的influence:
                    obs_diverge_i = self.args.beta * th.sum(log_q_o_i, -1, keepdim=True) - th.sum(log_p_o_i, -1, keepdim=True)  # (bs, T-1, 1)
                    obs_diverge.append(obs_diverge_i * mask_next * self.args.beta1)
                influence_intrinsic_reward = th.stack(obs_diverge, dim=2).squeeze(-1).detach()  # bs, T-1, n_agents
                '''
        # Qi_target
        if self.args.double_q:
            # 获取最大化实际Q值的动作（用于双Q学习）   Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            # 从目标网络中选择最大化实际Q值的动作的Q值
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            # 计算目标网络的最大Q值和实际Q值的最大化动作
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = to_cuda(th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)), self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs,_ = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Qtot
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                if self.args.is_use_influence:
                    # 计算值网络的输出、注意力正则化项和注意力分布的熵
                    ans_chosen, q_attend_regs, head_entropies =  mixer(chosen_action_qvals - intrinsic_rewards, batch["state"][:, :-1], is_v=True)
                    # 计算优势网络的输出
                    ans_adv, _, _ = mixer(chosen_action_qvals - intrinsic_rewards , batch["state"][:, :-1], actions=actions_onehot,
                                          max_q_i=max_action_qvals, is_v=False)
                    chosen_action_qvals = ans_chosen + ans_adv
                else:
                    # 计算值网络的输出、注意力正则化项和注意力分布的熵
                    ans_chosen, q_attend_regs, head_entropies = \
                        mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                    # 计算优势网络的输出
                    ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"][:, :-1],
                                          actions=actions_onehot,
                                          max_q_i=max_action_qvals, is_v=False)
                    chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # H(s)
        if self.args.use_emdqn:
            ec_buffer.update_counter += 1
            qec_input = chosen_action_qvals.clone().detach()
            qec_input_new = []
            for i in range(self.args.batch_size):
                # 对于第i个样本，即第i条轨迹？
                qec_tmp = qec_input[i, :]
                # 遍历每一个时间步
                for j in range(1, batch.max_seq_length):
                    if not mask[i, j - 1]: # 如果mask为0，代表当前已结束，则继续执行下一轮循环
                        continue
                    # 将当前状态与随机投影矩阵相乘，得到投影向量z
                    z = np.dot(ec_buffer.random_projection, batch["state"][i][j].cpu())  # Φ(s[j])
                    # 通过 ec_buffer 查找 z 对应的 Q 值
                    q = ec_buffer.peek(z, None, modify=False, propagate=self.args.is_value_propagation)  # H(Φ(s[j]))
                    if q != None: # 如果找到了相应的 Q 值，则更新 qec_tmp 中对应位置的数值
                        qec_tmp[j - 1] = self.args.gamma * q + rewards[i][j - 1] # H(Φ(s[j-1]),a(j-1)) = r(j-1) + gamma * H
                        ec_buffer.qecwatch.append(q)
                        ec_buffer.qec_found += 1  # 记录成功找到 Q 值的次数
                qec_input_new.append(qec_tmp)
            qec_input_new = th.stack(qec_input_new, dim=0)

            # print("qec_mean:", np.mean(ec_buffer.qecwatch))
            episodic_q_hit_pro = 1.0 * ec_buffer.qec_found / self.args.batch_size / ec_buffer.update_counter / batch.max_seq_length
            # print("qec_fount: %.2f" % episodic_q_hit_pro)

        # P(s)
        if self.args.is_use_statewon:
            won_lose = self.predict_state_win(state, is_softmax = True)  #(bs, T-1, 2)
            won_lose_clone = won_lose[:,:,0].clone().detach().unsqueeze(-1)
            # print('won_lose_clone.shape={}, rewards.shape={}'.format(won_lose_clone.shape, rewards.shape))
            targets = won_lose_clone + rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        # print('td_error.shape={}'.format(td_error.shape))
        # print('mask1.shape={}'.format(mask.shape))
        mask = mask.expand_as(td_error)
        # print('mask2.shape={}'.format(mask.shape))
        if self.args.use_emdqn:
            emdqn_td_error = qec_input_new.detach() - chosen_action_qvals
            emdqn_masked_td_error = emdqn_td_error * mask

        if show_v:
            mask_elems = mask.sum().item()

            actual_v = rewards.clone().detach()
            for t in reversed(range(rewards.shape[1] - 1)):
                actual_v[:, t] += self.args.gamma * actual_v[:, t + 1]
            self.logger.log_stat("test_actual_return", (actual_v * mask).sum().item() / mask_elems, t_env)

            self.logger.log_stat("test_q_taken_mean", (chosen_action_qvals * mask).sum().item() / mask_elems, t_env)
            return

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss
        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()
        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()
        # 更新预测 state win的参数
        if self.args.is_use_statewon:
            flag_win = batch["flag_win"][:, :-1].float()  #(bs,T-1,1)?
            flag_res = th.cat((flag_win, 1 - flag_win), dim = 2)  #(bs,T-1,2)?
            pre_state_loss = self.predict_state_win.update_ps(state, flag_res, mask)
            self.pre_state_optimiser.zero_grad()
            pre_state_loss.backward()
            self.pre_state_optimiser.step()
        # ——————————————————————更新用于计算influence reward的预测obs网络的参数————————————————
        if self.args.is_use_influence:
            # predictor error
            actions_one_hot = batch["actions_onehot"][:, :-1]  # bs,t, n_agents, n_actions
            actions_one_hot_clone = actions_one_hot.clone()
            # inf predictor error
            eval_obs_diverge = []
            p_o_loss = th.tensor(0.0).to(loss.device)
            q_o_loss = th.tensor(0.0).to(loss.device)
            influence_kl_loss = th.tensor(0.0).to(loss.device)
            
            # influence reward B(infb):将policy变化作为影响
            for i in range(self.args.n_agents):
                # ①基于predict_with_action网络得：with ai情况下对应的 porobs_with_i
                # hidden_out_clone = hidden_out.clone().detach()
                actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)  # (bs,T-1,n_agent-1,n_actions) ,a[t][i]   得到当前 agent i 动作
                influence_inputs_with_i = th.cat((hidden_out_clone_without_i, actions_one_hot_i), dim=-1)  # (bs,T-1,n_agent-1,(rnn_hidden_dim+n_actions)  # 考虑了agent i 在时刻t的动作
                porobs_with_i = self.eval_predict_with_i(influence_inputs_with_i, pre_action=True)
                # ②实际选择的动作分布
                actions_one_hot_i_pre = th.cat((actions_one_hot_clone[:, :, :i], actions_one_hot_clone[:, :, i + 1:]), dim=2)
                # ③对 policy_Q 和 porobs_with_i 求JS散度
                even_probs = (actions_one_hot_i_pre + porobs_with_i) / 2 + epsilon
                actions_one_hot_i_pre_e, porobs_with_i_e = actions_one_hot_i_pre + epsilon, porobs_with_i + epsilon
                diverge_i = 0.5 * th.sum((actions_one_hot_i_pre_e * th.log(actions_one_hot_i_pre_e / even_probs + porobs_with_i_e * th.log(porobs_with_i_e / even_probs))), dim=-1)  # (bs,T-1,n_agent-1)
                diverge_i = th.mean(diverge_i, dim=-1).unsqueeze(-1)  # (bs,T-1,1)
                influence_kl_loss += th.sum(diverge_i)
            '''
            # influence reward C(infc):将policy变化作为影响(agent i影响的是下一个时刻的j的动作）
            for i in range(self.args.n_agents):
                # ①基于predict_with_action网络得：with ai情况下对应的 porobs_with_i
                # hidden_out_clone = hidden_out.clone().detach()
                actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)  # (bs,T-1,n_agent-1,n_actions) ,a[t][i]   得到当前 agent i 动作
                influence_inputs_with_i = th.cat((hidden_out_clone_without_i, actions_one_hot_i), dim=-1)  # (bs,T-1,n_agent-1,(rnn_hidden_dim+n_actions)  # 考虑了agent i 在时刻t的动作
                porobs_with_i = self.eval_predict_with_i(influence_inputs_with_i, pre_action=True)
                # ②实际选择的动作分布(应该是agent j在t+1时刻选择的动作)
                actions_one_hot_i_pre = th.cat((actions_one_hot_clone[:, :, :i], actions_one_hot_clone[:, :, i + 1:]), dim=2)
                # ③对 actions_one_hot_i_pre 和 porobs_with_i 求JS散度
                even_probs = (actions_one_hot_i_pre + porobs_with_i) / 2 + epsilon
                actions_one_hot_i_pre_e, porobs_with_i_e = actions_one_hot_i_pre + epsilon, porobs_with_i + epsilon
                diverge_i = 0.5 * th.sum((actions_one_hot_i_pre_e * th.log(actions_one_hot_i_pre_e / even_probs + porobs_with_i_e * th.log(porobs_with_i_e / even_probs))), dim=-1)  # (bs,T-1,n_agent-1)
                diverge_i = th.mean(diverge_i, dim=-1).unsqueeze(-1)  # (bs,T-1,1)
                influence_kl_loss += th.sum(diverge_i)
            '''
            '''
            # influence reward A(infa):将预测的下一时刻的观测变化作为influence reward（based on COIN论文）
            for i in range(self.args.n_agents):
                # 无i的actions
                actions_one_hot_i_pre = th.cat((actions_one_hot_clone[:, :, :i], actions_one_hot_clone[:, :, i + 1:]),dim=2)
                actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)
                # 有i的obs和无i的obs以及下时刻obs
                
                obs_next_i = th.cat((obs_next_clone[:, :, :i], obs_next_clone[:, :, i + 1:]), dim=2)

                influence_inputs_without_i = th.cat((obs_next_i, actions_one_hot_i_pre), dim=-1).detach()  # (bs,T-1,n_agent-1,obs_shape+n_action)  # 不考虑agent i 在时刻t的动作
                influence_inputs_with_i = th.cat((obs_next_i, actions_one_hot_i_pre, actions_one_hot_i),dim=-1).detach()  # (bs,T-1,n_agent-1,2*(obs_shape+n_actions))  # 考虑了agent i 在时刻t的动作

                p_o_i_loss = self.eval_predict_without_i.update(influence_inputs_without_i, obs_next_i, mask_next)
                q_o_i_loss = self.eval_predict_with_i.update(influence_inputs_with_i, obs_next_i, mask_next)
                p_o_loss += p_o_i_loss
                q_o_loss += q_o_i_loss
            '''
            # predictor_loss = p_o_loss + q_o_loss
            predictor_loss = influence_kl_loss
            # Optimise
            self.predictor_optimiser.zero_grad()
            predictor_loss.backward()
            self.predictor_optimiser.step()
        if self.args.is_use_influence and self.args.influence_exp_decay:
            if t_env - self.decay_stats_inf_t >= self.args.influence_exp_decay_cycle:
                if self.args.influence_exp_decay_rate <= 1.0:
                    if self.beta1 > self.args.influence_exp_decay_stop:
                        self.beta1 = self.beta1 * self.args.influence_exp_decay_rate

                    else:
                        self.beta1 = self.args.influence_exp_decay_stop
                else:
                    if self.beta1 < self.args.influence_exp_decay_stop:
                        self.beta1 = self.beta1 * self.args.influence_exp_decay_rate
                    else:
                        self.beta1 = self.args.influence_exp_decay_stop
                self.decay_stats_inf_t = t_env


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()

            if self.args.use_emdqn:
                self.logger.log_stat("e_m Q mean", (qec_input_new * mask).sum().item() /
                                     (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("em_ Q hit probability", episodic_q_hit_pro, t_env)
                self.logger.log_stat("emdqn_loss", emdqn_loss.item(), t_env)
                self.logger.log_stat("emdqn_curr_capacity", ec_buffer.ec_buffer.curr_capacity, t_env)
                self.logger.log_stat("emdqn_weight", self.args.emdqn_loss_weight, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            if self.args.is_use_influence:
                self.logger.log_stat("influence_mean", (influence_intrinsic_reward * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            if self.args.is_use_statewon:
                self.logger.log_stat("pre_state_loss", pre_state_loss, t_env)
                self.logger.log_stat("state_won",(won_lose[:,:,0].unsqueeze(-1) * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)

            # if value_propagation_flag:
            #     self.logger.log_stat("emdqn_vp_updata_rate", updata_rate, t_env)

            self.log_stats_t = t_env

        if self.args.is_prioritized_buffer:
            return masked_td_error ** 2, mask




    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False,ec_buffer=None):
        # 调用 self.vdn_learner.train 方法进行训练，并获取【内在奖励】
        intrinsic_rewards = self.vdn_learner.train(batch, t_env, episode_num,save_buffer=False, imac=self.mac, timac=self.target_mac)
        # 如果使用优先级缓冲区
        # sub_train后需要获取掩码误差（masked_td_error）和掩码（mask）
        if self.args.is_prioritized_buffer:
            # masked_td_error和mask是用于更新优先级缓冲区的关键数据
            masked_td_error, mask = self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)
        # 否则，只sub_train
        else:
            self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)

        if hasattr(self.args, 'save_buffer') and self.args.save_buffer:
            # 检查是否达到保存周期
            if self.buffer.episodes_in_buffer - self.save_buffer_cnt >= self.args.save_buffer_cycle:
                # 如果可以从缓冲区采样，进行训练并保存缓冲区数据
                if self.buffer.can_sample(self.args.save_buffer_cycle):
                    batch_tmp=self.buffer.sample(self.args.save_buffer_cycle, newest=True)
                    intrinsic_rewards_tmp=self.vdn_learner.train(batch_tmp, t_env, episode_num, save_buffer=True,
                                                                   imac=self.mac, timac=self.target_mac)
                    self.sub_train(batch_tmp, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards_tmp,
                        show_demo=show_demo, save_data=save_data, show_v=show_v, save_buffer=True)
                else:
                    print('**' * 20, self.buffer.episodes_in_buffer, self.save_buffer_cnt)
        # 如果达到目标网络更新的间隔
        if self.args.use_emdqn and episode_num == 50:
            # 初次构造二叉树
            print('[t_env={}]init kdtree......'.format(t_env))
            self.start_value_propagation = True
            ec_buffer.update_kdtree()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(ec_buffer,t_env)  # 更新ec_buffer中的二叉查找树、更新目标网络
            self.last_target_update_episode = episode_num  # 更新最后一次更新目标网络的回合数
        # 如果使用优先级缓冲区，计算并返回误差结果
        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res  # 为td_error

    def _update_targets(self,ec_buffer,t_env):
        if self.args.use_emdqn:
            ec_buffer.update_kdtree()
            # —————————————————————定期对episodic memory进行value propagation——————————————————
            if self.start_value_propagation and self.args.is_value_propagation and self.args.use_emdqn and t_env < self.args.value_propagation_max_t: # and t_env - self.last_update_memory_t > self.args.update_memory_internal:
                updata_rate = ec_buffer.value_propagation()
                print('[t_env={}] value propagation...updata_rate={}'.format(t_env, updata_rate))
                # self.last_update_memory_t = t_env
                # if self.last_update_memory_t > 1e4:
                #     self.args.update_memory_internal = 1e4
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.args.is_use_influence:
            self.target_predict_with_i.load_state_dict(self.eval_predict_with_i.state_dict())
            # self.target_predict_without_i.load_state_dict(self.eval_predict_without_i.state_dict())

        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        self.vdn_learner.cuda()
        to_cuda(self.eval_predict_with_i, self.args.device)
        to_cuda(self.target_predict_with_i, self.args.device)
        to_cuda(self.predict_state_win, self.args.device)
        
        # to_cuda(self.eval_predict_without_i, self.args.device)
        # to_cuda(self.target_predict_without_i, self.args.device)
        if self.mixer is not None:
            to_cuda(self.mixer, self.args.device)
            to_cuda(self.target_mixer, self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
