import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
from modules.w_predictor.predict_net import Predict_Network_OS, Predict_Network
import torch.nn.functional as F
import torch as th
# from torch.optim import RMSprop
from torch.optim import RMSprop, Adam
from utils.torch_utils import to_cuda
import numpy as np
from .vdn_Qlearner import vdn_QLearner
import os


class QPLEX_curiosity_vdn_Learner:
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
        self.vdn_learner = vdn_QLearner(mac, scheme, logger, args)
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':  # this
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        #————————————————————————————————————与influence reward有关——————————————————————————————————————
        '''##根据trajectory预测下时刻obs和state
        # self.eval_predict_os = Predict_Network_OS((args.rnn_hidden_dim + args.n_actions) * self.args.n_agents,
        #                                           args.predict_net_dim,
        #                                           self.args.obs_shape, self.args.state_shape, self.args.n_agents)
        # self.target_predict_os = copy.deepcopy(self.eval_predict_os)
        # 根据obs和action预测下时刻obs
        '''
        self.eval_predict_with_i = Predict_Network((args.obs_shape + args.n_actions)*2 , args.predict_net_dim, args.obs_shape)
        self.target_predict_with_i = copy.deepcopy(self.eval_predict_with_i)
        self.eval_predict_without_i = Predict_Network(args.obs_shape + args.n_actions , args.predict_net_dim, args.obs_shape)
        self.target_predict_without_i = copy.deepcopy(self.eval_predict_without_i)
        # 与influence reward有关参数与Adam
        self.predictor_params = list(self.eval_predict_with_i.parameters())
        self.predictor_params += list(self.eval_predict_without_i.parameters())
        self.predictor_optimiser = Adam(params=self.predictor_params, lr=args.predictor_lr)
        self.decay_stats_cur_t = 0
        self.decay_stats_inf_t = 0
        self.beta1 = self.args.beta1
        self.beta2 = self.args.beta2
        self.beta = self.args.beta
        #————————————————————————————————————————————止——————————————————————————————————————————————————————
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_update_memory_t = 0
        self.save_buffer_cnt = 0

        self.n_actions = self.args.n_actions

    def update_memory(self, memory_buffer, mac, mixer, updata_batchsize):  # value propagation
        # 遍历所有trajectory，去更新
        for i in range(memory_buffer.episodes_in_buffer - updata_batchsize):
            batch = memory_buffer[i:i + updata_batchsize]
            # _____________________________________________
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()  # 只有终止时间步之前的时间步对应的元素为1，其他时间步对应的元素为0,则terminated全为0
            mask = batch["filled"][:, :-1].float()  # 获取所有时间步的填充信息
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # mask终止状态之前的时间步对应的元素为填充信息，终止状态后为0
            avail_actions = batch["avail_actions"]
            actions_onehot = batch["actions_onehot"][:, :-1]
            # ————————————————————————————————————-计算最大Q1值与对应的action——————————————————————————————————————
            mac.init_hidden(batch.batch_size)
            mac_out = mac.forward(batch, batch.max_seq_length, batch_inf=True)  # [bs,T,n_agents,n_actions]
            x_mac_out = mac_out.clone().detach()  # [bs,T,n_agent,n_actions]
            x_mac_out[avail_actions == 0] = -9999999  # 并将不可行动作的Q值设置为一个极小值，以便在下一步找到最大动作

            # 【a*: 最大Q值:Qi(s,a*)】找到每个时间步的最大动作的Q值和对应的动作索引
            max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3, keepdim=True)  # 都是【bs,T-1,n_agent，1】不要最后一个时刻
            max_action_qvals = max_action_qvals.squeeze(3)  # [bs,T-1,n_agent]
            # 最大Q1值对应动作hot
            max_actions_onehot = to_cuda(th.zeros(max_action_index.squeeze(3).shape + (self.n_actions,)),
                                         self.args.device)
            max_actions_onehot = max_actions_onehot.scatter_(3, max_action_index, 1)  # [bs,T-1,n_agents,n_actions]


            self.target_mac.init_hidden(batch.batch_size)
            target_mac_out = self.target_mac.forward(batch, batch.max_seq_length,
                                                     batch_inf=True)  # [bs,T,n_agents,n_actions]
            target_mac_out[avail_actions == 0] = -9999999  # 屏蔽不可行动作
            if self.args.double_q:  # true
                target_out_detach = target_mac_out.clone().detach()
                target_out_detach[avail_actions == 0] = -9999999
                # 【Q_tgt(s,a**)】a**是在Qtarget上找的最好动作
                target_max_actions_qvals, target_max_actions_index = target_out_detach[:, :-1].max(dim=3, keepdim=True) # [bs,T-1,n_agents,1]
                # 【Q_tgt(s,a*)】a*是在Q上找的最好动作
                target_max_qvals = th.gather(target_out_detach[:, :-1], 3, index=max_action_index).squeeze(3)     # [bs,T-1,n_agents]
                # a** hot编码
                target_max_actions_onehot = to_cuda(
                    th.zeros(max_action_index.squeeze(3).shape + (self.n_actions,)), self.args.device)
                target_max_actions_onehot = target_max_actions_onehot.scatter_(3, target_max_actions_index,
                                                                               1)  # [bs,T-1,n_agents,n_actions]
                # print('SUNMENGYAO___________________target_max_actions_onehot.shape={}'.format(target_max_actions_onehot.shape))

            else:
                # Calculate the Q-Values necessary for the target
                target_mac_out = []
                self.target_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)
                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
                target_max_qvals = target_mac_out.max(dim=3)[0]

            # 应用混合器（mixer）计算Qtot_tgt
            if mixer is not None:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, :-1], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, :-1],
                                                             actions=target_max_actions_onehot,
                                                             max_q_i=target_max_actions_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv  # 【bs, T-1, 1】
                else:
                    target_chosen = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_max_qvals, batch["state"][:, 1:],
                                                       actions=target_max_actions_onehot,
                                                       max_q_i=target_max_actions_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

            # ____________________________________ 基于target_max_qvals来计算R_______________________________
            rtn = th.zeros((int(batch.batch_size), batch.max_seq_length - 1, batch.max_seq_length - 1)).to(batch.device)  # 【bs,T-1,T-1】
            # print('SUNMENGYAO___________________rtn_1.shape={}'.format(rtn_1.shape))

            # approximate_qs = th.cat([max_qvals, target_max_qvals], dim=2)  # 【bs,T-1,2】
            approximate_qs = th.transpose(target_max_qvals, 1, 2)            # 【bs,1,T-1】
            approximate_qs = th.cat([(th.zeros((batch.batch_size, 1, 1))).to(approximate_qs.device), approximate_qs],dim=2)  # 【bs,1,T】
            # print('SUNMENGYAO___________________ approximate_qs.shape={}'.format(approximate_qs.shape))

            # 遍历每一步，计算1步return，即h=0
            rewards = batch["reward"][:, :-1]  # terminated, [bs,T-1,1]
            for i in range(batch.max_seq_length - 1):
                rtns = rewards[:, i] + self.args.gamma * (1 - terminated[:, i]) * (approximate_qs[:, :, i])  # [1, 1]
                # print('SUNMENGYAO___________________approximate_qs={} rtns={}'.format(approximate_qs[:,:,i], rtns))
                rtn[:, i, 0] = rtns[:, 0]

            # 遍历每一步，计算1到T步return（公式7）
            for i in range(batch.max_seq_length - 1):
                rtn[:, i, 1:] = rewards[:, i] + self.args.gamma * rtn[:, i - 1, 1:]
            rtn = rtn.detach().cpu() #.numpy() #[bs,T-1,T-1]
            # 计算R预估最高回报 公式78
            res_rtn = [np.array(th.gather(rtn[:, i, :batch.max_seq_length - 1], 1, th.argmax(rtn[:, i, :batch.max_seq_length - 1], axis=1).unsqueeze(1))) for i in range(batch.max_seq_length - 1)]
            # print('SUNMENGYAO___________________ res_rtn={}'.format(np.array(res_rtn).shape)) # (T-1,  bs, 1) res_rtn=(150,  128, 1)
            res_rtn = res_rtn
            one_step_q = np.array([rtn[:, :, 0].numpy()]).transpose(2, 1, 0)  # [T-1,bs,1]  # 一步return
            # 选择计算得到的R值和一步TD return的最大值作为预估最高回报q_values
            return_H_hat = np.maximum(np.array(res_rtn), one_step_q).transpose(1, 0, 2)
            return_H_hat = th.tensor(return_H_hat).to(batch.device)  # [bs,T-1,1]
            return_H = batch["return_H"][:, :-1]                     # [bs,T-1,1]
            # print('SUNMENGYAO___________________ return_H_hat={}'.format(return_H_hat.shape))
            # 更新这条trajectory上估计的最高回报
            batch["return_H"][:, :-1] = th.max(return_H_hat, return_H)

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,intrinsic_rewards,
                  show_demo=False, save_data=None, show_v=False, save_buffer=False,ec_buffer=None,replay_buffer=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float() #  [bs, T-1, 1] 只有终止时间步之前的时间步对应的元素为1，其他时间步对应的元素为0,则terminated全为0
        mask = batch["filled"][:, :-1].float()                # 获取所有时间步的填充信息
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # mask终止状态之前的时间步对应的元素为填充信息，终止状态后为0
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        # EDTI所需
        obs = batch["obs"][:, :-1]
        obs_next = batch["obs"][:, 1:]
        state = batch["state"][:, :-1] #3s5z:[bs, T, 216])
        state_next = batch["state"][:, 1:]
        mask_next = batch["filled"][:, 1:].float()  #  [bs, T-1, 1] 
        # print('SMY: state.shape={}'.format(state.shape))

        mask_next[:, 1:] = mask_next[:, 1:] * (1 - terminated[:, 1:])
        mask_next_obs = mask_next.repeat(1, 1, self.args.obs_shape * self.args.n_agents)  # [bs, T-1, 1024]

        mask_next_state = mask_next.repeat(1, 1, self.args.state_shape)
        bs = obs.shape[0]
        obs_next_flatten = obs_next.view(bs, -1, self.args.obs_shape * self.args.n_agents) #[bs,T-1,obs_shape,n_agent]?
        obs_next_clone = obs_next.clone()
        obs_clone = obs.clone()
        # 【Qi(s,a)】
        mac.init_hidden(batch.batch_size)
        mac_out, all_hidden = mac.forward(batch, batch.max_seq_length, batch_inf=True) # [bs,T,n_agents,n_actions] T=max_seq_length
        # print('SMY: mac_out.shape={}, bach.max_seq_length={}'.format(mac_out.shape,batch.max_seq_length))

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

        # 【实际选的动作对应的Qi(s,a)】
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  #  [bs,T-1,n_agent] t:0-T-1
        x_mac_out = mac_out.clone().detach()        # [bs,T,n_agents,n_actions]
        x_mac_out[avail_actions == 0] = -9999999    # 将不可行动作的Q值设置为一个极小值 [bs,T,n_agents,n_actions]

        # 【Qi值最大对应的动作a*:Qi(s,a*)】
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)   # [bs,T-1,n_agents]
        max_action_index = max_action_index.detach().unsqueeze(3)           # [bs,T-1,n_agents,1]

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()

        # 【Q_target(s',a)】
        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[0][:, 1:, ...] # [bs,T-1,n_agents,n_actions]
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999                                                 # [bs,T-1,n_agents,n_actions]

        # 计算目标网络的最大Q值
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()                                               # [bs,T,n_agents,n_actions]
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]                     # [bs,T-1,n_agents,1],t=1~
            # 【Q_target(s',a*】 从t=1开始
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)               # [bs,T-1,n_agent] Q(s',a*)
            # 【Q_target(s',a**】从t=1开始，计算目标网络的最大Q值
            target_max_qvals = target_mac_out.max(dim=3)[0]     # [bs,T-1,n_agents,1]

            cur_max_actions_onehot = to_cuda(th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)), self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)  #[bs,T-1,n_agent,n_actions]
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, _ = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]
        
        # ————————————————————————计算influence-based intrinsic reward——————————————————————
        if self.args.is_use_influence:
            with th.no_grad():
                actions_one_hot = batch["actions_onehot"][:, :-1]  # bs,t, n_agents, n_actions
                actions_one_hot_clone = actions_one_hot.clone()  # （bs,T-1,n_agent,n_actions)
     
                obs_diverge = []  # 存放当前agent动作对其他agent观测变化对影响
                for i in range(self.args.n_agents):
                    actions_one_hot_i_pre = th.cat((actions_one_hot_clone[:, :, :i], actions_one_hot_clone[:, :, i + 1:]), dim=2)  # (bs,T-1,n_agent-1,n_actions) ,a[t][-i]  得到除了当前 agent i 其他智能体对动作
                    actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1,  1)  # (bs,T-1,n_agent-1,n_actions) ,a[t][i]   得到当前 agent i 动作
                    obs_i = obs_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)
                    obs_next_i = th.cat((obs_next_clone[:, :, :i], obs_next_clone[:, :, i + 1:]),  dim=2)  # (bs,T-1,n_agent-1,obs_dim)    o[t+1][-i]得到除了当前 agent i 其他智能体下时刻的观测
    
                    obs_without_i = th.cat((obs_clone[:, :, :i], obs_clone[:, :, i + 1:]), dim=2)
                    obs_with_i = obs_clone
                    # 构造输入（obs+action）
                    influence_inputs_without_i = th.cat((obs_without_i, actions_one_hot_i_pre), dim=-1)  # (bs,T-1,n_agent-1,obs_shape+n_action)  # 不考虑agent i 在时刻t的动作
                    influence_inputs_with_i = th.cat( [obs_without_i, obs_i, actions_one_hot_i_pre, actions_one_hot_i],  dim=-1)  # (bs,T-1,n_agent-1,2*(obs_shape+n_actions))  # 考虑了agent i 在时刻t的动作
    
                    log_p_o_i = self.target_predict_without_i.get_log_pi(influence_inputs_without_i,
                                                                         obs_next_i)  # (bs,T-1,n_agent-1)
    
                    log_q_o_i = self.target_predict_with_i.get_log_pi(influence_inputs_with_i,
                                                                      obs_next_i)  # (bs,T-1,n_agent-1)
                    # 计算agent i获得的influence:
                    obs_diverge_i = self.args.beta * th.abs(th.sum(log_q_o_i, -1, keepdim=True) - th.sum(log_p_o_i, -1, keepdim=True))  # (bs, T-1, 1)
                    # obs_diverge_i = self.args.beta * th.sum(1 - log_p_o_i / log_q_o_i, -1, keepdim=True) # (bs,T-1,1)
                    # obs_diverge_i = self.args.beta * th.sum(1 - self.args.beta1*(log_p_o_i / log_q_o_i), -1, keepdim=True) # (bs,T-1,1)
    
                    obs_diverge.append(obs_diverge_i * mask_next * self.args.beta1)
                influence_intrinsic_reward = th.stack(obs_diverge, dim=2).squeeze(-1)  # bs, T-1, n_agents
                # print(influence_intrinsic_reward.shape)

        # ——————————————————————————————————计算Qtot————————————————————————————————————————
        if mixer is not None:
            # ①【Qtot(s,a)】
            if self.args.mixer == "dmaq_qatten":
                if self.args.is_use_influence:
                    ans_chosen, q_attend_regs, head_entropies = mixer(chosen_action_qvals-influence_intrinsic_reward, batch["state"][:, :-1], is_v=True)                           # 计算V(s)
                    ans_adv, _, _ = mixer(chosen_action_qvals-influence_intrinsic_reward, batch["state"][:, :-1], actions=actions_onehot,  # 计算A(s,a)
                                          max_q_i=max_action_qvals, is_v=False)
                    chosen_action_qvals = ans_chosen + ans_adv  # 【bs, T-1, 1】 torch.Size([32, 71, 1])
                else:
                    ans_chosen, q_attend_regs, head_entropies = mixer(chosen_action_qvals,batch["state"][:, :-1], is_v=True)  # 计算V(s)
                    ans_adv, _, _ = mixer(chosen_action_qvals , batch["state"][:, :-1],
                                          actions=actions_onehot,  # 计算A(s,a)
                                          max_q_i=max_action_qvals, is_v=False)
                    chosen_action_qvals = ans_chosen + ans_adv  # 【bs, T-1, 1】 torch.Size([32, 71, 1])
            else:
                ans_chosen = mixer(chosen_action_qvals-influence_intrinsic_reward, batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals-influence_intrinsic_reward, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            # ②计算Qtot2
            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    # a. 【Qtot_tgt(s',a*)】 用于计算y
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv           # 【bs, T-1, 1】
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)
        # ———————————————————————————基于episodic memory计算预计最高回报——————————————————————
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
                    # # 通过 ec_buffer 查找 z 对应的 Q 值
                    # q = ec_buffer.peek(z, None, modify=False)  # H(Φ(s[j]))
                    # 通过 ec_buffer 查找 z 对应对 Q值和一步TDreturn并取最大（即进行propagation）
                    q = ec_buffer.peek(z, None, modify=False, propagate=True)
                    if q != None: # 如果找到了相应的 Q 值，则更新 qec_tmp 中对应位置的数值
                        qec_tmp[j - 1] = self.args.gamma * q + rewards[i][j - 1] # H(Φ(s[j-1]),a(j-1)) = r(j-1) + gamma * H
                        ec_buffer.qecwatch.append(q)
                        ec_buffer.qec_found += 1  # 记录成功找到 Q 值的次数
                qec_input_new.append(qec_tmp)
            qec_input_new = th.stack(qec_input_new, dim=0)
            episodic_q_hit_pro = 1.0 * ec_buffer.qec_found / self.args.batch_size / ec_buffer.update_counter / batch.max_seq_length
        # ——————————————————————————————————计算损失————————————————————————————————————————
        targets = intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        # y
        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return
        # 第一部分损失：Td-error=Qtot-y
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 第二部分基于memory的损失：
        if self.args.use_emdqn:
            # print('SMY: return_H.shape={}, chosen_action_qvals.shape={}'.format(batch["return_H"][:,:-1].shape, chosen_action_qvals.shape))
            # emdqn_td_error = batch["return_H"][:,:-1].detach() - chosen_action_qvals  # 改为用return❗️[bs, T-1, 1]
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
        masked_td_error = td_error * mask
        # 总损失函数
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

        #——————————————————————————————————计算KL 散度正则化项,帮助确保潜变量具有良好的分布特性 然后更新network————————————————————————————
        # kl for vae
        # q_normal = th.normal(0, 1, size=hidden_out.size()).to(hidden_out.device)
        # Kl_hidden_out = 0.0001 * F.kl_div(th.log_softmax(hidden_out, -1), th.softmax(q_normal, -1), reduction='mean').sum()
        # loss += Kl_hidden_out

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()
        #——————————————————————更新用于计算influence reward的预测obs网络的参数————————————————
        if self.args.is_use_influence:
            # predictor error
            actions_one_hot = batch["actions_onehot"][:, :-1]  # bs,t, n_agents, n_actions
            actions_one_hot_clone = actions_one_hot.clone()
        # # os predictor error
        # os_predict_input = th.cat([hidden_out_clone, actions_one_hot_clone], -1).view(bs, -1, (
        #             self.args.rnn_hidden_dim + self.args.n_actions) * self.args.n_agents).detach()
        # eval_predict_obs_next, eval_predict_state_next = self.eval_predict_os(os_predict_input)
        # predictor_os_diff = th.sum((((eval_predict_obs_next - obs_next_flatten) * mask_next_obs) ** 2), dim=-1,
        #                            keepdim=True) + \
        #                     th.sum(((eval_predict_state_next - state_next) * mask_next_state) ** 2, dim=-1,
        #                            keepdim=True)  # bs,t,1
        # predictor_os_loss = predictor_os_diff.sum() / mask_next.sum()

        # inf predictor error
            eval_obs_diverge = []
            p_o_loss = th.tensor(0.0).to(loss.device)
            q_o_loss = th.tensor(0.0).to(loss.device)
            for i in range(self.args.n_agents):
                # 有i的actions和无i的actions
                actions_one_hot_i_pre = th.cat((actions_one_hot_clone[:, :, :i], actions_one_hot_clone[:, :, i + 1:]), dim=2)
                actions_one_hot_i = actions_one_hot_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)
                # 有i的obs和无i的obs以及下时刻obs
                obs_i = obs_clone[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)
                obs_without_i = th.cat((obs_clone[:, :, :i], obs_clone[:, :, i + 1:]), dim=2)
                obs_next_i = th.cat((obs_next_clone[:, :, :i], obs_next_clone[:, :, i + 1:]), dim=2)
                
                influence_inputs_without_i = th.cat((obs_without_i, actions_one_hot_i_pre), dim=-1).detach()  # (bs,T-1,n_agent-1,obs_shape+n_action)  # 不考虑agent i 在时刻t的动作
                influence_inputs_with_i = th.cat([obs_without_i, obs_i, actions_one_hot_i_pre, actions_one_hot_i], dim=-1).detach()  # (bs,T-1,n_agent-1,2*(obs_shape+n_actions))  # 考虑了agent i 在时刻t的动作
    
                p_o_i_loss = self.eval_predict_without_i.update(influence_inputs_without_i, obs_next_i, mask_next)
                q_o_i_loss = self.eval_predict_with_i.update(influence_inputs_with_i, obs_next_i, mask_next)
                p_o_loss += p_o_i_loss
                q_o_loss += q_o_i_loss
    
            predictor_loss = p_o_loss + q_o_loss
            # Optimise
            self.predictor_optimiser.zero_grad()
            predictor_loss.backward()
            # predictor_loss_grad_norm = th.nn.utils.clip_grad_norm_(self.predictor_params, self.args.grad_norm_clip)
            self.predictor_optimiser.step()
        # —————————————————————定期对episodic memory进行value propagation——————————————————
        if self.args.use_emdqn and t_env < 1e6 and t_env - self.last_update_memory_t > self.args.update_memory_internal:
            print('[t_env={}] value propagation...'.format(t_env))
            ec_buffer.value_propagation()
            self.last_update_memory_t = t_env
            if self.last_update_memory_t > 1e4:
                self.args.update_memory_internal = 1e4
                if self.last_update_memory_t > 1e5:
                    self.args.update_memory_internal = 3e5
                    if self.last_update_memory_t > 1e6:
                        self.args.update_memory_internal = 1e6
            
            
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
        '''
        if self.args.curiosity_exp_decay:
            if t_env - self.decay_stats_cur_t >= self.args.curiosity_exp_decay_cycle:
                if self.args.curiosity_exp_decay_rate <= 1.0:
                    if self.beta2 > self.args.curiosity_exp_decay_stop:
                        self.beta2 = self.beta2 * self.args.curiosity_exp_decay_rate
                    else:
                        self.beta2 = self.args.curiosity_exp_decay_stop
                else:
                    if self.beta2 < self.args.curiosity_exp_decay_stop:
                        self.beta2 = self.beta2 * self.args.curiosity_exp_decay_rate
                    else:
                        self.beta2 = self.args.curiosity_exp_decay_stop

                self.decay_stats_cur_t = t_env
        '''
        #——————————————————————————————————————————————log————————————————————————————————————————————————————
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            if self.args.use_emdqn:
                # self.logger.log_stat("e_m Q mean", (qec_input_new * mask).sum().item() /
                #                      (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("em_ Q hit probability", episodic_q_hit_pro, t_env)
                self.logger.log_stat("emdqn_loss", emdqn_loss.item(), t_env)
                # self.logger.log_stat("emdqn_curr_capacity", ec_buffer.ec_buffer.curr_capacity, t_env)
                self.logger.log_stat("emdqn_weight", self.args.emdqn_loss_weight, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            if self.args.is_use_influence:
                self.logger.log_stat("influence_mean", (influence_intrinsic_reward * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        if self.args.is_prioritized_buffer:
            return masked_td_error ** 2, mask


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False,ec_buffer=None,replay_buffer=None):
        # 调用 self.vdn_learner.train 方法进行训练，并获取【内在奖励】
        intrinsic_rewards = self.vdn_learner.train(batch, t_env, episode_num,save_buffer=False, imac=self.mac, timac=self.target_mac)
        # 如果使用优先级缓冲区：sub_train后需要获取掩码误差（masked_td_error）和掩码（mask）
        if self.args.is_prioritized_buffer: # false
            # masked_td_error和mask是用于更新优先级缓冲区的关键数据
            masked_td_error, mask = self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)
        # 否则，只sub_train
        else:
            self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer,replay_buffer=replay_buffer)

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
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(ec_buffer)  # 更新ec_buffer中的二叉查找树、更新目标网络
            self.last_target_update_episode = episode_num  # 更新最后一次更新目标网络的回合数
        # 如果使用优先级缓冲区，计算并返回误差结果
        if self.args.is_prioritized_buffer: # false
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res  # 为td_error

    def _update_targets(self,ec_buffer):
        if self.args.use_emdqn:
            ec_buffer.update_kdtree()
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.target_predict_os.load_state_dict(self.eval_predict_os.state_dict())
        if self.args.is_use_influence:
            self.target_predict_with_i.load_state_dict(self.eval_predict_with_i.state_dict())
            self.target_predict_without_i.load_state_dict(self.eval_predict_without_i.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        self.vdn_learner.cuda()
        to_cuda(self.eval_predict_with_i, self.args.device)
        to_cuda(self.target_predict_with_i, self.args.device)
        to_cuda(self.eval_predict_without_i, self.args.device)
        to_cuda(self.target_predict_without_i, self.args.device)
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
