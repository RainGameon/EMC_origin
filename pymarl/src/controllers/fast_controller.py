from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class FastMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        # 表示每个智能体在当前时间步可执行的动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if hasattr(self.args, 'use_individual_Q') and self.args.use_individual_Q:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # 利用动作选择器action_selector从智能体的输出和可用动作中选择动作
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    # 计算智能体在给定批次数据和时间步的情况下的动作输出
    def forward(self, ep_batch, t, test_mode=False, batch_inf=False): # batch_inf=True
        # 构造智能体输入，obs
        agent_inputs = self._build_inputs(ep_batch, t, batch_inf) #[bs*n_agent,T,_]
        epi_len = t if batch_inf else 1 # epi_len表示批次无限时的时间步数，如果不是批次无限，则为1
        # 所有时间步可用的动作
        avail_actions = ep_batch["avail_actions"][:, :t] if batch_inf else ep_batch["avail_actions"][:, t:t+1]
        # 如果存在参数use_individual_Q并且其值为True
        if hasattr(self.args, 'use_individual_Q') and self.args.use_individual_Q:
            # 计算智能体的【输出（agent_outs）、隐藏状态（self.hidden_states）和个体Q值（individual_Q）】
            # agent_outs形状：（bs*n_agent, T, n_actions)
            agent_outs, self.hidden_states, individual_Q = self.agent(agent_inputs, self.hidden_states)
        else: # 否则，只计算智能体的【输出和隐藏状态】
            agent_outs, self.hidden_states, all_hidden = self.agent(agent_inputs, self.hidden_states)

        # 如果智能体的输出类型是"pi_logits"（策略的logits）
        if self.agent_output_type == "pi_logits": # 对智能体的输出进行softmax操作
            # 如果在softmax之前需要mask不可用动作
            if getattr(self.args, "mask_before_softmax", True):
                # 为不可用动作设置非常小的logits值以减少其在softmax中的影响
                reshaped_avail_actions = avail_actions.transpose(1, 2).reshape(ep_batch.batch_size * self.n_agents, epi_len, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10  # 通过将不可用动作的logits设置为很小的负数（-1e10），可以最小化它们对softmax操作的影响
            # 对智能体输出应用softmax
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1) #在最后一维操作
            # 如果不处于测试模式，应用epsilon贪心策略
            if not test_mode: # 则根据epsilon值对智能体的输出进行epsilon-greedy策略的调整
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # 以epsilon概率均匀选择一个可用动作
                    epsilon_action_num = reshaped_avail_actions.sum(dim=-1, keepdim=True).float()
                #
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # 将不可用动作的概率设置为0
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        # 如果启用了独立Q值功能
        if hasattr(self.args, 'use_individual_Q') and self.args.use_individual_Q:
            # 返回处理后的智能体输出和独立Q值 [bs,n_agents,T*n_actions]
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), individual_Q.view(ep_batch.batch_size, self.n_agents, -1)
        else:
            if batch_inf:
                # print("SMY————agent_outs.shape = {}".format(agent_outs.shape))    #torch.Size([bs*n_agent, T, n_actions])
                return agent_outs.view(ep_batch.batch_size, self.n_agents, epi_len, -1).transpose(1, 2), all_hidden.view(ep_batch.batch_size, self.n_agents, epi_len, -1).transpose(1, 2)
            else:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)  #, self.hidden_states.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_hid(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t+1, batch_inf=True)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), self.hidden_states.view(ep_batch.batch_size, self.n_agents, -1)



    def init_hidden(self, batch_size):
        # self.agent.init_hidden(): (1, self.args.rnn_hidden_dim)
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # (bs,n_agent,rnn_hidden_dim)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def to(self, *args, **kwargs):
        self.agent.to(*args, **kwargs)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t, batch_inf): # 与obs有关
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        if batch_inf: # true
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, :t])  # 【b，t-1，n_agent,v】
            # print('SMY: inputs.shape={}，hidden_out.shape={}'.format(inputs.shape, hidden_out.shape))

            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))  # 上一个时间步动作（t=0时为0）
                else:
                    # inputs.append(batch["actions_onehot"][:, t - 1])
                    last_actions = th.zeros_like(batch["actions_onehot"][:, :t]) # 上一个时间步动作（t=0时为0，其余=t-1时刻onehot）
                    last_actions[:, 1:] = batch["actions_onehot"][:, :t-1]
                    inputs.append(last_actions)
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t, -1, -1))

            inputs = th.cat([x.transpose(1, 2).reshape(bs*self.n_agents, t, -1) for x in inputs], dim=2)
            # print('SMY-mac: inputs.shape={}，hidden_out.shape={}'.format(inputs.shape, hidden_out.shape))

            return inputs # input:[bs*n,T,num_feat]
        else:
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, t])  # b1av
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t-1])
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            inputs = th.cat([x.reshape(bs*self.n_agents, 1, -1) for x in inputs], dim=2)
            return inputs

    def _get_input_shape(self, scheme): #输入到agent进行Q值计算的变量形状为：观测obs形状+agent数量
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id: # true
            input_shape += self.n_agents

        return input_shape
