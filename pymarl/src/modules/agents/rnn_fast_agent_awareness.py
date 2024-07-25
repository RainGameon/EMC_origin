import torch.nn as nn
import torch.nn.functional as F


class RNNFastAgentAwareness(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNFastAgentAwareness, self).__init__()
        self.args = args
        self.awareness_dim = args.awareness_dim
        self.var_floor = args.var_floor

        activation_func = nn.LeakyReLU()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True,
        )

        self.awareness_encoder = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.nn_hidden_dim),
                                               nn.BatchNorm1d(args.nn_hidden_dim),
                                               activation_func,
                                               nn.Linear(args.nn_hidden_dim, args.n_agents * args.awareness_dim * 2))
        self.infer_net = nn.Sequential(nn.Linear(2 * args.rnn_hidden_dim, args.nn_hidden_dim),
                                       nn.BatchNorm1d(args.nn_hidden_dim),
                                       activation_func,
                                       nn.Linear(args.nn_hidden_dim, args.awareness_dim * 2))

        self.multi_head_attention = MultiHeadAttention(n_head=4, d_model=args.awareness_dim, d_k=32, d_v=32)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.n_agents * args.awareness_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state): # input:[bs*n_agent?,T,num_feat]
        bs = inputs.shape[0]        # bs*n_agent?
        epi_len = inputs.shape[1]
        num_feat = inputs.shape[2]
        inputs = inputs.reshape(bs * epi_len, num_feat)  #(bs*n_agent*T, num_feat)

        x = F.relu(self.fc1(inputs))                            #(bs*n_agent*T, rnn_hidden_dim)
        x = x.reshape(bs, epi_len, self.args.rnn_hidden_dim)    #(bs*n_agent, T, rnn_hidden_dim)
        h_in = hidden_state.reshape(1, bs, self.args.rnn_hidden_dim).contiguous() # (1, bs*n_agent,rnn_hidden_dim)
        x, h = self.rnn(x, h_in)  # x(b,T,N,R)
        awareness_input = x.reshape(bs * epi_len, self.args.rnn_hidden_dim) #(bNT,R)

        awareness_params = self.awareness_encoder(awareness_input)  # (BNT, 2 * N * A)
        latent_size = self.n_agents * self.awareness_dim

        awareness_mean = awareness_params[:, :latent_size].reshape(-1,self.n_agents, epi_len, latent_size)  # (B,N, T, NA)
        awareness_var = th.clamp(th.exp(awareness_params[:, latent_size:]), min=self.var_floor).reshape(-1,self.n_agents, epi_len,  latent_size)  # (B,N, T, NA)

        awareness_dist = D.Normal(awareness_mean, awareness_var ** 0.5)
        awareness = awareness_dist.rsample().view(-1, latent_size) #(BNT,NA)

        kld = th.zeros(bs, 1).to(self.args.device)
        x_detach = x.view(-1,  self.n_agents, epi_len, self.rnn_hidden_dim).detach() #(B,N,T,R)
        if not test_mode:  # 训练时候
            infer_input = th.zeros(self.n_agents, bs*epi_len, 2 * self.rnn_hidden_dim).to(self.args.device) #(N,BNT,2R)
            for agent_i in range(self.n_agents):
                x_detach_i = x_detach[:, agent_i:agent_i + 1].repeat(1, self.n_agents, 1,1) #(B,N,T,R)
                infer_input[agent_i, :, :] = th.cat([x_detach_i, x_detach], dim=-1).view(-1, 2 * self.rnn_hidden_dim)  #(BNT,2R)
            infer_input = infer_input.view(self.n_agents * bs * epi_len, 2 * self.rnn_hidden_dim)  # (NBNT, 2R)

            infer_params = self.infer_net(infer_input) # (NBNT, 2A)
            infer_means = infer_params[:, :self.awareness_dim].reshape(-1,self.n_agents, epi_len, latent_size)  # (B,N, T, N * A)
            infer_vars = th.clamp(th.exp(infer_params[:, self.awareness_dim:]),
                                  min=self.var_floor).reshape( -1,self.n_agents,  epi_len,latent_size) # (B,N, T, N * A)
            # infer_means_t = th.transpose(infer_means, 0, 1)  # (B, N, N * A)
            # infer_vars_t = th.transpose(infer_vars, 0, 1)  # (B, N, N * A)

            kld = my_kld(awareness_mean, awareness_var, infer_means_t, infer_vars_t).mean(dim=-1).mean(dim=-1,
                                                                                                       keepdim=True)

        h = h.view(bs * self.n_agents, -1)
        atten_in = awareness.view(bs * epi_len, self.args.n_agents, -1)  # (BNT, N, A)
        atten_out, atten = self.multi_head_attention(atten_in, atten_in,atten_in)  # atten_out: (BNT, N, A), //atten: (B * N, n_head, N, N)？
        atten_out = atten_out.view(bs * epi_len, -1)  # (BNT，N*A)

        x = x.reshape(bs * epi_len, self.args.rnn_hidden_dim) #(BN*T,R)即(BNT,R)
        q = self.fc2(th.cat([x,atten_out],dim=-1)) #(BN*T,n_actions)
        q = q.reshape(bs, epi_len, self.args.n_actions)
        return q, h, x, kld
#计算两个正态分布之间的 KL 散度
def my_kld(mu1, var1, mu2, var2):
    kld = 0.5 * th.log(var2 / var1) + 0.5 * (var1 + (mu1 - mu2).square()) / var2 - 0.5
    return kld