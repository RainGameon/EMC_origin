import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE architecture
class Predict_Network_vae(nn.Module):
    def __init__(self, args, num_inputs,hidden_dim, output_shape ):
        super(Predict_Network_vae, self).__init__()

        self.args      = args
        self.lambda_kl = args.lambda_kl
        self.num_inputs = num_inputs
        self.output_shape = output_shape
        self.latent_dim    = 4
        self.hidden_dim    = hidden_dim

        self.encoder = Encoder_VAE(args, self.num_inputs, self.hidden_dim, self.latent_dim)

        self.decoder = Decoder(args, self.latent_dim, self.hidden_dim, self.output_shape)
        self.criterion = nn.MSELoss(reduction="sum") # for reconstruction loss

    def reparameterize(self, mu, log_var, flagTraining):
        #if (self.stochastic_sample == True):
        if flagTraining == True:
            std = th.exp(0.5 * log_var)
            eps = th.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z

    #.. VAE forward --------------------------------------------
    def forward(self, input ,flagTraining):  #  (bs,T-1,n_agent-1,num_inputs)
        mu, log_var = self.encoder(input)  # 按理为 (bs,T-1,n_agent-1,4)
        x = self.reparameterize(mu, log_var, flagTraining=flagTraining)
        obs_next_pre = self.decoder( x )

        return obs_next_pre, mu, log_var
    #-----------------------------------------------------------
    # Define the loss function (negative ELBO) and move it to the GPU
    def loss_function_vae(self, obs_next_pre, obs_next, mu, log_var, mask):
        loss = F.mse_loss(obs_next_pre, obs_next, reduction='none')
        loss = loss.sum(dim=-1, keepdim=False)  # bs,t , self.n_agents-1
        loss = loss.sum(dim=-1, keepdim=True)   # bs,t , 1
        recon_obs_loss = (loss * mask).sum() / mask.sum()
        kl_divergence = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())  # 进行平滑
        return recon_obs_loss  + self.lambda_kl * kl_divergence
    # 单纯计算预测obs，不是为了更新参数
    def get_log_pi(self, own_variable, other_variable,flagTraining = False): 
        predict_variable, mu, log_var = self.forward(own_variable,flagTraining=flagTraining)  # (bs,T-1,n_agent-1,obs_shape)：即预测的observation
        log_prob = -1 * F.mse_loss(predict_variable, other_variable, reduction='none')   # (bs,T-1,n_agent-1,1)：即每个agent预测的obs和实际obs的均方误差（为了保留每个agent的单独损失值,而不是直接返回一个汇总后的标量损失值）
        log_prob = th.sum(log_prob, -1, keepdim=False)
        return log_prob

    # 计算loss for train
    def update(self, own_variable, other_variable, mask, flagTraining=True): 
        if mask.sum() > 0:
            predict_variable, mu, log_var  = self.forward(own_variable, flagTraining)
            loss_vae = self.loss_function_vae(predict_variable, other_variable, mu, log_var, mask)
            return loss_vae
        return None

# Define the encoder architecture
class Encoder_VAE(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        device = args.device

        super(Encoder_VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc31 = nn.Linear(hidden_dim, latent_dim).to(device)
        self.fc32 = nn.Linear(hidden_dim, latent_dim).to(device)

    def forward(self, input):  #  (bs,T-1,n_agent-1,num_inputs)
        # bs    = inputs.size()[0]
        # max_t = inputs.size()[1]
        # net_inputs = inputs.reshape(bs * max_t, -1)

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        mu = self.fc31(x)
        log_var = self.fc32(x)  # 按理为（bs,t,n_agent-1,4)
        
        return mu, log_var


# Define the decoder architecture
class Decoder(nn.Module):
    def __init__(self, args, latent_dim, hidden_dim, output_dim):
        device = args.device
        input_dim = latent_dim

        super(Decoder, self).__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, input): # 按理为 (bs,T-1,n_agent-1,4)
        # bs    = input.size()[0]
        # max_t = input.size()[1]
        # net_inputs = input.reshape(bs * max_t, -1)

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        obs_hat = self.fc3(x)  #  按理为 (bs,T-1,n_agent-1,obs_shape)
        
        return obs_hat
