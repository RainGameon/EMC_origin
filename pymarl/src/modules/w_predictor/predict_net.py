from numpy import add
import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F


class Predict_Network_OS(nn.Module):

    def __init__(self, num_inputs,hidden_dim, obs_shape, state_shape, n_agents, lr=3e-4):
        super(Predict_Network_OS, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc_obs = nn.Linear(hidden_dim, obs_shape* n_agents)
        self.last_fc_state = nn.Linear(hidden_dim, state_shape)
        self.temperature = 0.7

    def forward(self, input):
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        next_obs = self.last_fc_obs(h)
        next_state = self.last_fc_state(h)
        return next_obs, next_state

    # def get_log_pi(self, own_variable, other_variable):
    #     predict_variable = self.forward(own_variable)
    #     log_prob = -1 * F.mse_loss(predict_variable,
    #                                other_variable, reduction='none')
    #     log_prob = torch.sum(log_prob, -1, keepdim=True)
    #     return log_prob

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None


class Predict_Network(nn.Module):

    def __init__(self, num_inputs,hidden_dim, output_shape , lr=3e-4): # output_shape：obs_shape
        super(Predict_Network, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, output_shape)
        self.temperature = 0.7 #0.5

    def forward(self, input, is_softmax=False): # (bs,T-1,n_agent-1,num_inputs)
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        if is_softmax:
            x = F.softmax(x / self.temperature, dim = -1)
        return x
    
    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)    # (bs,T-1,n_agent-1,obs_shape)：即预测的observation
        log_prob = -1 * F.mse_loss(predict_variable,     # (bs,T-1,n_agent-1,1)：即每个agent预测的obs和实际obs的均方误差（为了保留每个agent的单独损失值,而不是直接返回一个汇总后的标量损失值）
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=False)
  
        return log_prob
    
    def update_ps(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            # print(predict_variable.shape, other_variable.shape)
            loss = F.mse_loss(predict_variable, other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)  # bs,t,1
            # print('loss.shape={}, mask.shape={}'.format(loss.shape, mask.shape)) # [bs, t,1])
            loss = (loss * mask).sum() / mask.sum()

            return loss
    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            # print(mask.shape, other_variable.shape)
            loss = F.mse_loss(predict_variable, other_variable, reduction='none')  # bs,t, 1
            loss = loss.sum(dim=-1, keepdim=True) # bs,t , 1
            
            loss = (loss * mask).sum() / mask.sum()

            return loss

        return None


class Predict_Network_VAE(nn.Module):

    def __init__(self, num_inputs, hidden_dim, output_shape, lr=3e-4):  # output_shape：obs_shape
        super(Predict_Network_VAE, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, output_shape)

    def forward(self, input):
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)  # (bs,T-1,n_agent-1,obs_shape)：即预测的observation
        log_prob = -1 * F.mse_loss(predict_variable,other_variable, reduction='none')
                                   # (bs,T-1,n_agent-1,1)：即每个agent预测的obs和实际obs的均方误差（为了保留每个agent的单独损失值,而不是直接返回一个汇总后的标量损失值）

        log_prob = torch.sum(log_prob, -1, keepdim=False)

        return log_prob
    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable, other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=False)  # bs,t , self.n_agents-1
            loss = loss.sum(dim=-1, keepdim=True)  # bs,t , 1
            loss = (loss * mask).sum() / mask.sum()
            return loss

        return None