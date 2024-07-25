import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from utils.torch_utils import to_cuda
from torch.optim import Adam
import torch.nn.functional as func
from controllers import REGISTRY as mac_REGISTRY
import numpy as np

class vdn_QLearner_Curiosity:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        # self.mac = copy.deepcopy(mac)
        # self.target_mac = copy.deepcopy(mac)
        self.soft_target_mac = copy.deepcopy(mac)
        self.predict_mac = mac_REGISTRY[args.mac](scheme, groups, args)

        self.logger = logger
        # self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0

        # self.mixer = VDNMixer()
        # self.target_mixer = copy.deepcopy(self.mixer)

        # self.params += list(self.mixer.parameters())
        self.predict_params = list(self.predict_mac.parameters())

        self.decay_stats_t = 0
        self.decay_stats_t_2 = 0
        self.state_shape = scheme["state"]["vshape"]

        

        # self.optimiser = Adam(params=self.params, lr=args.lr)
        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.predict_optimiser = Adam(params=self.predict_params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        #testS
        self.log_stats_t = -self.args.learner_log_interval - 1

    def subtrain(self, batch: EpisodeBatch, t_env: int, episode_num: int,save_buffer=False, imac=None, timac=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.predict_mac.init_hidden(batch.batch_size)
        self.soft_target_mac.init_hidden(batch.batch_size)
        soft_target_mac_out= self.soft_target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[0][:, 1:, ...]

        predict_mac_out = self.predict_mac.forward(batch, batch.max_seq_length, batch_inf=True)[0][:, 1:, ...]

        soft_target_mac_out_next = soft_target_mac_out.clone().detach()
        soft_target_mac_out_next_mean =th.mean(soft_target_mac_out_next)
        soft_target_mac_out_next = soft_target_mac_out_next.contiguous().view(-1, self.args.n_actions)  * 10

        predict_mac_out = predict_mac_out.contiguous().view(-1, self.args.n_actions)
        predict_mac_out_mean = th.mean(predict_mac_out)
        prediction_error = func.pairwise_distance(predict_mac_out, soft_target_mac_out_next, p=2.0, keepdim=True)
        prediction_mask = mask.repeat(1, 1, self.args.n_agents)
        prediction_error = prediction_error.reshape(batch.batch_size, -1, self.args.n_agents) * prediction_mask

        if hasattr(self.args, 'mask_other_agents') and self.args.mask_other_agents:
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.detach()[:, :, 0:1])
        else:
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.mean(dim=-1, keepdim=True).detach())

        prediction_loss = prediction_error.sum() / prediction_mask.sum()
        ############################

        if save_buffer:
            return intrinsic_rewards

        self.predict_optimiser.zero_grad()
        prediction_loss.backward()
        predict_grad_norm = th.nn.utils.clip_grad_norm_(self.predict_params, self.args.grad_norm_clip)
        self.predict_optimiser.step()

        if self.args.curiosity_decay:
            if t_env - self.decay_stats_t >= self.args.curiosity_decay_cycle:
                if self.args.curiosity_decay_rate <= 1.0:
                    if self.args.curiosity_scale > self.args.curiosity_decay_stop:
                         self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                    else:
                         self.args.curiosity_scale = self.args.curiosity_decay_stop
                else:
                     if self.args.curiosity_scale < self.args.curiosity_decay_stop:
                         self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                     else:
                         self.args.curiosity_scale = self.args.curiosity_decay_stop

                self.decay_stats_t=t_env



        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("vdn loss", loss.item(), t_env)
            self.logger.log_stat("curiosity_scale", self.args.curiosity_scale, t_env)
            self.logger.log_stat("curiosity_decay_rate", self.args.curiosity_decay_rate, t_env)
            self.logger.log_stat("curiosity_decay_cycle", self.args.curiosity_decay_cycle, t_env)
            self.logger.log_stat("curiosity_decay_stop", self.args.curiosity_decay_stop, t_env)
            # self.logger.log_stat("vdn hit_prob", hit_prob.item(), t_env)
            # self.logger.log_stat("vdn grad_norm", grad_norm, t_env)
            self.logger.log_stat("vdn predict_grad_norm", predict_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("vdn prediction loss", prediction_loss.item(), t_env)

            self.logger.log_stat("vdn intrinsic rewards", intrinsic_rewards.sum().item() / mask_elems, t_env)
            self.logger.log_stat("vdn extrinsic rewards", rewards.sum().item() / mask_elems, t_env)
            # self.logger.log_stat("vdn td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # self.logger.log_stat("vdn q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("vdn target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("vdn soft_target_mac_out_next_mean",soft_target_mac_out_next_mean, t_env)
            self.logger.log_stat("vdn predict_mac_out_mean", predict_mac_out_mean, t_env)

            self.log_stats_t = t_env

        return intrinsic_rewards


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int,save_buffer=False, imac=None, timac=None):
        if episode_num < 100:
            self.soft_target_mac.load_state(imac)
            print('episode_num = {} init soft_target_mac'.format(episode_num))

        self._smooth_update_predict_targets(timac)
        intrinsic_rewards = \
            self.subtrain(batch, t_env, episode_num, save_buffer=save_buffer, imac=imac, timac=timac)
        # if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_episode = episode_num
        return intrinsic_rewards


    def _update_targets(self):
        # self.target_mac.load_state(self.mac)
        # if self.mixer is not None:
        #     self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _smooth_update_predict_targets(self, timac=None):
        self.soft_update(self.soft_target_mac, timac, self.args.soft_update_tau)
        # self.logger.console_logger.info("Updated soft target network")

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)




    def cuda(self):
        # to_cuda(self.mac, self.args.device)
        # to_cuda(self.target_mac, self.args.device)
        to_cuda(self.soft_target_mac, self.args.device)
        to_cuda(self.predict_mac, self.args.device)
        
        # if self.mixer is not None:
        #     to_cuda(self.mixer, self.args.device)
        #     to_cuda(self.target_mixer, self.args.device)


    def save_models(self, path):
        # self.mac.save_models(path)
        self.predict_mac.save_models('{}/predict_mac'.format(path))
        

        # if self.mixer is not None:
        #     th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        #     th.save(self.target_mixer.state_dict(), "{}/target_mixer.th".format(path))

        # th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.predict_optimiser.state_dict(), "{}/predict_opt.th".format(path))
        

    def load_models(self, path):
        # self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        # self.target_mac.load_models(path)
        self.soft_target_mac.load_models(path)
        self.predict_mac.load_models('{}/predict_mac'.format(path))
        

        # if self.mixer is not None:
        #     self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        #     self.target_mixer.load_state_dict(th.load("{}/target_mixer.th".format(path),
        #                                               map_location=lambda storage, loc: storage))

        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.predict_optimiser.load_state_dict(th.load("{}/predict_opt.th".format(path),
                                                       map_location=lambda storage, loc: storage))
       
