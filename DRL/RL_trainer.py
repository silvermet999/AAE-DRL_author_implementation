import csv
import random

import pandas as pd
import torch
import torch.utils.data
import os

from torch.nn.functional import one_hot

import utils
from EnvClass import Env

import numpy as np

from clfs import classifier
from data import main_u
from utils import RL_dataloader
from RL import TD3
from AAE import AAE_archi_opt


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False

def evaluate_policy(policy, dataloader, env, eval_episodes=10):
    torch.manual_seed(0)
    np.random.seed(0)
    # random.seed(42)

    avg_reward = 0.

    for _ in range(eval_episodes):
        input, episode_target = dataloader.next_data()
        obs = env.set_state(input)
        env.reset()
        done = False
        # episode_target = (label + torch.randint(4, label.shape)) % 4

        while not done:
            continuous_act, discrete_act = policy.select_action(obs)
            new_state, reward, done = env(continuous_act, discrete_act, episode_target)
            avg_reward += reward

    avg_reward /= eval_episodes

    return avg_reward




class Trainer(object):
    def __init__(self, train_loader, valid_loader, model_encoder, model_disc, model_decoder, model_classifier, in_out, discrete):
        torch.manual_seed(0)
        np.random.seed(0)
        # random.seed(42)

        self.train_loader = RL_dataloader(train_loader)
        self.valid_loader = RL_dataloader(valid_loader)

        self.epoch_size = len(self.valid_loader)
        self.max_timesteps = 4000

        self.batch_size = 32
        self.eval_freq = 400
        self.start_timesteps = 50
        self.max_episodes_steps = 100

        self.expl_noise = 0.3

        self.encoder = model_encoder
        self.discriminator = model_disc
        self.decoder = model_decoder
        self.classifier = model_classifier

        self.env = Env(self.encoder, self.discriminator, self.classifier, self.decoder)
        self.replay_buffer = utils.ReplayBuffer()

        torch.manual_seed(0)
        np.random.seed(0)

        self.state_dim = in_out
        self.action_dim = 7
        self.discrete_features = discrete
        self.max_action = 1
        self.policy = TD3(self.state_dim, self.action_dim, self.discrete_features, self.max_action)

        self.continue_timesteps = 0

        self.evaluations = []

    def train(self):
        sum_return = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        d_list = []
        c_list = []
        b_list = []

        state_t, episode_target = self.train_loader.next_data()
        state = self.env.set_state(state_t)
        # episode_target = (torch.randint(4, labels.shape) + labels) % 4

        done = False
        self.env.reset()


        for t in range(int(self.continue_timesteps), int(self.max_timesteps)):
            episode_timesteps += 1
            if t < self.start_timesteps:
                continuous_act = torch.randn(self.batch_size, self.action_dim)
                discrete_act = {name: random.randrange(num_actions) for name, num_actions in self.discrete_features.items()}
            else:
                continuous_act, discrete_act = self.policy.select_action(state)

            next_state, reward, done = self.env(continuous_act, discrete_act, episode_target)

            self.policy.store_transition(state, continuous_act, discrete_act, next_state, reward, done, episode_target)

            state = next_state
            episode_reward += reward

            if t >= self.start_timesteps:
                self.policy.train()


            if done:
                state_t, episode_target = self.train_loader.next_data()
                state = self.env.set_state(state_t)

                done = False
                self.env.reset()

                print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_reward), end='')
                sum_return += episode_reward
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                episode_result = "episode: {} average reward: {}".format(episode_num,
                                                                                  sum_return / episode_num)
                print('\r' + episode_result)

                self.evaluations.append(evaluate_policy(self.policy, self.valid_loader, self.env))
                eval_result = "episodes: {}".format(self.evaluations[-1])
                print(eval_result)

            new_state = self.encoder(torch.tensor(state).float().cuda() if cuda else torch.tensor(state).float())
            labels = one_hot(episode_target, num_classes=4).cuda() if cuda else one_hot(episode_target, num_classes=4)
            nd, nc, nb = self.decoder(torch.cat([new_state, labels.squeeze(1)], dim=1).cuda() if cuda else
                                      torch.cat([new_state, labels.squeeze(1)], dim=1))
            d_list.append(nd)
            c_list.append(nc)
            b_list.append(nb)
        d_cat = {key: torch.cat([d[key] for d in d_list], dim=0) for key in d_list[0]}
        c_cat = {key: torch.cat([d[key] for d in c_list], dim=0) for key in c_list[0]}
        b_cat = {key: torch.cat([d[key] for d in b_list], dim=0) for key in b_list[0]}
        return d_cat, c_cat, b_cat




in_out = 30
z_dim = 10
label_dim = 4

discrete = {"ct_state_ttl": 6,
            "trans_depth": 11,
            "proto": 2,
}
encoder_generator = AAE_archi_opt.EncoderGenerator(in_out, z_dim).cuda() if cuda else (
    AAE_archi_opt.EncoderGenerator(in_out, z_dim))
# encoder_generator.load_state_dict(torch.load("/home/silver/PycharmProjects/AAEDRL/AAE/aae3.pth", map_location="cpu")["enc_gen"])
encoder_generator.eval()

decoder = AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary).cuda() if cuda else (
    AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary))
decoder.load_state_dict(torch.load("/home/silver/PycharmProjects/AAEDRL/AAE/aae3.pth")["dec"])
decoder.eval()


discriminator = AAE_archi_opt.Discriminator(z_dim, ).cuda() if cuda else (
    AAE_archi_opt.Discriminator(z_dim, ))
discriminator.load_state_dict(torch.load("/home/silver/PycharmProjects/AAEDRL/AAE/aae3.pth")["disc"])
discriminator.eval()

classifier_model = classifier.TabNetModel().cuda() if cuda else classifier.TabNetModel()
classifier_model.load_state_dict(torch.load("/home/silver/PycharmProjects/AAEDRL/clfs/best_model_rl1.pth"))
classifier_model.eval()

dataset = utils.dataset(original=True)
train_loader, val_loader = utils.dataset_function(dataset, 32, 64, train=True)

d, c, b = Trainer(train_loader, val_loader, encoder_generator, discriminator, decoder, classifier_model, in_out, discrete).train()

# d_dict = {key: tensor.detach().cpu().numpy() for key, tensor in d.items()}
# c_dict = {key: tensor.detach().cpu().numpy() for key, tensor in c.items()}
# b_dict = {key: tensor.detach().cpu().numpy() for key, tensor in b.items()}
#
# d_max = {key: np.argmax(value, axis=1) for key, value in d_dict.items()}
# b_max = {key: np.argmax(value, axis=1) for key, value in b_dict.items()}
#
# all_dict = {**d_max, **c_dict, **b_max}


# with open('rl_bal.csv', 'w', newline='') as file_d:
#     writer = csv.writer(file_d)
#     keys = list(all_dict.keys())
#     writer.writerow(keys)
#     max_len = max(len(all_dict[key]) for key in keys)
#     for i in range(max_len):
#         row = [all_dict[key][i] if i < len(all_dict[key]) else '' for key in keys]
#         writer.writerow(row)
