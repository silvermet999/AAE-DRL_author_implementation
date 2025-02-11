import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.nn.functional import one_hot

import utils
from EnvClass import Env
from AAE import AAE_archi_opt
from clfs import classifier
from data import main_u

from utils import RL_dataloader
from RL import TD3
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)

class Tester(object):
    def __init__(self, test_loader, model_encoder, model_decoder, model_disc, classifier, discrete):


        self.test_loader = RL_dataloader(test_loader)

        self.max_timesteps = 10000

        self.batch_size = 32
        self.eval_freq = 100
        self.start_timesteps = 50
        self.max_episodes_steps = 1000

        self.expl_noise = 0.3

        self.encoder = model_encoder
        self.discriminator = model_disc
        self.decoder = model_decoder
        self.classifier = classifier

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



    def evaluate(self):
        episode_num = 0
        number_correct = 0
        while True:
            print('input loader')
            try:
                state_t, label = self.test_loader.next_data()
                # episode_target = (torch.randint(4, label.shape) + label) % 4
                state = self.env.set_state(state_t)
                done = False
                episode_return = 0
            except:
                break

            while not done:
                with torch.no_grad():
                    continuous_act, discrete_act = self.policy.select_action(state)
                    next_state, reward, done = self.env(continuous_act, discrete_act, label)

                state = next_state
                episode_return += reward.mean()
            print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_return))
            episode_num += 1

            self.env.reset()

            with torch.no_grad():
                new_state = self.encoder(torch.tensor(state).cuda() if cuda else torch.tensor(state))
                labels = one_hot(label, num_classes=4).cuda() if cuda else one_hot(label, num_classes=4)

                nd, nc, nb = self.decoder((torch.cat([new_state, labels.squeeze(1)], dim=1).cuda() if cuda else
                                      torch.cat([new_state, labels.squeeze(1)], dim=1)))

            yield nd, nc, nb, episode_return



in_out = 30
z_dim = 10
label_dim = 4

discrete = {
    "ct_state_ttl": 6,
            "trans_depth": 11,
            "proto": 2,
}


dataset = utils.dataset(original=True, train=False)
test_loader =utils.dataset_function(dataset, 32, 64, train=False)

encoder_generator = AAE_archi_opt.EncoderGenerator(in_out, z_dim).cuda() if cuda else (
    AAE_archi_opt.EncoderGenerator(in_out, z_dim))
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

tester = Tester(test_loader, encoder_generator, decoder, discriminator, classifier_model, discrete)
evaluater = tester.evaluate()
for i in range(100):
    next(evaluater)
