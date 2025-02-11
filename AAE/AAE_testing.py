import pandas as pd
from torch.optim import SGD

import utils
from AAE import AAE_archi_opt
import os
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, one_hot

from torch import cuda
import itertools
from data import main_u
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)


"""--------------------------------------------------dataset and models--------------------------------------------------"""
in_out = 30
z_dim = 10
label_dim = 4


dataset = utils.dataset(original=False, train=False)
test_loader =utils.dataset_function(dataset, 32, 64, train=False)


encoder_generator = AAE_archi_opt.EncoderGenerator(in_out, z_dim).cuda() if cuda else (
    AAE_archi_opt.EncoderGenerator(in_out, z_dim))
encoder_generator.load_state_dict(torch.load("aae_fin.pth")["enc_gen"])


decoder = AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary).cuda() if cuda \
    else (AAE_archi_opt.Decoder(z_dim+label_dim, in_out, utils.discrete, utils.continuous, utils.binary))
decoder.load_state_dict(torch.load("aae_fin.pth")["dec"])

discriminator = AAE_archi_opt.Discriminator(z_dim, ).cuda() if cuda else (
    AAE_archi_opt.Discriminator(z_dim, ))
discriminator.load_state_dict(torch.load("aae_fin.pth")["disc"])


optimizer_G = SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters()),
                              lr=0.001)
optimizer_D = SGD(discriminator.parameters(), lr=0.001)


def test_model(test_loader):
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()

    total_g_loss = 0
    total_d_loss = 0

    with torch.no_grad():
        for X, y in test_loader:
            valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                    requires_grad=False)
            fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                     requires_grad=False)

            real = X.type(torch.FloatTensor).cuda() if cuda else X.type(torch.FloatTensor)
            y = y.type(torch.LongTensor).squeeze().cuda() if cuda else y.type(torch.LongTensor).squeeze()
            y = one_hot(y, num_classes=4)

            discrete_targets = {}
            continuous_targets = {}
            binary_targets = {}

            for feature, _ in decoder.discrete_features.items():
                discrete_targets[feature] = real[:, :3]

            for feature in decoder.continuous_features:
                continuous_targets[feature] = real[:, 5:]

            for feature in decoder.binary_features:
                binary_targets[feature] = real[:, 3:5]

            encoded = encoder_generator(real)
            dec_input = torch.cat([encoded, y], dim=1)
            discrete_outputs, continuous_outputs, binary_outputs = decoder(dec_input)

            g_loss = (0.1 * binary_cross_entropy(discriminator(encoded),
                                                 torch.ones((X.shape[0], 1),
                                                            requires_grad=False).cuda() if cuda else torch.ones(
                                                     (X.shape[0], 1), requires_grad=False)) +
                      0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                                 (discrete_targets, continuous_targets, binary_targets)))


            z = torch.rand(real.shape[0], z_dim).cuda() if cuda else torch.rand(real.shape[0], z_dim)
            real_loss = binary_cross_entropy(discriminator(z), valid)
            fake_loss = binary_cross_entropy(discriminator(encoded.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_g_loss = total_g_loss / len(test_loader)
        avg_d_loss = total_d_loss / len(test_loader)

    return {
        'g_loss': avg_g_loss,
        'd_loss': avg_d_loss
    }