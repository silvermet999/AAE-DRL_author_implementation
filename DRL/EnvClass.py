# libraries
import torch
from torch.nn import Module, BCELoss, CrossEntropyLoss
import os

from torch.nn.functional import one_hot

import utils


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
cuda = True if torch.cuda.is_available() else False

class Env(Module):
    def __init__(self, model_G, model_D, model_c, model_de):
        super(Env, self).__init__()

        self._state = None
        self.decoder = model_de
        self.generator = model_G
        self.disciminator = model_D
        self.classifier = model_c

        self.d_reward_coeff = 0.4
        self.cl_reward_coeff = 0.2

        self.count = 0

    def reset(self):
        self._state = None
        self.count = 0
        return self._state


    def set_state(self, state):
        self._state = state
        return state.detach().cpu().numpy()

    def forward(self, action, disc, episode_target):
        d_decoded = {feature: [] for feature in self.decoder.discrete_features}
        c_decoded = {feature: [] for feature in self.decoder.continuous_features}
        b_decoded = {feature: [] for feature in self.decoder.binary_features}

        with (torch.no_grad()):
            episode_target = episode_target.to(torch.long).cuda() if cuda else episode_target.to(torch.long)
            labels = one_hot(episode_target, num_classes=4)
            episode_target_indices = labels.argmax(dim=1)

            z_cont = torch.tensor(action).cuda() if cuda else torch.tensor(action)
            z_disc = torch.tensor(list(disc.values())).cuda() if cuda else torch.tensor(list(disc.values()))
            z_disc = z_disc.expand(z_cont.size(0), -1).cuda() if cuda else z_disc.expand(z_cont.size(0), -1)
            for i in range(z_cont.shape[0]):
                z = torch.concat([z_cont[i].unsqueeze(0), z_disc[i].unsqueeze(0), labels[i].unsqueeze(0)], 1)
                d, c, b = self.decoder(z)
                d_decoded, c_decoded, b_decoded = utils.types_append(self.decoder, d, c, b, d_decoded, c_decoded, b_decoded)
            d_decoded, c_decoded, b_decoded = utils.type_concat(self.decoder, d_decoded, c_decoded, b_decoded)
            decoded = utils.all_samples(d_decoded, c_decoded, b_decoded)

            gen_out = self.generator(decoded)

            dis_judge = self.disciminator(gen_out)
            decoded_normalized = (decoded - decoded.min(dim=0)[0]) / (decoded.max(dim=0)[0] - decoded.min(dim=0)[0] + 1e-7)

            classifier_output, entropy = self.classifier.encoder(decoded_normalized)
            classifier_logits, predictions = self.classifier.classify(classifier_output)
            max_probs, pseudo_labels = torch.max(predictions, dim=1)

        class_weights = torch.tensor([4.0, 2.0, 1.0, 1.0]).cuda() if cuda else torch.tensor(
            [4.0, 2.0, 1.0, 1.0])
        reward_cl = self.cl_reward_coeff * torch.nn.functional.cross_entropy(classifier_logits,
                                                                             pseudo_labels, weight=class_weights).cpu().data.numpy()

        reward_d = self.d_reward_coeff * torch.nn.functional.binary_cross_entropy(dis_judge,
                    torch.ones((dis_judge.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones(
                                                    (dis_judge.shape[0], 1), requires_grad=False)).cpu().data.numpy()

        print("d", torch.nn.functional.binary_cross_entropy(dis_judge, torch.ones((dis_judge.shape[0], 1),
                requires_grad=False).cuda() if cuda else torch.ones((dis_judge.shape[0], 1), requires_grad=False)).cpu().data.numpy())
        print("cl", torch.nn.functional.cross_entropy(classifier_logits, pseudo_labels, weight=class_weights).cpu().data.numpy())
        reward = reward_cl + reward_d


        done = True
        self.count += 1

        next_state = decoded.detach().cpu().data.numpy()
        self._state = decoded
        return next_state, reward, done
