import pickle

import pandas as pd
from scipy.stats import gamma, rayleigh, expon, chi2, exponpow, lognorm, norm
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import torch
from data import main_u

cuda = True if torch.cuda.is_available() else False


def custom_dist(size):
    for i, _ in enumerate(main_u.X.columns):
        if i in [22]:
            z = lognorm(1, 0, 0).rvs(size)
        elif i in [0, 27, 28, 29]:
            z = chi2(0.8, 0, 0.4).rvs(size)
        elif i in [1, 6, 7, 8, 9, 11, 12, 16, 23]:
            z = rayleigh(0, 0).rvs(size)
        elif i in [17, 18, 24, 26]:
            z= gamma(0.1, 0, 0.2).rvs(size)
        elif i in [3]:
            z=norm(1.4, 0.8).rvs(size)
        elif i in [2, 25]:
            z = exponpow(1.5, 0.1, 1).rvs(size)
        else:
            z = expon(0, 0).rvs(size)

        z = torch.tensor(z, dtype=torch.float32).cuda() if cuda else torch.tensor(z, dtype=torch.float32)
    return z

discrete = {
    # "attack_cat": 10,
            "state": 5,
            # "service": 13,
            "ct_state_ttl": 6,
            # "dttl": 9,
            # "sttl": 13,
            "trans_depth": 11
}

binary = ["proto", "is_ftp_login"]
discrete_and_binary = set(discrete.keys()).union(set(binary))
continuous = [feature for feature in main_u.X_train.columns if feature not in discrete_and_binary]


def types_append(decoder, discrete_out, continuous_out, binary_out, discrete_samples, continuous_samples, binary_samples):
    for feature in decoder.discrete_features:
        discrete_samples[feature].append(torch.argmax(torch.round(discrete_out[feature]), dim=-1))

    for feature in decoder.continuous_features:
        continuous_samples[feature].append(continuous_out[feature])

    for feature in decoder.binary_features:
        binary_samples[feature].append(torch.argmax(torch.round(binary_out[feature]), dim=-1))
    return discrete_samples, continuous_samples, binary_samples

def type_concat(decoder, discrete_samples, continuous_samples, binary_samples):
    for feature in decoder.discrete_features:
        discrete_samples[feature] = torch.cat(discrete_samples[feature], dim=0)

    for feature in decoder.continuous_features:
        continuous_samples[feature] = torch.cat(continuous_samples[feature], dim=0)

    for feature in decoder.binary_features:
        binary_samples[feature] = torch.cat(binary_samples[feature], dim=0)

    return discrete_samples, continuous_samples, binary_samples



def all_samples(discrete_samples, continuous_samples, binary_samples):
    discrete_tensors = list(discrete_samples.values())
    continuous_tensors = list(continuous_samples.values())
    binary_tensors = list(binary_samples.values())

    all_tensors = discrete_tensors + continuous_tensors + binary_tensors
    all_tensors = [t.unsqueeze(-1) if t.dim() == 1 else t for t in all_tensors]
    combined = torch.cat(all_tensors, dim=1)
    return combined


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


def dataset_function(dataset, batch_size_t, batch_size_o, train=True):
    total_size = len(dataset)
    test_size = total_size // 5
    val_size = total_size // 10
    train_size = total_size - (test_size + val_size)
    train_subset = Subset(dataset, range(train_size))
    val_subset = Subset(dataset, range(train_size, train_size + val_size))
    test_subset = Subset(dataset, range(train_size + val_size, total_size))
    if train:
        train_loader = DataLoader(train_subset, batch_size=batch_size_t, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size_o, shuffle=False)
        return train_loader, val_loader

    else:
        test_loader = DataLoader(test_subset, batch_size=batch_size_o, shuffle=False)

        return test_loader


def inverse_sc_cont(X, synth):
    synth_inv = synth * (X.max() - X.min()) + X.min()
    return pd.DataFrame(synth_inv, columns=X.columns, index=synth.index)



def dataset(original=False, train=True):
    if original:
        if train:
            dataset = CustomDataset(main_u.X_train_sc.to_numpy(), main_u.y_train.to_numpy())
        else:
            dataset = CustomDataset(main_u.X_test_sc.to_numpy(), main_u.y_test.to_numpy())
    else:
        df_org = pd.concat([main_u.X_sc, main_u.y], axis=1)
        X_rl = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/DDPG/rl_bal1.csv"))
        X_rl = X_rl.apply(lambda col: col.str.strip("[]").astype(float) if col.dtype == "object" else col)
        y_rl = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/clfs/labels.csv"))
        df_rl = pd.concat([X_rl, y_rl], axis=1)
        df_rl = df_rl[df_rl["attack_cat"] != 2]
        df = pd.concat([df_org, df_rl], axis=0)
        X = df.drop(["attack_cat"], axis=1)
        y = df["attack_cat"]

        X_train, X_test, y_train, y_test = main_u.vertical_split(X, y)
        if train:
            dataset = CustomDataset(X_train.to_numpy(), labels=y_train.to_numpy())
        else:
            dataset = CustomDataset(X_test.to_numpy(), labels=y_test.to_numpy())
    return dataset



class RL_dataloader:
    def __init__(self, dataloader):
        self.loader = dataloader
        self.loader_iter = iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def next_data(self):
        try:
            data, label = next(self.loader_iter)

        except:
            self.loader_iter = iter(self.loader)
            data, label = next(self.loader_iter)

        return data, label




class ReplayBuffer(object):
    def __init__(self):
        self.storage = []
        self._saved = []
        self._sample_ind = None
        self._ind_to_save = 0

    def add(self, data):
        self.storage.append(data)
        self._saved.append(False)

    def sample(self):
        ind = np.random.randint(0, len(self.storage))
        self._sample_ind = ind
        return self[ind]

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, items):
        if hasattr(items, '__iter__'):
            items_iter = items
        else:
            items_iter = [items]

        s, a1, a2, n, r, d, t = [], [], [], [], [], [], []
        for i in items_iter:
            S, A1, A2, N, R, D, T = self.storage[i]
            s.append(np.array(S, copy=False))
            a1.append(np.array(A1.detach().cpu().numpy(), copy=False))
            a2.append(np.array(list(A2.items()), copy=False))
            n.append(np.array(N, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            t.append(np.array(T, copy=False))

        return (np.array(s).squeeze(0), np.array(a1).squeeze(0), np.array(a2).squeeze(0),
                np.array(n).squeeze(0), np.array(r).squeeze(0), np.array(d).squeeze(0).reshape(-1, 1), np.array(t).squeeze(0))



MINORITY_CLASSES = [0, 1]
def is_minority_class(generated_sample, classifier=None):
    """
    Checks if the generated sample belongs to a minority class.

    Args:
        generated_sample: The generated data sample.
        classifier: (Optional) A pre-trained classifier to predict the sample's class if not explicitly available.

    Returns:
        bool: True if the sample belongs to a minority class, False otherwise.
    """
    # 1. Extract the class label directly, or use a classifier if the label is not provided
    if classifier:  # Use a classifier to predict the class
        _, sample_class = classifier.classify(generated_sample)
    else:
        raise ValueError("no classifier provided for prediction.")

    # 2. Check if the class is in the minority classes
    return sample_class in MINORITY_CLASSES


def calculate_reward(classifier, reward, generated_sample, minority_class_ratio=3):
    if is_minority_class(generated_sample, classifier):
        minority_boost = 1 / minority_class_ratio
        return reward * minority_boost

    return reward

