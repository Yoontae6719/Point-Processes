import pandas as pd
import torch
import numpy as np
import GTPP.dataset.generate_utils as gen_utils
from torch.utils.data.dataset import Dataset


class generatePointdata(Dataset):
    def __init__(self, cnf, train_type):

        self.time_step = cnf.time_step
        self.gen_type = cnf.gen_type
        self.train_type = train_type
        self.log_bool = cnf.log_bool #todo

        if self.gen_type == "hawkes1":
            [T, score_ref] = gen_utils.gen_hawkes1()

            self.train_input, self.train_target, self.val_input, self.val_target, self.test_input, self.test_target = self.ediff_fun(T)
        else:
            pass

        if self.train_type == "train":
            self.time_gap, self.output = self.train_input, self.train_target
            self.event_type = np.zeros(self.time_gap.shape)
            self.length = len(self.time_gap)

        elif self.train_type == "val":
            self.time_gap,self.output = self.val_input, self.val_target
            self.event_type = np.zeros(self.time_gap.shape)
            self.length = len(self.time_gap)

        elif self.train_type == "test":
            self.time_gap, self.output = self.test_input, self.test_target
            self.event_type = np.zeros(self.time_gap.shape)

            self.length = len(self.time_gap)

    def __len__(self):

        return self.length

    def __getitem__(self, item):

        return self.time_gap[item], self.event_type[item].astype(int),  self.length


    def rolling_matrix(self, x, window_size):
        x = x.flatten()
        n = x.shape[0]
        stride = x.strides[0]
        return np.lib.stride_tricks.as_strided(x, shape=(n-window_size+1, window_size), strides=(stride,stride) ).copy()


    def ediff_fun(self, T) -> object:
        train, val, test = np.ediff1d(T[0:60000]), np.ediff1d(T[60000:80000]), np.ediff1d(T[80000:])

        train = self.rolling_matrix(train, self.time_step)
        val = self.rolling_matrix(val, self.time_step)
        test = self.rolling_matrix(test, self.time_step)

        train_input,train_target = train[:, :-1].reshape(-1, self.time_step), train[:, [-1]]
        val_input, val_target      = val[:, :-1].reshape(-1, self.time_step),   val[:, [-1]]
        test_input,test_target = test[:, :-1].reshape(-1, self.time_step), test[:, [-1]]

        pd.DataFrame(train_input).to_csv(f"./data/{self.gen_type}_train_input.csv")
        pd.DataFrame(train_target).to_csv(f"./data/{self.gen_type}_train_target.csv")

        pd.DataFrame(val_input).to_csv(f"./data/{self.gen_type}_val_input.csv")
        pd.DataFrame(val_target).to_csv(f"./data/{self.gen_type}_val_target.csv")

        pd.DataFrame(test_input).to_csv(f"./data/{self.gen_type}_test_input.csv")
        pd.DataFrame(test_target).to_csv(f"./data/{self.gen_type}_test_target.csv")

        return train_input, train_target, val_input, val_target, test_input, test_target









