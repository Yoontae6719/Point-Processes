import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from time import time
import click


from conf import Conf
from progressbar import ProgressBar
from models.net import Net
from dataset.dataset import generatePointdata

class Trainer(object):
    def __init__(self, cnf):
        self.cnf = cnf

        train_loader = generatePointdata(cnf, "train")
        val_loader = generatePointdata(cnf, "test")

        self.dataset_train = DataLoader(train_loader, num_workers=self.cnf.n_workers,  batch_size= self.cnf.batch_size,  shuffle=True)
        self.dataset_val      = DataLoader(val_loader, num_workers=self.cnf.n_workers,  batch_size= self.cnf.batch_size, shuffle=False)


        # Model init
        self.model = Net(cnf = self.cnf )
        self.model = self.model.to(cnf.device)
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), self.cnf.lr, betas=(0.9, 0.999), eps=1e-05)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 10, gamma=0.5)

        self.nll_loss, self.log_hazard_loss, self.cumsum_hazard_loss = 0, 0, 0
        self.val_nll_loss, self.val_log_hazard_loss, self.val_cumsum_hazard_loss = 0, 0, 0

        self.best_test_loss = None
        self.epoch = 0

        # init log path
        self.log_path = cnf.exp_log_path
        self.sw = SummaryWriter(self.log_path)

        # init progress bar
        self.log_freq = len(self.dataset_train)
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

    def train(self):
        self.model.train()

        times =[]
        for step, batch in enumerate(self.dataset_train):
            self.optimizer.zero_grad()
            t = time()

            # Feed input to the model
            time_gap, event_type, length = map(lambda x: x.to(self.cnf.device), batch)

            nll, log_hazard_function, cumsum_hazard_function, hazard_function = self.model.forward(time_gap, event_type)


            loss = nll.float().to(self.cnf.device)

            loss.backward()

            # update parameters
            self.optimizer.step()

            # Write the train loss
            self.nll_loss += nll.item()
            self.log_hazard_loss += log_hazard_function.item()
            self.cumsum_hazard_loss += cumsum_hazard_function.item()

            # print progress bar
            times.append(time() - t)
            #print(length.cpu().numpy())

            if self.cnf.log_each_step:
                print(f'\r{self.progress_bar} '
                      f'│ Log-Likelihood: {np.mean(self.nll_loss / length.cpu().numpy()):.6f} '
                      f'│ Log-Hazard: {-np.mean(self.log_hazard_loss / length.cpu().numpy()):.6f} '
                      f'│ Cumsum-Hazard: {np.mean(self.cumsum_hazard_loss / length.cpu().numpy()):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')

                self.progress_bar.inc()


    def test(self):
        self.model.eval()

        t = time()

        print("get test start")

        for step, batch in enumerate(self.dataset_val):

            # Feed input to the model
            time_gap, event_type, length = map(lambda x: x.to(self.cnf.device), batch)
            length = length.cpu().numpy()

            nll, log_hazard_function, cumsum_hazard_function, _ = self.model.forward(time_gap, event_type)

            self.val_nll_loss += nll.item()
            self.val_log_hazard_loss += log_hazard_function.item()
            self.val_cumsum_hazard_loss += cumsum_hazard_function.item()

        if self.best_test_loss is None or self.best_test_loss > np.mean(self.val_nll_loss / length):
            self.best_test_loss = np.mean(self.val_nll_loss / length)
            print(f'\t \t \t \t ● Save Best model at the lowest loss')
            torch.save(self.model.state_dict(), self.log_path / f"{self.cnf.exp_name}_best.pth")

        print(f"\n")
        print(f'\t● AVG Log-Likelihood on TEST-set: {np.mean(self.val_nll_loss / length):.6f}                 │ T: {time() - t:.2f} s')
        print(f'\t● AVG Log-Hazard on TEST-set: {np.mean(self.log_hazard_loss / length):.6f}                  │ T: {time() - t:.2f} s')
        print(f'\t● AVG Cumsum-Hazard on TEST-set: {np.mean(self.cumsum_hazard_loss / length):.6f}            │ T: {time() - t:.2f} s')
        print(f"\n")



    def run(self):

        for _ in range(self.epoch, self.cnf.epochs): #self.cnf.epochs
            self.train()
            #self.scheduler.step()

            self.test()

            self.epoch += 1
            self.save_model()

    def save_model(self):

        save_data = {
            "epoch" : self.epoch,
            "model" : self.model.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "Loss" :  self.best_test_loss
        }

        torch.save(save_data, self.log_path / f"epoch_{self.epoch}_training.ck")

    def load_ck(self, epoch):
        """
        load training checkpoint
        if you need ck, then used that
        """
        ck_path = self.log_path / f'epoch_{epoch}_training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.best_test_loss = self.best_test_loss



@click.command()
@click.option("--conf_file_path", type = str, default = ".//conf//test.yaml")
def main(conf_file_path):
    click.echo("[INFO] It just test result of Dataset")
    cnf = Conf(conf_file_path=conf_file_path, seed=15, exp_name="test", log=True)
    a = Trainer(cnf)

    a.run()





if __name__ == '__main__':
    main()