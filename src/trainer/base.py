from abc import ABC, abstractmethod
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class Base(ABC):
    """
    base trainer class
    """

    def __init__(self, model, optimizer, lr_scheduler, dataloader, writer, epoch,
                 save_freq, device, criterion):
        super().__init__()

        # self.model = model.to(device)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader

        self.writer = writer
        self.epoch = epoch
        self.save_freq = save_freq
        self.iter = 0
        self.device = device
        self.criterion = criterion


    @abstractmethod
    def _train_epoch(self):
        pass

    def train(self):
        for i in tqdm(range(self.epoch)):
            res = self._train_epoch()

            if (i + 1) % self.save_freq == 0:
                self._save_epoch()
            self.lr_scheduler.step()


    def _save_model(self):
        pass

    def _load_model(self):
        pass
