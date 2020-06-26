import hydra
import os
from pathlib import Path
import mlflow

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from networks import BT as net
from datasets import BT as data
import trainer
# from utils.mlflow_writer import MlflowWriter


@hydra.main(config_path='../config/config.yaml')
def main(cfg):
    """
    train model file
    first Train autoencoder
    next train model
    """

    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment('BT_DeepSVDD')

    with mlflow.start_run() as run:
        # tensorboard writer
        writer = SummaryWriter(log_dir=os.getcwd())

        # define dataset for autoencoder
        dataset_dict = dict(cfg.dataset.args)
        dataset_dict['root'] = Path(hydra.utils.get_original_cwd() + cfg.dataset.args.root).resolve()
        dataset = getattr(data, cfg.dataset.name)(**dataset_dict)
        dataloader= DataLoader(dataset,**cfg.dataloader.args)


        ##############################
        #        autoencoder         #
        ##############################

        # define auto encoder for pretrain
        autoencoder = getattr(net,cfg.networks.autoencoder.name) \
            (cfg.networks.base.args,**cfg.networks.autoencoder.args)

        # define optimizer for autoencoder
        ae_optimizer = getattr(optim, cfg.ae_optimizer.name) \
            (autoencoder.parameters(),**cfg.ae_optimizer.args)

        # define lr scheduler for autoencoder
        ae_lr_scheduler = getattr(lr_scheduler, cfg.ae_lr_scheduler.name) \
            (ae_optimizer, **cfg.ae_lr_scheduler.args)

        # ae criterion
        ae_criterion = getattr(torch.nn, cfg.ae_criterion.name)()

        # autoendoer train
        ae_trainer = getattr(trainer, cfg.ae_trainer.name)\
            (model = autoencoder, optimizer=ae_optimizer, lr_scheduler=ae_lr_scheduler,
             dataloader=dataloader, writer=writer, criterion=ae_criterion,**cfg.ae_trainer.args)
        ae_trainer.train()

        ##############################
        #        main model          #
        ##############################

        # model
        model = autoencoder.encoder
        # model = model.to(cfg.Trainer.args.device)

        # define optimizer
        optimizer = getattr(optim, cfg.optimizer.name) \
            (autoencoder.parameters(),**cfg.optimizer.args)

        # define lr scheduler
        main_lr_scheduler = getattr(lr_scheduler, cfg.lr_scheduler.name) \
            (optimizer, **cfg.lr_scheduler.args)

        main_trainer = getattr(trainer, cfg.Trainer.name)\
            (model=model, optimizer=optimizer, lr_scheduler=main_lr_scheduler,
             dataloader=dataloader, writer=writer, criterion=None,**cfg.Trainer.args)
        main_trainer.train()

        mlflow.log_artifact(Path.cwd() / '.hydra/config.yaml')


if __name__ == '__main__':
    main()
