from pathlib import Path
import torch
import logging
import random
import numpy as np
import hydra
import os

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


@hydra.main(config_path='../config/config.yaml')
def main(cfg):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    # cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = 'output' + '/log.txt'
    log_file = os.getcwd() + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % cfg.data_path)
    logger.info('Export path is %s.' % cfg.xp_path)

    logger.info('Dataset: %s' % cfg.dataset_name)
    logger.info('Normal class: %d' % cfg.normal_class)
    logger.info('Network: %s' % cfg.net_name)


    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.objective)
    logger.info('Nu-paramerter: %.2f' % cfg.nu)

    # Set seed
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        logger.info('Set seed to %d.' % cfg.seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % cfg.device)
    logger.info('Number of dataloader workers: %d' % cfg.n_jobs_dataloader)

    # Load data
    root = Path(hydra.utils.get_original_cwd() + cfg.data_path).resolve()
    dataset = load_dataset(cfg.dataset_name, root,
                           cfg.normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg=cfg, objective=cfg.objective, nu=cfg.nu)
    deep_SVDD.set_network(cfg.net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if cfg.load_model:
        deep_SVDD.load_model(model_path=cfg.load_model, load_ae=True)
        logger.info('Loading model from %s.' % cfg.load_model)

    logger.info('Pretraining: %s' % cfg.pretrain)
    if cfg.pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.ae_optimizer_name)
        logger.info('Pretraining learning rate: %g' % cfg.ae_lr)
        logger.info('Pretraining epochs: %d' % cfg.ae_n_epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.ae_lr_milestone,))
        logger.info('Pretraining batch size: %d' % cfg.ae_batch_size)
        logger.info('Pretraining weight decay: %g' % cfg.ae_weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.ae_optimizer_name,
                           lr=cfg.ae_lr,
                           n_epochs=cfg.ae_n_epochs,
                           lr_milestones=cfg.ae_lr_milestone,
                           batch_size=cfg.ae_batch_size,
                           weight_decay=cfg.ae_weight_decay,
                           device=cfg.device,
                           n_jobs_dataloader=cfg.n_jobs_dataloader)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.optimizer_name)
    logger.info('Training learning rate: %g' % cfg.lr)
    logger.info('Training epochs: %d' % cfg.n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.lr_milestone,))
    logger.info('Training batch size: %d' % cfg.batch_size)
    logger.info('Training weight decay: %g' % cfg.weight_decay)

    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.optimizer_name,
                    lr=cfg.lr,
                    n_epochs=cfg.n_epochs,
                    lr_milestones=cfg.lr_milestone,
                    batch_size=cfg.batch_size,
                    weight_decay=cfg.weight_decay,
                    device=cfg.device,
                    n_jobs_dataloader=cfg.n_jobs_dataloader)

    # Test model
    deep_SVDD.test(dataset, device=cfg.device, n_jobs_dataloader=cfg.n_jobs_dataloader)

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    if dataset_name in ('mnist', 'cifar10'):

        if dataset_name == 'mnist':
            X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
            X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

        plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
        plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results.json')
    deep_SVDD.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
