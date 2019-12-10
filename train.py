import importlib
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchcontrib

import BraTS
from datasets.hdf5 import get_brats_train_loaders
from unet3d.config import load_config
from unet3d.losses import get_loss_criterion
from unet3d.metrics import get_evaluation_metric
from unet3d.model import get_model
# from unet3d.trainer import UNet3DTrainer
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters
from  preprocess.lookahead import Lookahead

from MedicalNet.model import generate_model
from MedicalNet import config as MedConfig


def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, logger):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)

    if resume is not None:
        # continue training from a given checkpoint
        return UNet3DTrainer.from_checkpoint(resume, model,
                                             optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, loaders,
                                             logger=logger)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return UNet3DTrainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, device=config['device'], loaders=loaders,
                                             max_num_epochs=trainer_config['epochs'],
                                             max_num_iterations=trainer_config['iters'],
                                             validate_after_iters=trainer_config['validate_after_iters'],
                                             log_after_iters=trainer_config['log_after_iters'],
                                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                                             logger=logger)
    else:
        # start training from scratch
        return UNet3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                             config['device'], loaders, trainer_config['checkpoint_dir'],
                             max_num_epochs=trainer_config['epochs'],
                             max_num_iterations=trainer_config['iters'],
                             validate_after_iters=trainer_config['validate_after_iters'],
                             log_after_iters=trainer_config['log_after_iters'],
                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                             model_name=config['model']['name'],
                             validate_iters=66,
                             logger=logger)


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    if optimizer_config['mode'] == 'Adam_SE':
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        model_dict = model.state_dict()
        dict_name = list(model_dict)
        for i, p in enumerate(dict_name):
            print(i, p)
        for i, p in enumerate(model.parameters()):
            if i < 65 and i > 43:
                p.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_config['mode'] == 'Adam':
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        model_dict = model.state_dict()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_config['mode'] == 'look_ahead':
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        base_opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
        optimizer = Lookahead(base_opt, k=5, alpha=0.5)
    elif optimizer_config['mode'] == 'SWA':
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        momentum = optimizer_config['momentum']
        base_opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        optimizer = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    elif optimizer_config['mode'] == 'SGD':
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        momentum = 0.9
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    return optimizer


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)


def main():
    # Load and log experiment configuration
    config = load_config()

    # Create main logger
    logger = get_logger('UNet3DTrainer', file_name=config['trainer']['checkpoint_dir'])
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    model = get_model(config)
    # model, parameters = generate_model(MedConfig)

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_brats_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders,
                              logger=logger)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
