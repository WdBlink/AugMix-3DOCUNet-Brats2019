import logging
import os

import numpy as np
import torch
import datetime
from unet3d.config import load_config
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from visualization import board_add_images, board_add_image

from . import utils

config = load_config()


class UNet3DTrainer:
    """3D UNet trainer.
    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir, model_name,
                 max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=0, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 logger=None):
        # if logger is None:
        #     self.logger = utils.get_logger('VaeUnetTrainer', level=logging.DEBUG)
        # else:
        #     self.logger = logger
        self.logger = logger
        self.config = load_config()
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(logdir=os.path.join(checkpoint_dir, self._get_job_name()))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    def _get_job_name(self):
        now = '{:%Y-%m-%d.%H:%M}'.format(datetime.datetime.now())
        return "%s_model_%s" % (now, self.model_name)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        logger=None):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. "
            f"Best val score: {state['best_eval_score']}. "
            f"Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   model_name=config['model']['name'],
                   logger=logger)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        logger=None):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_checkpoint(pre_trained, model, None)
        # checkpoint_dir = os.path.split(pre_trained)[0]
        checkpoint_dir = config['trainer']['checkpoint_dir']
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   model_name=config['model']['name'],
                   logger=logger)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break
            if self.config['optimizer']['mode'] == 'SWA':
                self.optimizer.swap_swa_sgd()
            self.num_epoch += 1

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores_multi = utils.RunningAverageMulti()
        train_eval_scores_duality = utils.RunningAverageDuality()

        # make predictions
        # self.makePredictions(self.loaders['challenge'])

        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(tqdm(train_loader)):

            print(
                f'Training iteration {self.num_iterations}. '
                f'Batch {i}. '
                f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            input, pid, target = self._split_training_batch(t)

            output, loss, feature_maps = self._forward_pass(input, target, weight=None)

            # output_sample = output[0, 1, :, :, 80].cpu().detach().numpy()
            # self.draw_picture(output_sample)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                # adjust learning rate if necessary
                # if isinstance(self.scheduler, ReduceLROnPlateau):
                    # self.scheduler.step(eval_score)
                if self.scheduler is None:
                    pass
                else:
                    self.scheduler.step()
                    # pass

                # log current learning rate in tensorboard
                self._log_lr()

                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

                # make challenge predict
                if is_best:
                    pass
                    # self.makePredictions(self.loaders['challenge'])

            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                if hasattr(self.model, 'final_activation'):
                    output = self.model.final_activation(output)

                # visualize the feature map to tensorboard
                board_list = [input[0:1, 1:4, :, :, input.size(4)//2], output[0:1, :, :, :, input.size(4)//2], target[0:1, :, :, :, input.size(4)//2]]
                board_add_images(self.writer, 'train_output', board_list, self.num_iterations)
                if self.model_name == 'NNNet_Cae':
                    for i, t in enumerate(feature_maps):
                        board_add_image(self.writer, f'feature_map{i}', t, self.num_iterations)
                    # board_add_images(self.writer, 'feature_map', feature_maps, self.num_iterations)

                # compute eval criterion
                eval_score = self.eval_criterion(output, target)

                train_eval_scores_duality.update(eval_score, self._batch_size(input))

                # log stats, params and images
                self.logger.info(
                    f'Training stats.\n'
                    f'Train_Loss: {train_losses.avg}. \n'
                    f'Train_WT:{train_eval_scores_multi.dice_WT}, \n')

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += config['loaders']['batch_size']

        # evaluate on validation set
        eval_score = self.validate(self.loaders['val'])
        # adjust learning rate if necessary
        # if isinstance(self.scheduler, ReduceLROnPlateau):
        # self.scheduler.step(eval_score)
        if self.scheduler is None:
            pass
        else:
            self.scheduler.step()
            # pass

        # log current learning rate in tensorboard
        self._log_lr()

        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)

        # save checkpoint
        self._save_checkpoint(is_best)

        # make challenge predict
        if eval_score > 0.90:
            self.makePredictions(self.loaders['challenge'])

        return False

    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        # val_scores = utils.RunningAverage()
        val_scores_duality = utils.RunningAverageDuality()

        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            self.model.eval()
            with torch.no_grad():
                for i, t in enumerate(val_loader):
                    self.logger.info(f'Validation iteration {i}')

                    input, pid, target = self._split_training_batch(t)

                    output, loss, feature_map = self._forward_pass(input, target, mode='val', weight=None)
                    val_losses.update(loss.item(), self._batch_size(input))

                    eval_score = self.eval_criterion(output, target)

                    # print the bad guy
                    if eval_score[0] < 0.5:
                        wt_gt = (target[:, 0, ...] == 1).sum()
                        # tc_gt = (target[:, 1, ...] == 1).sum()
                        # et_gt = (target[:, 2, ...] == 1).sum()

                        wt_pred = (output[:, 0, ...] >= 0.5).sum()
                        # tc_pred = (output[:, 1, ...] >= 0.5).sum()
                        # et_pred = (output[:, 2, ...] >= 0.5).sum()
                        self.logger.info(f'The patient {pid} score is {eval_score}!!!\n'
                                         f'The pixel of WT_GT|WT_OUT is {wt_gt}|{wt_pred}\n')


                    # val_scores.update(eval_score.item(), self._batch_size(input))
                    val_scores_duality.update(eval_score, self._batch_size(input))

                    # visualize the feature map to tensorboard
                    board_list = [input[0:1, 1:4, :, :, input.size(4)//2], output[0:1, :, :, :, output.size(4)//2],
                                  target[0:1, :, :, :, target.size(4)//2]]
                    board_add_images(self.writer, 'validate_output', board_list, self.num_iterations)

                    if self.validate_iters is not None and self.validate_iters <= i:
                        # stop validation
                        break

                self.logger.info(f'Validation finished. \n'
                                 f'Loss: {val_losses.avg} \n'
                                 f'Evaluation score \n'
                                 f'Val_WT:{val_scores_duality.dice_WT} \n'
                                 f'Val_sensitivity WT:{val_scores_duality.sens_WT}')

                return val_scores_duality.dice_WT
        finally:
            # set back in training mode
            self.model.train()

    def makePredictions(self, challenge_loader):
        # model is already loaded from disk by constructor
        basePath = os.path.join(config['trainer']['checkpoint_dir'], "epoch{}".format(self.num_epoch+1))
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(challenge_loader)):
                inputs, pids, xOffset, yOffset, zOffset = data
                print("processing {}".format(pids[0]))
                inputs = inputs.to(config['device'])

                # predict labels and bring into required shape
                outputs = self.model(inputs)
                # TTA
                outputs += self.model(inputs.flip(dims=(2,))).flip(dims=(2,))
                outputs += self.model(inputs.flip(dims=(3,))).flip(dims=(3,))
                outputs += self.model(inputs.flip(dims=(4,))).flip(dims=(4,))
                outputs += self.model(inputs.flip(dims=(2, 3))).flip(dims=(2, 3))
                outputs += self.model(inputs.flip(dims=(2, 4))).flip(dims=(2, 4))
                outputs += self.model(inputs.flip(dims=(3, 4))).flip(dims=(3, 4))
                outputs += self.model(inputs.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4))
                outputs = outputs / 8.0  # mean

                outputs = outputs[:, :, :, :, :155]
                s = outputs.shape
                fullsize = outputs.new_zeros((s[0], s[1], 240, 240, 155))
                if xOffset + s[2] > 240:
                    outputs = outputs[:, :, :240 - xOffset, :, :]
                if yOffset + s[3] > 240:
                    outputs = outputs[:, :, :, :240 - yOffset, :]
                if zOffset + s[4] > 155:
                    outputs = outputs[:, :, :, :, :155 - zOffset]
                fullsize[:, :, xOffset:xOffset + s[2], yOffset:yOffset + s[3], zOffset:zOffset + s[4]] = outputs

                # binarize output
                wt, tc, et = fullsize.chunk(3, dim=1)
                s = fullsize.shape
                wt = (wt > 0.6).view(s[2], s[3], s[4])
                tc = (tc > 0.5).view(s[2], s[3], s[4])
                et = (et > 0.7).view(s[2], s[3], s[4])

                result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
                result[wt] = 2
                result[tc] = 1
                result[et] = 4

                npResult = result.cpu().numpy()
                WT_voxels = (npResult == 4).sum()
                if WT_voxels < 100:
                    # torch.where(result == 4, result, torch.ones_like(result))
                    npResult[np.where(npResult == 4)] = 1

                path = os.path.join(basePath, "{}.nii.gz".format(pids[0]))
                utils.save_nii(path, npResult, None, None)

        print("Done :)")

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):

                return tuple([_move_to_device(input[0]), input[1], _move_to_device(input[2])])
            else:
                return input.to(self.device, dtype=torch.float)

        t = _move_to_device(t)
        if len(t) == 2:
            input, target = t
        else:
            input, pid, target = t
        return input, pid, target

    def _forward_pass(self, input, target, mode='train', weight=None):
        # forward pass
        output = self.model(input)
        feature_maps = []
        # compute the loss
        if self.model_name == 'NvNet':
            loss = self.loss_criterion(input, output[0], output[1], target)
            output = output[0][:, :3, :, :, :]
        elif self.model_name == 'NNNet_Cae':
            loss = self.loss_criterion(input, target, output[0], output[1])
            if mode == 'train':
                cae_out = output[1]
                for i, t in enumerate(output[2]):
                    size = t.size(4)
                    channel = t.size(1)
                    t = torch.sum(t, dim=1, keepdim=True)//channel
                    feature_maps.append(t[:, :, :, :, size//2])
                feature_maps.append(cae_out[:, :, :, :, cae_out.size(4)//2])
            output = output[0]
        else:
            if weight is None:
                loss = self.loss_criterion(output, target)
            else:
                loss = self.loss_criterion(output, target, weight)

        return output, loss, feature_maps

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)
        last_file_path = os.path.join(self.checkpoint_dir, f'epoch{self.num_epoch+1}_model.pth')
        torch.save(self.model, last_file_path)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_stats_multi(self, phase, loss_avg, eval_score_avg1, eval_score_avg2, eval_score_avg3, eval_score_avg4, eval_score_avg5, eval_score_avg6):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg1': eval_score_avg1,
            f'{phase}_eval_score_avg2': eval_score_avg2,
            f'{phase}_eval_score_avg3': eval_score_avg3,
            f'{phase}_sensitivity_WT': eval_score_avg4,
            f'{phase}_sensitivity_TC': eval_score_avg5,
            f'{phase}_sensitivity_ET': eval_score_avg6
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '_grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')

    def _images_from_batch(self, name, batch):
        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCHWD
            slice_idx = batch.shape[4] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NHWD
            slice_idx = batch.shape[3] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return (img - np.min(img)) / np.ptp(img)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
