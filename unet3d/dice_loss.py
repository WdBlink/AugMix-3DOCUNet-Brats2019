"""
File: dice_loss.py
Author: VDeamoV
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    fork from https://github.com/hubutui/DiceLoss-PyTorch.git
    add ohem funciton
    1. define each classes' weight
    2. use ohem or every class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean', ohem_ratio=None):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.ohem_ratio = ohem_ratio

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)


        # add ohem here
        if self.ohem_ratio:
            assert len(predict.view(-1)) == len(target.view(-1))
            num = torch.mul(predict, target) + self.smooth/len(predict.view(-1))
            den = predict.pow(self.p) + target.pow(self.p) + self.smooth/len(predict.view(-1))
            loss_flatten = 1 - num/den # caclutate 1 - num/den for each pixel

            # cacluate the mask
            with torch.no_grad():
                values, _ = torch.topk(loss_flatten.reshape(-1),
                                       int(loss_flatten.nelement()*self.ohem_ratio))
                loss_mask = loss_flatten >= values[-1]

            loss_flatten = loss_flatten * loss_mask.type(dtype=torch.float)
            loss = loss_flatten.mean()
        else:
            num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
            den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
            loss = 1 - num / den


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def test_binary_dice_loss():
    """
    test binary loss
    """
    A = torch.zeros((3, 10, 10))
    A[:,:,:3] = 1
    B = torch.zeros((3, 10, 10))
    B[:,:,:3] = 1
    print(f"A.shape = {A.shape} B.shape = {B.shape}")
    dl = BinaryDiceLoss(reduction='mean', ohem_ratio=0.2)
    loss = dl(A, B)
    print(loss)


def test_dice_loss():
    """
    test dice loss
    """
    A = torch.zeros((1, 3, 256, 256))
    B = torch.ones((1, 3, 256, 256))
    print(f"A.shape = {A.shape} B.shape = {B.shape}")
    dl = DiceLoss(reduction='mean', ohem_ratio=0.5)
    loss = dl(A, B)
    print(loss)

if __name__ == "__main__":
    test_binary_dice_loss()
    # test_dice_loss()
