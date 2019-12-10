import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torch import Tensor, einsum
from unet3d.utils import simplex
from torch.nn.modules.loss import _Loss


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    seg_pred = torch.reshape(input[0], [4, -1])
    seg_true = torch.reshape(target[0], [4, -1])
    # seg_pred = seg_pred.to(dtype=torch.float64)
    # seg_true = seg_true.to(dtype=torch.float64)

    seg_true = seg_true[:, 1:].to(dtype=torch.float32)
    seg_pred = seg_pred[:, 1:].to(dtype=torch.float32)
    # target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (seg_pred * seg_true).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (seg_pred + seg_true).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1 - dice

        return dice_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''

    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(
            torch.log(torch.mul(std, std))) - 1


class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''

    def __init__(self, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = BratsDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, vae_truth, y_pred, y_mid, y_true):
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred, seg_truth = (y_pred[:, :3, :, :, :], y_true)
        vae_pred, vae_truth = (y_pred[:, 3:, :, :, :], vae_truth)
        dice_loss = self.dice_loss(seg_pred, seg_truth)
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        # print("dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:%.4f"%(dice_loss,l2_loss,kl_div,combined_loss))

        return combined_loss


class BratsDiceLoss(nn.Module):
    """Dice loss of Brats dataset
    Args:
        outputs: A tensor of shape [N, *]
        labels: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, nonSquared=False, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(BratsDiceLoss, self).__init__()
        self.nonSquared = nonSquared

    def forward(self, outputs, labels):
        # bring outputs into correct shape
        wt, tc, et = outputs.chunk(3, dim=1)
        s = wt.shape
        wt = wt.view(s[0], s[2], s[3], s[4])
        tc = tc.view(s[0], s[2], s[3], s[4])
        et = et.view(s[0], s[2], s[3], s[4])

        # bring masks into correct shape
        wtMask, tcMask, etMask = labels.chunk(3, dim=1)
        s = wtMask.shape
        wtMask = wtMask.view(s[0], s[2], s[3], s[4])
        tcMask = tcMask.view(s[0], s[2], s[3], s[4])
        etMask = etMask.view(s[0], s[2], s[3], s[4])

        # calculate losses
        wtLoss = self.weightedDiceLoss(wt, wtMask, mean=0.05)
        tcLoss = self.weightedDiceLoss(tc, tcMask, mean=0.02)
        etLoss = self.weightedDiceLoss(et, etMask, mean=0.01)

        return (wtLoss + tcLoss + etLoss) / 5

    def diceLoss(self, pred, target, nonSquared=False):
        return 1 - self.softDice(pred, target, nonSquared=nonSquared)

    def weightedDiceLoss(self, pred, target, smoothing=1, mean=0.01):

        mean = mean
        w_1 = 1 / mean ** 2
        w_0 = 1 / (1 - mean) ** 2

        pred_1 = pred
        target_1 = target
        pred_0 = 1 - pred
        target_0 = 1 - target

        intersection_1 = (pred_1 * target_1).sum(dim=(1, 2, 3))
        intersection_0 = (pred_0 * target_0).sum(dim=(1, 2, 3))
        intersection = w_0 * intersection_0 + w_1 * intersection_1

        union_1 = (pred_1).sum() + (target_1).sum()
        union_0 = (pred_0).sum() + (target_0).sum()
        union = w_0 * union_0 + w_1 * union_1

        dice = (2 * intersection + smoothing) / (union + smoothing)
        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])
        return 1 - dice.mean()

    def softDice(self, pred, target, smoothing=1, nonSquared=False):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        if nonSquared:
            union = (pred).sum() + (target).sum()
        else:
            union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + smoothing) / (union + smoothing)

        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])

        return dice.mean()


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, output, target):
        class_loss = (target * F.relu(0.9 - output) + 0.5 * (1 - target) * F.relu(output - 0.1)).mean()
        return class_loss


class ReconstructLoss(nn.Module):
    def __init__(self):
        super(ReconstructLoss, self).__init__()

    def forward(self, *input):
        a = input[0]
        recon_a = input[1]
        loss = F.mse_loss(a[:, 3:4, :, :, :], recon_a)
        return loss, 0


class TwoClassLoss(nn.Module):
    """Dice loss of Brats dataset
        Args:
            outputs: A tensor of shape [N, *]
            labels: A tensor of shape same with predict
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """

    def __init__(self, nonSquared=False, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(TwoClassLoss, self).__init__()
        self.nonSquared = nonSquared

    def forward(self, outputs, labels):
        # bring outputs into correct shape
        wt = outputs
        s = wt.shape
        wt = wt.view(s[0], s[2], s[3], s[4])

        # bring masks into correct shape
        wtMask = labels
        s = wtMask.shape
        wtMask = wtMask.view(s[0], s[2], s[3], s[4])

        # calculate losses
        wtLoss = self.weightedDiceLoss(wt, wtMask, mean=0.05)

        return wtLoss / 5

    def diceLoss(self, pred, target, nonSquared=False):
        return 1 - self.softDice(pred, target, nonSquared=nonSquared)

    def weightedDiceLoss(self, pred, target, smoothing=1, mean=0.01):

        mean = mean
        w_1 = 1 / mean ** 2
        w_0 = 1 / (1 - mean) ** 2

        pred_1 = pred
        target_1 = target
        pred_0 = 1 - pred
        target_0 = 1 - target

        intersection_1 = (pred_1 * target_1).sum(dim=(1, 2, 3))
        intersection_0 = (pred_0 * target_0).sum(dim=(1, 2, 3))
        intersection = w_0 * intersection_0 + w_1 * intersection_1

        union_1 = (pred_1).sum() + (target_1).sum()
        union_0 = (pred_0).sum() + (target_0).sum()
        union = w_0 * union_0 + w_1 * union_1

        dice = (2 * intersection + smoothing) / (union + smoothing)
        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])
        return 1 - dice.mean()

    def softDice(self, pred, target, smoothing=1, nonSquared=False):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        if nonSquared:
            union = (pred).sum() + (target).sum()
        else:
            union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + smoothing) / (union + smoothing)

        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])

        return dice.mean()


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
            num = torch.mul(predict, target)
            den = predict.pow(self.p) + target.pow(self.p)
            with torch.no_grad():
                num_values, _ = torch.topk(num.reshape(-1),
                                           int(num.nelement()*self.ohem_ratio))
                den_values, _ = torch.topk(den.reshape(-1),
                                           int(den.nelement()*self.ohem_ratio))
                num_mask = num >= num_values[-1]
                den_mask = den >= den_values[-1]
            num = torch.sum(num * num_mask.type(dtype=torch.float), dim=1) + self.smooth
            den = torch.sum(den * den_mask.type(dtype=torch.float), dim=1) + self.smooth
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
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)


class SurfaceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SurfaceLoss, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc = 1
        self.normalization = nn.Softmax(dim=1)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        probs = self.normalization(probs)
        assert simplex(probs)
        # assert not one_hot(dist_maps)

        pc = probs[:, :, ...].type(torch.float32)
        dc = dist_maps[:, :, ...].type(torch.float32)

        multipled = einsum("bcwhd,bcwhd->bcwhd", pc, dc)

        loss = multipled.mean()

        return loss


class DiceLoss_SurfaceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss_SurfaceLoss, self).__init__()
        self.surface_loss = SurfaceLoss()
        self.dice_loss = BratsDiceLoss()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        surface_loss = self.surface_loss(input, target)

        # get probabilities from logits
        # input = self.normalization(input)
        dice_loss = self.dice_loss(input, target)
        # Average the Dice score across all channels/classes
        return dice_loss + surface_loss


class OhemDiceLoss(nn.Module):
    def __init__(self, ohem_ratio=0.7 , weight=None, ignore_index=-100,
                 eps=1e-7):
        super(OhemDiceLoss, self).__init__()
        self.ignore_label = ignore_index
        self.criterion = DiceLoss()
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = self._ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def _ohem_mask(self, loss, ohem_ratio):
        with torch.no_grad():
            values, _ = torch.topk(loss.reshape(-1),
                                   int(loss.nelement() * ohem_ratio))
            mask = loss >= values[-1]
        return mask.float()


class VaeLoss(nn.Module):
    """
    loss(input_shape, inp, out_VAE, z_mean, z_var, e=1e-8, weight_L2=0.1, weight_KL=0.1)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper, involving
    outputs from multiple layers.

    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>

    - L<dice> is the dice loss between input and segmentation output.
    - L<L2> is the L2 loss between the output of VAE part and the input.
    - L<KL> is the standard KL divergence loss term for the VAE.

    Parameters
    ----------
    `input_shape`: A 4-tuple, required
        The shape of an image as the tuple (c, H, W, D), where c is
        the no. of channels; H, W and D is the height, width and depth of the
        input image, respectively.
    `inp`: An keras.layers.Layer instance, required
        The input layer of the model. Used internally.
    `out_VAE`: An keras.layers.Layer instance, required
        The output of VAE part of the decoder. Used internally.
    `z_mean`: An keras.layers.Layer instance, required
        The vector representing values of mean for the learned distribution
        in the VAE part. Used internally.
    `z_var`: An keras.layers.Layer instance, required
        The vector representing values of variance for the learned distribution
        in the VAE part. Used internally.
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.

    Returns
    -------
    loss_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss. Combined with the L<KL> and L<L2 computed
        earlier, it returns the total loss.
    """
    def __init__(self, weight_L2=0.1, weight_KL=0.1):
        super(VaeLoss, self).__init__()
        self.boundary_loss = SurfaceLoss()
        self.dice_loss = BratsDiceLoss()
        self.weight_L2 = weight_L2
        self.weight_KL = weight_KL

    def forward(self, inp, vae_out, unet_out, label, z_mean, z_var):
        _, c, h, w, d = inp.size()
        n = h * w * d

        loss_L2 = torch.mean(torch.pow(inp - vae_out, 2))
        loss_KL = (1/n) * torch.sum(
            torch.exp(z_var) + torch.pow(z_mean, 2) - 1. - z_var)

        boundary_loss = self.boundary_loss(unet_out, label)
        dice_loss = self.dice_loss(unet_out, label)

        return boundary_loss + dice_loss + self.weight_L2 * loss_L2 + self.weight_KL * loss_KL


def focalLoss(outputs, labels):

    alpha = 0.1
    gamma = 2.0
    pt_1 = torch.where(torch.eq(labels, 1), outputs, torch.ones_like(outputs))
    pt_0 = torch.where(torch.eq(labels, 0), outputs, torch.zeros_like(outputs))
    pt_1 = torch.clamp(pt_1, 1e-3, .999)
    pt_0 = torch.clamp(pt_0, 1e-3, .999)

    # return -torch.sum(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1))\
    # -torch.sum((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))

    return -torch.sum(torch.log(pt_1)) - torch.sum(torch.pow(pt_0, gamma) * torch.log(1. - pt_0))


class bratsFocalLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(bratsFocalLoss, self).__init__()

    def forward(self, outputs, labels):
        wt, tc, et = outputs.chunk(3, dim=1)
        s = wt.shape
        wt = wt.view(s[0], s[2], s[3], s[4])
        tc = tc.view(s[0], s[2], s[3], s[4])
        et = et.view(s[0], s[2], s[3], s[4])

        # bring masks into correct shape
        wtMask, tcMask, etMask = labels.chunk(3, dim=1)
        s = wtMask.shape
        wtMask = wtMask.view(s[0], s[2], s[3], s[4])
        tcMask = tcMask.view(s[0], s[2], s[3], s[4])
        etMask = etMask.view(s[0], s[2], s[3], s[4])

        wtloss = focalLoss(wt, wtMask)
        tcloss = focalLoss(tc, tcMask)
        etloss = focalLoss(et, etMask)

        return (wtloss + tcloss + etloss) / 10


class bratsMixedLoss(nn.Module):
    def __init__(self, alpha=0.00001, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(bratsMixedLoss, self).__init__()
        self.dice_loss = BratsDiceLoss()
        self.focal_loss = bratsFocalLoss()
        self.alpha = alpha
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, outputs, labels):
        focal_loss = self.focal_loss(outputs, labels)
        dice_loss = self.dice_loss(outputs, labels)
        return self.alpha*focal_loss+dice_loss


class CaeLoss(nn.Module):
    """
    loss(input_shape, inp, out_VAE, z_mean, z_var, e=1e-8, weight_L2=0.1, weight_KL=0.1)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper, involving
    outputs from multiple layers.

    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>

    - L<dice> is the dice loss between input and segmentation output.
    - L<L2> is the L2 loss between the output of VAE part and the input.
    - L<KL> is the standard KL divergence loss term for the VAE.

    Parameters
    ----------
    `input_shape`: A 4-tuple, required
        The shape of an image as the tuple (c, H, W, D), where c is
        the no. of channels; H, W and D is the height, width and depth of the
        input image, respectively.
    `inp`: An keras.layers.Layer instance, required
        The input layer of the model. Used internally.
    `out_VAE`: An keras.layers.Layer instance, required
        The output of VAE part of the decoder. Used internally.
    `z_mean`: An keras.layers.Layer instance, required
        The vector representing values of mean for the learned distribution
        in the VAE part. Used internally.
    `z_var`: An keras.layers.Layer instance, required
        The vector representing values of variance for the learned distribution
        in the VAE part. Used internally.
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.

    Returns
    -------
    loss_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss. Combined with the L<KL> and L<L2 computed
        earlier, it returns the total loss.
    """
    def __init__(self, weight=0.1):
        super(CaeLoss, self).__init__()
        self.boundary_loss = SurfaceLoss()
        self.dice_loss = BratsDiceLoss()
        self.cae_loss = nn.MSELoss()

    def forward(self, inp, label, unet_out, cae_out):
        # cae_loss = torch.mean(torch.pow(inp - cae_out, 2))
        cae_loss = self.cae_loss(inp[:, 2:3, :, :, :], cae_out)
        boundary_loss = self.boundary_loss(unet_out, label)

        dice_loss = self.dice_loss(unet_out, label)

        return dice_loss + cae_loss + boundary_loss


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        class_weights = self._class_weights(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, _stacklevel=5)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class BCELossWrapper:
    """
    Wrapper around BCE loss functions allowing to pass 'ignore_index' as well as 'skip_last_target' option.
    """

    def __init__(self, loss_criterion, ignore_index=-1, skip_last_target=False):
        if hasattr(loss_criterion, 'ignore_index'):
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index
        self.skip_last_target = skip_last_target

    def __call__(self, input, target):
        if self.skip_last_target:
            target = target[:, :-1, ...]

        assert input.size() == target.size()

        masked_input = input
        masked_target = target
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            masked_input = input * mask
            masked_target = target * mask

        return self.loss_criterion(masked_input, masked_target)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            log_probabilities = log_probabilities * mask
            target = target * mask

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
            self.register_buffer('class_weights', class_weights)

        # resize class_weights to be broadcastable into the weights
        class_weights = self.class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class MSEWithLogitsLoss(MSELoss):
    """
    This loss combines a `Sigmoid` layer and the `MSELoss` in one single class.
    """

    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, target):
        return super().forward(self.sigmoid(input), target)


class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        # if there is just one output head the 'inputs' is going to be a singleton list [tensor]
        # and 'targets' is just going to be a tensor (that's how the HDF5Dataloader works)
        # so wrap targets in a list in this case
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)

        return loss


def square_angular_loss(input, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.

    :param input: 5D input tensor (NCDHW)
    :param target: 5D target tensor (NCDHW)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses
    """
    assert input.size() == target.size()
    # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
    stability_coeff = 0.999999
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    # compute cosine map
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    # expand the input tensor to Nx1xDxHxW
    src = input.unsqueeze(0)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        src = src.clone()
        src[src == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, src, 1)


SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss', 'PixelWiseCrossEntropyLoss',
                    'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSEWithLogitsLoss', 'MSELoss',
                    'SmoothL1Loss', 'L1Loss']


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config['name']

    ignore_index = loss_config.get('ignore_index', None)
    weight = loss_config.get('weight', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    if name == 'BCEWithLogitsLoss':
        skip_last_target = loss_config.get('skip_last_target', False)
        if ignore_index is None and not skip_last_target:
            return nn.BCEWithLogitsLoss()
        else:
            return BCELossWrapper(nn.BCEWithLogitsLoss(), ignore_index=ignore_index, skip_last_target=skip_last_target)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        return GeneralizedDiceLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'DiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        skip_last_target = loss_config.get('skip_last_target', False)
        return DiceLoss(weight=weight, ignore_index=ignore_index, sigmoid_normalization=sigmoid_normalization,
                        skip_last_target=skip_last_target)
    elif name == 'TagsAngularLoss':
        tags_coefficients = loss_config['tags_coefficients']
        return TagsAngularLoss(tags_coefficients)
    elif name == 'MSEWithLogitsLoss':
        return MSEWithLogitsLoss()
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'VaeLoss':
        return VaeLoss()
    elif name == 'OhemDiceLoss':
        return OhemDiceLoss()
    elif name == 'DiceLoss_SurfaceLoss':
        return DiceLoss_SurfaceLoss()
    elif name == 'BinaryDiceLoss':
        return BinaryDiceLoss()
    elif name == 'BratsDiceLoss':
        return BratsDiceLoss()
    elif name == 'bratsMixedLoss':
        return bratsMixedLoss()
    elif name == 'CaeLoss':
        return CaeLoss()
    elif name == 'CombinedLoss':
        return CombinedLoss()
    elif name == 'TwoClassLoss':
        return TwoClassLoss()
    elif name == 'CapsuleLoss':
        return CapsuleLoss()
    elif name == 'ReconstructLoss':
        return ReconstructLoss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")



