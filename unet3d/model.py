import importlib

import torch
import torch.nn as nn
from torch.nn import init

import matplotlib.pyplot as plt
import random

from unet3d.buildingblocks import conv3d, Encoder, Decoder, FinalConv, DoubleConv, \
    ExtResNetBlock, SingleConv, GreenBlock, DownBlock, UpBlock, VaeBlock, CaeBlock, unetUp, unetConv3, \
    EncoderModule, DecoderModule, ResEncoderModule, ResDecoderModule, \
    UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3, GridAttentionBlockND, \
    MedicaNetBasicBlock, MedicaNetBottleneck, SELayer, CapsuleLayer, OctaveConv, Conv_BN, Conv_BN_ACT

from unet3d.utils import create_feature_maps
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable


# initalize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class ResidualUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, conv_layer_order='cge', num_groups=8,
                 **kwargs):
        super(ResidualUNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses ExtResNetBlock as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses ExtResNetBlock as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            decoder = Decoder(reversed_f_maps[i], reversed_f_maps[i + 1], basic_module=ExtResNetBlock,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class Noise2NoiseUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, f_maps=16, num_groups=8, **kwargs):
        super(Noise2NoiseUNet3D, self).__init__()

        # Use LeakyReLU activation everywhere except the last layer
        conv_layer_order = 'clg'

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 1x1x1 conv + simple ReLU in the final convolution
        self.final_conv = SingleConv(f_maps[0], out_channels, kernel_size=1, order='cr', padding=0)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('unet3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


###############################################Supervised Tags 3DUnet###################################################

class TagsUNet3D(nn.Module):
    """
    Supervised tags 3DUnet
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels; since most often we're trying to learn
            3D unit vectors we use 3 as a default value
        output_heads (int): number of output heads from the network, each head corresponds to different
            semantic tag/direction to be learned
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `DoubleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels=3, output_heads=1, conv_layer_order='crg', init_channel_number=32,
                 **kwargs):
        super(TagsUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups)
        ])

        self.final_heads = nn.ModuleList(
            [FinalConv(init_channel_number, out_channels, num_groups=num_groups) for _ in
             range(output_heads)])

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final layer per each output head
        tags = [final_head(x) for final_head in self.final_heads]

        # normalize directions with L2 norm
        return [tag / torch.norm(tag, p=2, dim=1).detach().clamp(min=1e-8) for tag in tags]


################################################Distance transform 3DUNet##############################################
class DistanceTransformUNet3D(nn.Module):
    """
    Predict Distance Transform to the boundary signal based on the output from the Tags3DUnet. Fore training use either:
        1. PixelWiseCrossEntropyLoss if the distance transform is quantized (classification)
        2. MSELoss if the distance transform is continuous (regression)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        final_sigmoid (bool): 'sigmoid'/'softmax' whether element-wise nn.Sigmoid or nn.Softmax should be applied after
            the final 1x1 convolution
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, init_channel_number=32, **kwargs):
        super(DistanceTransformUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order='crg',
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, pool_type='avg', conv_layer_order='crg',
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(3 * init_channel_number, init_channel_number, conv_layer_order='crg', num_groups=num_groups)
        ])

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        # allow multiple heads
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final 1x1 convolution
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class EndToEndDTUNet3D(nn.Module):
    def __init__(self, tags_in_channels, tags_out_channels, tags_output_heads, tags_init_channel_number,
                 dt_in_channels, dt_out_channels, dt_final_sigmoid, dt_init_channel_number,
                 tags_net_path=None, dt_net_path=None, **kwargs):
        super(EndToEndDTUNet3D, self).__init__()

        self.tags_net = TagsUNet3D(tags_in_channels, tags_out_channels, tags_output_heads,
                                   init_channel_number=tags_init_channel_number)
        if tags_net_path is not None:
            # load pre-trained TagsUNet3D
            self.tags_net = self._load_net(tags_net_path, self.tags_net)

        self.dt_net = DistanceTransformUNet3D(dt_in_channels, dt_out_channels, dt_final_sigmoid,
                                              init_channel_number=dt_init_channel_number)
        if dt_net_path is not None:
            # load pre-trained DistanceTransformUNet3D
            self.dt_net = self._load_net(dt_net_path, self.dt_net)

    @staticmethod
    def _load_net(checkpoint_path, model):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model_state_dict'])
        return model

    def forward(self, x):
        x = self.tags_net(x)
        return self.dt_net(x)


class VaeUNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cbr', num_groups=8,
                 **kwargs):
        super(VaeUNet, self).__init__()
        self.conv3d = conv3d(in_channels, 32, kernel_size=3, bias=True)
        self.dropout = nn.Dropout(p=0.2)
        self.convblock = SingleConv(32, 32, order=layer_order, kernel_size=3)
        self.downblock1 = DownBlock(32, 16)
        self.downblock2 = DownBlock(16, 32)
        self.downblock3 = DownBlock(32, 64)

        self.greenblock1 = GreenBlock(64, 64)
        self.greenblock2 = GreenBlock(64, 64)

        self.upblock1 = UpBlock(64, 32)
        self.convblock1 = SingleConv(96, 32, order=layer_order, num_groups=num_groups)
        self.convblock2 = SingleConv(32, 32, order=layer_order, num_groups=num_groups)
        self.upblock2 = UpBlock(32, 16)
        self.convblock3 = SingleConv(48, 16, order=layer_order, num_groups=num_groups)
        self.convblock4 = SingleConv(16, 16, order=layer_order, num_groups=num_groups)
        self.upblock3 = UpBlock(16, 8)
        self.convblock5 = SingleConv(24, 4, order=layer_order, num_groups=num_groups)
        self.convblock6 = SingleConv(4, 4, order=layer_order, num_groups=num_groups)
        self.final_conv = conv3d(4, out_channels, kernel_size=1, bias=True, padding=0)

        self.vae_block = VaeBlock(64, in_channels)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv3d(x)
        x = self.dropout(x)
        x = self.convblock(x)

        level1, l1_conv = self.downblock1(x)
        level2, l2_conv = self.downblock2(level1)
        level3, l3_conv = self.downblock3(level2)

        conv1 = self.greenblock1(level3)
        conv2 = self.greenblock2(conv1)

        level3_up = self.upblock1(conv2)
        concat = torch.cat([level3_up, l3_conv], 1)
        level3_up = self.convblock1(concat)
        level3_up = self.convblock2(level3_up)

        level2_up = self.upblock2(level3_up)
        concat = torch.cat([level2_up, l2_conv], 1)
        level2_up = self.convblock3(concat)
        level2_up = self.convblock4(level2_up)

        level1_up = self.upblock3(level2_up)
        concat = torch.cat([level1_up, l1_conv], 1)
        level1_up = self.convblock5(concat)
        level1_up = self.convblock6(level1_up)

        output = self.final_conv(level1_up)
        vae_out, z_mean, z_var = self.vae_block(conv2)

        if not self.training:
            output = self.final_activation(output)
        return output, vae_out, z_mean, z_var


class unetConv3(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n

        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size
        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, 'kaiming')

    def forward(self, input):
        x = input
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x


class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv=False, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv3(in_size+(n_concat-2)*out_size, out_size, is_batchnorm=False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_size, out_size, kernel_size=1)
            )

        for m in self.children():
            if m.__class__.__name__.find('unetConv3') != -1:continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):

        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class UNet_Nested(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cgr', num_groups=8,
                n_classes=4, feature_scale=3, is_deconv=False, is_batchnorm=True, is_ds=True, ** kwargs):

        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds                    # deep supervision

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        filters = [16, 32, 64, 128, 256]

        # downsampling
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv00 = unetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv3d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv3d(filters[0], n_classes, 1)
        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        x00 = self.conv00(inputs)
        maxpool0 = self.maxpool(x00)
        x10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool(x10)
        x20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool(x20)
        x30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool(x30)
        x40 = self.conv40(maxpool3)
        # column : 1
        x01 = self.up_concat01(x10, x00)
        x11 = self.up_concat11(x20, x10)
        x21 = self.up_concat21(x30, x20)
        x31 = self.up_concat31(x40, x30)
        # column : 2
        x02 = self.up_concat02(x11, x00, x01)
        x12 = self.up_concat12(x21, x10, x11)
        x22 = self.up_concat22(x31, x20, x21)
        # column : 3
        x03 = self.up_concat03(x12, x00, x01, x02)
        x13 = self.up_concat13(x22, x10, x11, x12)
        # column : 4
        x04 = self.up_concat04(x13, x00, x01, x02, x03)

        # output_sample = x04[0, 1, :, :, 80].cpu().detach().numpy()
        # fig = plt.figure()
        # feature_image = fig.add_subplot(1, 1, 1)
        # plt.imshow(output_sample, cmap="gray")
        # feature_image.set_title('output')
        # plt.savefig('/home/liujing/pytorch-3dunet/picture/{}.png'.format(str(random.randint(1, 1000))))
        # plt.close()

        # final layer
        final_1 = self.final_1(x01)
        final_2 = self.final_2(x02)
        final_3 = self.final_3(x03)
        final_4 = self.final_4(x04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return self.sigmoid(final)
        else:
            return final_4


class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(DownSampling, self).__init__()

        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=inChans,
                               out_channels=outChans,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):
    '''
    Encoder block
    '''

    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu",
                 normalizaiton="group_normalization"):
        super(EncoderBlock, self).__init__()

        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        return out


class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''

    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)

    def forward(self, x, skipx=None):
        out = self.conv1(x)
        # out = self.up1(out)
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode,
                                        align_corners=self.align_corners)

        if skipx is not None:
            out = torch.cat((out, skipx), 1)
            out = self.conv2(out)

        return out


class DecoderBlock(nn.Module):
    '''
    Decoder block
    '''

    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu",
                 normalizaiton="group_normalization"):
        super(DecoderBlock, self).__init__()

        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        return out


class OutputTransition(nn.Module):
    '''
    Decoder output layer
    output the prediction of segmentation result
    '''

    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = torch.sigmoid

    def forward(self, x):
        return self.actv1(self.conv1(x))


class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''

    def __init__(self, inChans=256, outChans=256, dense_features=(10, 12, 8), stride=2, kernel_size=3, padding=1,
                 activation="relu", normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()

        midChans = int(inChans / 2)
        self.dense_features = dense_features
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.dense1 = nn.Linear(in_features=16 * dense_features[0] * dense_features[1] * dense_features[2],
                                out_features=inChans)
        self.dense2 = nn.Linear(in_features=midChans,
                                out_features=midChans * dense_features[0] * dense_features[1] * dense_features[2])
        self.up0 = LinearUpSampling(midChans, outChans)

    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = out.view(-1, self.num_flat_features(out))
        out_vd = self.dense1(out)
        distr = out_vd
        out = VDraw(out_vd)
        out = self.dense2(out)
        out = self.actv2(out)
        out = out.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        out = self.up0(out)

        return out, distr

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


def VDraw(x):
    # Generate a Gaussian distribution with the given mean(128-d) and std(128-d)
    return torch.distributions.Normal(x[:, :128], x[:, 128:]).sample()


class VDecoderBlock(nn.Module):
    '''
    Variational Decoder block
    '''

    def __init__(self, inChans, outChans, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DecoderBlock(outChans, outChans, activation=activation, normalizaiton=normalizaiton)

    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out


class VAE(nn.Module):
    '''
    Variational Auto-Encoder : to group the features extracted by Encoder
    '''

    def __init__(self, inChans=256, outChans=4, dense_features=(10, 12, 8), activation="relu",
                 normalizaiton="group_normalization", mode="trilinear"):
        super(VAE, self).__init__()

        self.vd_resample = VDResampling(inChans=inChans, outChans=inChans, dense_features=dense_features)
        self.vd_block2 = VDecoderBlock(inChans, inChans // 2)
        self.vd_block1 = VDecoderBlock(inChans // 2, inChans // 4)
        self.vd_block0 = VDecoderBlock(inChans // 4, inChans // 8)
        self.vd_end = nn.Conv3d(inChans // 8, outChans, kernel_size=1)

    def forward(self, x):
        out, distr = self.vd_resample(x)
        out = self.vd_block2(out)
        out = self.vd_block1(out)
        out = self.vd_block0(out)
        out = self.vd_end(out)

        return out, distr


class NvNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(NvNet, self).__init__()

        # some critical parameters
        self.inChans = in_channels
        self.input_shape = (1, 6, 128, 128, 128)
        self.seg_outChans = out_channels
        self.activation = "relu"
        self.normalizaiton = "group_normalization"
        self.mode = "trilinear"
        self.vae = True

        # Encoder Blocks
        self.in_conv0 = DownSampling(inChans=self.inChans, outChans=32, stride=1, dropout_rate=0.2)
        self.en_block0 = EncoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block1_1 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block2_1 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_1 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_2 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_3 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)

        # Decoder Blocks
        self.de_up2 = LinearUpSampling(256, 128, mode=self.mode)
        self.de_block2 = DecoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up1 = LinearUpSampling(128, 64, mode=self.mode)
        self.de_block1 = DecoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up0 = LinearUpSampling(64, 32, mode=self.mode)
        self.de_block0 = DecoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_end = OutputTransition(32, self.seg_outChans)

        # Variational Auto-Encoder
        self.dense_features = (self.input_shape[2] // 16, self.input_shape[3] // 16, self.input_shape[4] // 16)
        self.vae = VAE(256, outChans=self.inChans, dense_features=self.dense_features)

    def forward(self, x):
        out_init = self.in_conv0(x)
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0)))
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        out_en3 = self.en_block3_3(
            self.en_block3_2(
                self.en_block3_1(
                    self.en_block3_0(
                        self.en_down3(out_en2)))))

        out_de2 = self.de_block2(self.de_up2(out_en3, out_en2))
        out_de1 = self.de_block1(self.de_up1(out_de2, out_en1))
        out_de0 = self.de_block0(self.de_up0(out_de1, out_en0))
        out_end = self.de_end(out_de0)

        if self.vae:
            out_vae, out_distr = self.vae(out_en3)
            out_final = torch.cat((out_end, out_vae), 1)
            return out_final, out_distr

        return out_end


class SegCaps(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 1, padding=2, bias=False),

        )
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", k=3, s=1, t_1=2, z_1=16, routing=1),
            CapsuleLayer(2, 16, "conv", k=3, s=1, t_1=4, z_1=16, routing=3),
        )
        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(4, 16, "conv", k=3, s=1, t_1=4, z_1=32, routing=3),
            CapsuleLayer(4, 32, "conv", k=3, s=1, t_1=8, z_1=32, routing=3)
        )
        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", k=3, s=1, t_1=8, z_1=64, routing=3),
            CapsuleLayer(8, 64, "conv", k=3, s=1, t_1=8, z_1=32, routing=3)
        )
        self.step_4 = CapsuleLayer(8, 32, "deconv", k=3, s=1, t_1=8, z_1=32, routing=3)

        self.step_5 = CapsuleLayer(16, 32, "conv", k=3, s=1, t_1=4, z_1=32, routing=3)

        self.step_6 = CapsuleLayer(4, 32, "deconv", k=3, s=1, t_1=4, z_1=16, routing=3)
        self.step_7 = CapsuleLayer(8, 16, "conv", k=3, s=1, t_1=4, z_1=16, routing=3)
        self.step_8 = CapsuleLayer(4, 16, "deconv", k=3, s=1, t_1=2, z_1=16, routing=3)
        self.step_10 = CapsuleLayer(3, 16, "conv", k=3, s=1, t_1=1, z_1=16, routing=3)

    def forward(self, x):
        x = F.relu(self.conv_1(x), inplace=True)
        x.unsqueeze_(1)

        skip_1 = x  # [N,1,16,H,W]

        x = self.step_1(x)

        skip_2 = x  # [N,4,16,H/2,W/2]
        x = self.step_2(x)

        skip_3 = x  # [N,8,32,H/4,W/4]

        x = self.step_3(x)  # [N,8,32,H/8,W/8]

        x = self.step_4(x)  # [N,8,32,H/4,W/4]

        x = torch.cat((x, skip_3), 1)  # [N,16,32,H/4,W/4]

        x = self.step_5(x)  # [N,4,32,H/4,W/4]

        x = self.step_6(x)  # [N,4,16,H/2,W/2]

        x = torch.cat((x, skip_2), 1)   # [N,8,16,H/2,W/2]
        x = self.step_7(x)  # [N,4,16,H/2,W/2]
        x = self.step_8(x)  # [N,2,16,H,W]

        x = torch.cat((x, skip_1), 1)
        x = self.step_10(x)
        x.squeeze_(1)
        v_lens = self.compute_vector_length(x)
        # v_lens = v_lens.squeeze(1)
        # output = torch.sigmoid(v_lens)
        return v_lens

    def compute_vector_length(self, x):
        out = (x.pow(2)).sum(1, True)+1e-9
        out = out.sqrt()
        return out


class NoNewCapsNet_step1(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(NoNewCapsNet_step1, self).__init__()
        channels = 30
        self.levels = 4

        # create caps encoder levels
        capsencoderModels = []
        capsencoderModels.append(EncoderModule(in_channels, channels, False, True))
        for i in range(self.levels - 2):
            capsencoderModels.append(EncoderModule(channels * pow(2, i), channels * pow(2, i + 1), True, True))
        capsencoderModels.append(
                EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), True, False))
        self.capsencoders = nn.ModuleList(capsencoderModels)

        # create reconstruct decoder
        recondecoderModels = []
        recondecoderModels.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, False))
        for i in range(self.levels - 2):
            recondecoderModels.append(DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        recondecoderModels.append(DecoderModule(channels, channels, False, True))
        self.recondecodersA = nn.ModuleList(recondecoderModels)
        self.lastConvA = nn.Conv3d(channels, 1, 1, bias=True)
        self.recondecodersB = nn.ModuleList(recondecoderModels)
        self.lastConvB = nn.Conv3d(channels, 1, 1, bias=True)

        # create feature fusion
        self.capsule_layer = CapsuleLayer(2, 240, "conv", k=3, s=1, t_1=1, z_1=240, routing=3)
        # self.routing_layer = nn.Conv3d(2 * channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 1), 1, bias=True)

    def forward(self, a, b):
        latent = a
        for i in range(self.levels):
            latent = self.capsencoders[i](latent)

        reconstructB = b
        for i in range(self.levels):
            reconstructB = self.capsencoders[i](reconstructB)

        reconstructA = latent
        for i in range(self.levels):
            reconstructA = self.recondecodersA[i](reconstructA)
        reconstructA = self.lastConvA(reconstructA)

        for i in range(self.levels):
            reconstructB = self.recondecodersB[i](reconstructB)
        reconstructB = self.lastConvB(reconstructB)

        return latent, reconstructA, reconstructB


class NoNewCapsNet_step2(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(NoNewCapsNet_step2, self).__init__()
        channels = 30
        self.levels = 4

        self.lastConv = nn.Conv3d(channels, out_channels, 1, bias=True)

        # create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(in_channels, channels, False, True))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i+1), True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), True, False))
        self.encoders = nn.ModuleList(encoderModules)

        # create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, False))
        for i in range(self.levels - 2):
            decoderModules.append(DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        decoderModules.append(DecoderModule(channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)

        # create reconstruct decoder
        recondecoderModels = []
        recondecoderModels.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, False))
        for i in range(self.levels - 2):
            recondecoderModels.append(DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        recondecoderModels.append(DecoderModule(channels, channels, False, True))
        self.recondecodersA = nn.ModuleList(recondecoderModels)
        self.lastConvA = nn.Conv3d(channels, 1, 1, bias=True)

        # create feature fusion
        self.capsule_layer = CapsuleLayer(2, 240, "conv", k=3, s=1, t_1=2, z_1=240, routing=3)
        self.conv_fusion = nn.Conv3d(480, 240, 1, bias=True)
        self.gn_fusion = nn.GroupNorm(2, 240)
        # self.routing_layer = nn.Conv3d(2 * channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 1), 1, bias=True)

    def forward(self, a, code_flair):
        inputStack = []
        for i in range(self.levels):
            a = self.encoders[i](a)
            if i < self.levels - 1:
                inputStack.append(a)

        # a = torch.cat([a.unsqueeze(1), code_flair.unsqueeze(1)], 1)
        a = torch.cat([a, code_flair], 1)
        a = F.leaky_relu(self.gn_fusion(self.conv_fusion(a)))

        # a = self.capsule_layer(a)
        # caps_list = [caps for caps in a.split(1, 1)]
        # ET = caps_list[0].squeeze(1)
        # TC = caps_list[1].squeeze(1)
        # WT = caps_list[0].squeeze(1)

        reconstructA = a
        for i in range(self.levels):
            reconstructA = self.recondecodersA[i](reconstructA)
        reconstructA = self.lastConvA(reconstructA)

        for i in range(self.levels):
            a = self.decoders[i](a)
            # TC = self.decoders[i](TC)
            # WT = self.decoders[i](WT)
            if i < self.levels - 1:
                a = a + inputStack.pop()

        a = self.lastConv(a)
        # TC = self.lastConv(TC)
        # WT = self.lastConv(WT)
        seg = torch.sigmoid(a)

        return seg, reconstructA


class NoNewNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(NoNewNet, self).__init__()
        channels = 90
        self.levels = 4

        self.lastConv = nn.Conv3d(channels, out_channels, 1, bias=True)
        self.se = SELayer(in_channels)

        # create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(in_channels, channels, True, True, alpha_in=0))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i+1), True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), True, True))
        self.encoders = nn.ModuleList(encoderModules)

        # create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, True))
        for i in range(self.levels - 2):
            decoderModules.append(
                    DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True,
                                  True))
        decoderModules.append(DecoderModule(channels, 2*channels, True, True))
        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        x, y_out = self.se(x)
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x, y_out


class ResNoNewNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(ResNoNewNet, self).__init__()
        channels = 30
        self.levels = 5

        self.lastConv = nn.Conv3d(channels, 3, 1, bias=True)

        #create encoder levels
        encoderModules = []
        encoderModules.append(ResEncoderModule(4, channels, False, True))
        for i in range(self.levels - 2):
            encoderModules.append(ResEncoderModule(channels * pow(2, i), channels * pow(2, i+1), True, True))
        encoderModules.append(ResEncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), True, False))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        decoderModules.append(ResDecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, False))
        for i in range(self.levels - 2):
            decoderModules.append(ResDecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        decoderModules.append(ResDecoderModule(channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x


class NNNet_Vae(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(NNNet_Vae, self).__init__()
        channels = 30
        self.levels = 5

        self.lastConv = nn.Conv3d(channels, 3, 1, bias=True)

        # create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(4, channels, False, True))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i + 1), True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1),
                                            True, False))
        self.encoders = nn.ModuleList(encoderModules)

        # create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2),
                                            True, False))
        for i in range(self.levels - 2):
            decoderModules.append(DecoderModule(
                channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        decoderModules.append(DecoderModule(channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)

        self.vae_block = VaeBlock(480, in_channels)

    def forward(self, x):
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        vae_out, z_mean, z_var = self.vae_block(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x, vae_out, z_mean, z_var


class NNNet_Cae(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(NNNet_Cae, self).__init__()
        channels = 30
        self.levels = 5

        self.lastConv = nn.Conv3d(channels, out_channels, 1, bias=True)

        # create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(4, channels, False, True))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i + 1), True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1),
                                            True, False))
        self.encoders = nn.ModuleList(encoderModules)

        self.greenblock = GreenBlock(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 1))
        # create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2),
                                            True, False))
        for i in range(self.levels - 2):
            decoderModules.append(DecoderModule(
                channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        decoderModules.append(DecoderModule(channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)

        self.Cae_block = CaeBlock(480, 1)

    def forward(self, x):
        inputStack = []
        feature_maps = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)
                # feature_maps.append(x)

        x = self.greenblock(x)
        cae_out = self.Cae_block(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x, cae_out, feature_maps


class unet_CT_multi_att_dsv_3D(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 feature_scale=4, is_deconv=True,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, **kwargs):
        super(unet_CT_multi_att_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=out_channels, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=out_channels, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=out_channels, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=out_channels, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(out_channels*4, out_channels, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))
        final = torch.sigmoid(final)

        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlockND(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlockND(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            4,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.level1 = nn.Conv3d(512 * block.expansion, 256, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm3d(256)
        self.relu1 = nn.ReLU(inplace=True)

        self.level2 = nn.Conv3d(256, 64, kernel_size=3 ,padding=1, bias=False)
        self.bn_2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.level3 = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.final_conv = nn.Conv3d(
                                    32,
                                    num_seg_classes,
                                    kernel_size=1,
                                    stride=(1, 1, 1),
                                    bias=False)

        self.conv_seg = nn.Sequential(
            nn.ConvTranspose3d(
                512 * block.expansion,
                512,
                2,
                stride=2
            ),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                512,
                512,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(
                512,
                256,
                2,
                stride=2
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                256,
                256,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(
                256,
                32,
                2,
                stride=2
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                32,
                num_seg_classes,
                kernel_size=1,
                stride=(1, 1, 1),
                bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x
        x = self.conv1(x)   #(1,64,64,64,64)
        x = self.bn1(x)     #(1,64,64,64,64)
        x1 = self.relu(x)
        x = self.maxpool(x1) #(1,64,32,32,32)
        x2 = self.layer1(x)  #(1,64,32,32,32)
        x = self.layer2(x2)  #(1,128,16,16,16)
        x = self.layer3(x)  #(1,256,16,16,16)
        x = self.layer4(x)  #(1,512,16,16,16)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.level1(x)
        x = self.bn_1(x)
        x = self.relu1(x)
        x = x + x2

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.level2(x)
        x = self.bn_2(x)
        x = self.relu2(x)
        x = x + x1

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.level3(x)
        x = self.bn_3(x)
        x = self.relu3(x)
        x = self.final_conv(x)
        # x = self.conv_seg(x)
        x = torch.sigmoid(x)
        return x

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class resnet101(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(resnet101, self).__init__()

    def forward(self, **kwargs):
        model = ResNet(MedicaNetBottleneck, [3, 4, 23, 3], **kwargs)
        return model

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(MedicaNetBasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(MedicaNetBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(MedicaNetBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(MedicaNetBottleneck, [3, 4, 6, 3], **kwargs)
    return model


# def resnet101(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(MedicaNetBottleneck, [3, 4, 23, 3], **kwargs)
#     return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(MedicaNetBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(MedicaNetBottleneck, [3, 24, 36, 3], **kwargs)
    return model