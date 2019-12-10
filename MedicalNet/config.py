data_root='./data'
input_D=128
input_H=128
input_W=128
n_seg_classes=3
gpu_id = [0, 1]
pretrain_path='/home/server/github/MedicalNet/pretrain/resnet_50.pth'

new_layer_names=['conv_seg']
# default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],

no_cuda='store_true'
model='resnet'
model_depth=50
resnet_shortcut='B'