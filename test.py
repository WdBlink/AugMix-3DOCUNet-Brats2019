from unet3d.config import load_config
import os
import torch
import torch.nn as nn
from datasets.hdf5 import BratsDataset
from unet3d.model import get_model
from unet3d import utils

logger = utils.get_logger('UNet3DPredictor')

# Load and log experiment configuration
config = load_config()

# Load model state
model = get_model(config)
model_path = config['trainer']['test_model']
logger.info(f'Loading model from {model_path}...')
utils.load_checkpoint(model_path, model)

# Run on GPU or CPU
# if torch.cuda.is_available():
#     print("using cuda (", torch.cuda.device_count(), "device(s))")
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
#     device = torch.device("cuda:1")
# else:
#     device = torch.device("cpu")
#     print("using cpu")
# model = model.to(device)
logger.info(f"Sending the model to '{config['device']}'")
model = model.to('cuda:1')

predictionsBasePath = config['loaders']['pred_path']
BRATS_VAL_PATH = config['loaders']['test_path']

challengeValset = BratsDataset(BRATS_VAL_PATH[0], mode="validation", hasMasks=False, returnOffsets=True)
challengeValloader = torch.utils.data.DataLoader(challengeValset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)


def makePredictions():
    # model is already loaded from disk by constructor

    basePath = os.path.join(predictionsBasePath[0])
    if not os.path.exists(basePath):
        os.makedirs(basePath)

    with torch.no_grad():
        for i, data in enumerate(challengeValloader):
            inputs, pids, xOffset, yOffset, zOffset = data
            print("processing {}".format(pids[0]))
            inputs = inputs.to(config['device'])

            # predict labels and bring into required shape
            outputs = model(inputs)
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
            max = npResult.max()
            min = npResult.min()
            path = os.path.join(basePath, "{}.nii.gz".format(pids[0]))
            utils.save_nii(path, npResult, None, None)

    print("Done :)")


if __name__ == "__main__":
    makePredictions()