import collections
import importlib

import h5py
import numpy as np
import torch
import torchvision
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import augment.transforms as transforms
import BraTS
from unet3d.utils import get_logger
from unet3d.losses import expand_as_one_hot
import preprocess.augmentation as aug
import nibabel as nib

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


class SliceBuilder:
    def __init__(self, raw_datasets, label_datasets, weight_dataset, patch_shape, stride_shape):
        self._raw_slices = self._build_slices(raw_datasets[0], patch_shape, stride_shape)
        if label_datasets is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            self._label_slices = self._build_slices(label_datasets[0], patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset[0], patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.8, slack_acceptance=0.01):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape)
        if label_datasets is None:
            return

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_datasets[0][label_idx]
            non_ignore_counts = np.array([np.count_nonzero(patch != ii) for ii in ignore_index])
            non_ignore_counts = non_ignore_counts / patch.size
            return np.any(non_ignore_counts > threshold) or np.random.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class BratsDataset(torch.utils.data.Dataset):
    # mode must be trian, test or val
    def __init__(self, filePath, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, doMixUp=True,
                 template=None, data_aug=True):
        super(BratsDataset, self).__init__()
        self.filePath = filePath
        self.mode = mode
        self.file = None
        # self.trainOriginalClasses = expConfig.TRAIN_ORIGINAL_CLASSES
        self.randomCrop = randomCrop
        self.hasMasks = hasMasks
        self.returnOffsets = returnOffsets
        self.doMixUp = doMixUp
        self.aug = data_aug

        # augmentation settings
        self.nnAugmentation = True
        self.softAugmentation = False
        self.doRotate = True
        self.rotDegrees = 20
        self.doScale = True
        self.scaleFactor = 1.1
        self.doFlip = True
        self.doElasticAug = True
        self.sigma = 10
        self.doIntensityShift = True
        self.maxIntensityShift = 0.1
        graph_brain, _, _ = load_nii(template)
        self.graph_brain = np.transpose(graph_brain, (1, 2, 3, 0))

    def __getitem__(self, index):

        # lazily open file
        self.openFileIfNotOpen()

        # load from hdf5 file
        image = self.file["images_" + self.mode][index, ...]
        if self.hasMasks: labels = self.file["masks_" + self.mode][index, ...]

        # Prepare data depeinding on soft/hard augmentation scheme
        if not self.nnAugmentation:
            if not self.trainOriginalClasses and (self.mode != "train" or self.softAugmentation):
                if self.hasMasks:
                    # et_labels = self._toEvaluationTwoClassification(labels, subrigon='ET')
                    # tc_labels = self._toEvaluationTwoClassification(labels, subrigon='TC')
                    wt_labels = self._toEvaluationTwoClassification(labels, subrigon='WT')
                    # bg_labels = self._toEvaluationTwoClassification(labels, subrigon='BG')
                defaultLabelValues = np.zeros(3, dtype=np.float32)
            else:
                if self.hasMasks:
                    # et_labels = self._toEvaluationTwoClassification(labels, subrigon='ET')
                    # tc_labels = self._toEvaluationTwoClassification(labels, subrigon='TC')
                    wt_labels = self._toEvaluationTwoClassification(labels, subrigon='WT')
                    # bg_labels = self._toEvaluationTwoClassification(labels, subrigon='BG')
                defaultLabelValues = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)
        elif self.hasMasks:
            if labels.ndim < 4:
                labels = np.expand_dims(labels, 3)
            defaultLabelValues = np.asarray([0], dtype=np.float32)

        if self.nnAugmentation:
            if self.hasMasks:
                # et_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='ET')
                # tc_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='TC')
                wt_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='WT')
                # bg_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='BG')
        else:
            if self.mode == "train" and not self.softAugmentation and not self.trainOriginalClasses and self.hasMasks:
                labels = self._toOrdinal(labels)
                # et_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='ET')
                # tc_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='TC')
                wt_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='WT')
                # bg_labels = self._toEvaluationTwoClassification(np.squeeze(labels, 3), subrigon='BG')

        if self.doMixUp:
            datasize = self.file["images_" + self.mode].shape[0]
            idx = np.random.randint(0, datasize)
            image2 = self.file["images_" + self.mode][idx, ...]
            labels2 = self.file["masks_" + self.mode][idx, ...]

            # Prepare data depeinding on soft/hard augmentation scheme
            if not self.nnAugmentation:
                if not self.trainOriginalClasses and (self.mode != "train" or self.softAugmentation):
                    if self.hasMasks: labels2 = self._toEvaluationOneHot(labels2)
                    defaultLabelValues = np.zeros(3, dtype=np.float32)
                else:
                    if self.hasMasks: labels2 = self._toOrignalCategoryOneHot(labels2)
                    defaultLabelValues = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)
            elif self.hasMasks:
                if labels2.ndim < 4:
                    labels2 = np.expand_dims(labels2, 3)
                defaultLabelValues = np.asarray([0], dtype=np.float32)

            if self.nnAugmentation:
                if self.hasMasks: labels2 = self._toEvaluationOneHot(np.squeeze(labels2, 3))
            else:
                if self.mode == "train" and not self.softAugmentation and not self.trainOriginalClasses and self.hasMasks:
                    labels2 = self._toOrdinal(labels2)
                    labels2 = self._toEvaluationOneHot(labels2)

            # random crop
            if not self.randomCrop is None:
                shape = image2.shape
                x = random.randint(0, shape[0] - self.randomCrop[0])
                y = random.randint(0, shape[1] - self.randomCrop[1])
                z = random.randint(0, shape[2] - self.randomCrop[2])
                image2 = image2[x:x + self.randomCrop[0], y:y + self.randomCrop[1], z:z + self.randomCrop[2], :]
                if self.hasMasks: labels2 = labels2[x:x + self.randomCrop[0], y:y + self.randomCrop[1],
                                           z:z + self.randomCrop[2], :]
            image2 = np.transpose(image2, (3, 0, 1, 2))  # bring into NCWH format

            # m1 = 0.5
            # m2 = 0.5
            # m3 = 0.5
            # alpha = 0.4
            # lam = np.random.beta(alpha, alpha)
            # image = lam * image + (1 - lam) * image2
            #
            # target = np.zeros_like(labels)
            # # if lam > m1:
            # #     target[..., 0] = target[..., 0] + labels[..., 0]
            # # if (1 - lam) > m1:
            # #     target[..., 0] = target[..., 0] + labels2[..., 0]
            # target[..., 0] = target[..., 0] + lam * labels[..., 0] + (1 - lam) * labels2[..., 0]
            #
            # # if lam > m2:
            # #     target[..., 1] = target[..., 1] + labels[..., 1]
            # # if (1 - lam) > m2:
            # #     target[..., 1] = target[..., 1] + labels2[..., 1]
            # target[..., 1] = target[..., 1] + lam * labels[..., 1] + (1 - lam) * labels2[..., 1]
            #
            # # if lam > m3:
            # #     target[..., 2] = target[..., 2] + labels[..., 2]
            # # if (1 - lam) > m3:
            # #     target[..., 2] = target[..., 2] + labels2[..., 2]
            # target[..., 2] = target[..., 2] + lam * labels[..., 2] + (1 - lam) * labels2[..., 2]
            #
            # target[target > 1] = 1
            # labels = target

        # # templateChannelAddition
        # minus = image - self.graph_brain
        # result = np.clip(np.abs(minus), 1, 255)
        # image = np.concatenate((image, result[:, :, :, 0:1]), 3)
        # image = np.concatenate((image, result[:, :, :, 3:4]), 3)

        labels = wt_labels
        # augment data
        # if self.mode == "train" and self.aug is True:
            # image, labels = aug.augment3DImage(image,
            #                                    labels,
            #                                    defaultLabelValues,
            #                                    self.nnAugmentation,
            #                                    self.doRotate,
            #                                    self.rotDegrees,
            #                                    self.doScale,
            #                                    self.scaleFactor,
            #                                    self.doFlip,
            #                                    self.doElasticAug,
            #                                    self.sigma,
            #                                    self.doIntensityShift,
            #                                    self.maxIntensityShift)


        # random crop
        if not self.randomCrop is None:
            shape = image.shape
            x = random.randint(0, shape[0] - self.randomCrop[0])
            y = random.randint(0, shape[1] - self.randomCrop[1])
            z = random.randint(0, shape[2] - self.randomCrop[2])
            image = image[x:x+self.randomCrop[0], y:y+self.randomCrop[1], z:z+self.randomCrop[2], :]
            if self.hasMasks:
                labels = labels[x:x + self.randomCrop[0], y:y + self.randomCrop[1], z:z + self.randomCrop[2], :]

        image_ori = image

        if self.mode == "train" and self.aug is True:
            augmix1, mixlabel1 = self.AugMix(image, labels)
            augmix2, mixlabel2 = self.AugMix(image, labels)
            augmix1 = np.transpose(augmix1, (3, 0, 1, 2))
            augmix2 = np.transpose(augmix2, (3, 0, 1, 2))

        image_ori = np.transpose(image_ori, (3, 0, 1, 2))  # bring into NCWH format

        if self.hasMasks:
            labels = np.transpose(labels, (3, 0, 1, 2))  # bring into NCWH format
            if self.mode == "train" and self.aug is True:
                mixlabel1 = np.transpose(mixlabel1, (3, 0, 1, 2))
                mixlabel2 = np.transpose(mixlabel2, (3, 0, 1, 2))

        # to tensor
        #image = image[:, 0:32, 0:32, 0:32]
        # image = self.make_crop(image)
        image_ori = torch.from_numpy(image_ori)
        if self.mode == "train" and self.aug is True:
            augmix1 = torch.from_numpy(augmix1)
            augmix2 = torch.from_numpy(augmix2)
        if self.hasMasks:
            labels = torch.from_numpy(labels)
            if self.mode == "train" and self.aug is True:
                mixlabel1 = torch.from_numpy(mixlabel1)
                mixlabel2 = torch.from_numpy(mixlabel2)

        # get pid
        pid = self.file["pids_" + self.mode][index]

        if self.returnOffsets:
            xOffset = self.file["xOffsets_" + self.mode][index]
            yOffset = self.file["yOffsets_" + self.mode][index]
            zOffset = self.file["zOffsets_" + self.mode][index]
            if self.hasMasks:
                return image_ori, str(pid), image2, xOffset, yOffset, zOffset
            else:
                return image_ori, pid, xOffset, yOffset, zOffset
        else:
            if self.hasMasks:
                if self.mode == "train" and self.aug is True:
                    return image_ori, str(pid), labels, augmix1, mixlabel1, augmix2, mixlabel2
                else:
                    return image_ori, str(pid), labels
            else:
                return image_ori, pid

    def __len__(self):
        # lazily open file
        self.openFileIfNotOpen()

        return self.file["images_" + self.mode].shape[0]

    def AugMix(self, image, labels):
        """Perform AugMix augmentations and compute mixture.

        Args:
          image: PIL.Image input image
          preprocess: Preprocessing function which should return a torch tensor.

        Returns:
          mixed: Augmented and mixed image.
        """
        defaultLabelValues = np.asarray([0], dtype=np.float32)
        ws = np.float32(np.random.dirichlet([1] * 3))
        m = np.float32(np.random.beta(1, 1))

        mix = np.zeros_like(image)
        labels_aug = np.zeros_like(labels)
        for i in range(3):
            image_aug = image.copy()
            labels_aug = labels.copy()
            depth = 1
            for _ in range(depth):
                op = np.random.choice([True, False], 5)
                image_aug, labels_aug = aug.augment3DImage(image_aug,
                                                           labels_aug,
                                                           defaultLabelValues,
                                                           nnAug=self.nnAugmentation,
                                                           do_rotate=op[0],
                                                           rotDegrees=self.rotDegrees,
                                                           do_scale=op[1],
                                                           scaleFactor=self.scaleFactor,
                                                           do_flip=op[2],
                                                           do_elasticAug=op[3],
                                                           do_intensityShift=op[4])
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * image_aug
            labels_aug += ws[i] * labels_aug

        mixed = (1 - m) * image + m * mix
        mixed_label = (1 - m) * labels + m * labels_aug
        return mixed, mixed_label

    def openFileIfNotOpen(self):
        if self.file == None:
            self.file = h5py.File(self.filePath, "r")

    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 3], dtype=np.float32)
        out[:, :, :, 0] = (labels != 0)
        out[:, :, :, 1] = (labels != 0) * (labels != 2)
        out[:, :, :, 2] = (labels == 4)
        return out

    def _toEvaluationTwoClassification(self, labels, subrigon):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 1], dtype=np.float32)
        if subrigon == 'ET':
            out[:, :, :, 0] = (labels == 4)
        elif subrigon == 'TC':
            out[:, :, :, 0] = (labels != 0) * (labels != 2)
        elif subrigon == 'WT':
            out[:, :, :, 0] = (labels != 0)
        elif subrigon == 'BG':
            out[:, :, :, 0] = (labels == 0)
        return out

    def _toOrignalCategoryOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 5], dtype=np.float32)
        for i in range(5):
            out[:, :, :, i] = (labels == i)
        return out

    def _toOrdinal(self, labels):
        return np.argmax(labels, axis=3)


class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path, patch_shape, stride_shape, phase, transformer_config,
                 raw_internal_path='raw', label_internal_path='label',
                 weight_internal_path=None, slice_builder_cls=SliceBuilder):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        :param slice_builder_cls: defines how to sample the patches from the volume
        """
        assert phase in ['train', 'val', 'test']
        self._check_patch_shape(patch_shape)
        self.phase = phase
        self.file_path = file_path

        # convert raw_internal_path, label_internal_path and weight_internal_path to list for ease of computation
        if isinstance(raw_internal_path, str):
            raw_internal_path = [raw_internal_path]
        if isinstance(label_internal_path, str):
            label_internal_path = [label_internal_path]
        if isinstance(weight_internal_path, str):
            weight_internal_path = [weight_internal_path]

        with h5py.File(file_path, 'r') as input_file:
            # WARN: we load everything into memory due to hdf5 bug when reading H5 from multiple subprocesses, i.e.
            # File "h5py/_proxy.pyx", line 84, in h5py._proxy.H5PY_H5Dread
            # OSError: Can't read data (inflate() failed)
            self.raws = [input_file[internal_path][...] for internal_path in raw_internal_path]
            # calculate global mean and std for Normalizati                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               on augmentation
            mean, std = self._calculate_mean_std(self.raws[0])

            self.transformer = transforms.get_transformer(transformer_config, mean, std, phase)
            self.raw_transform = self.transformer.raw_transform()

            if phase != 'test':
                # create label/weight transform only in train/val phase
                self.label_transform = self.transformer.label_transform()
                self.labels = [input_file[internal_path][...] for internal_path in label_internal_path]

                if weight_internal_path is not None:
                    # look for the weight map in the raw file
                    self.weight_maps = [input_file[internal_path][...] for internal_path in weight_internal_path]
                    self.weight_transform = self.transformer.weight_transform()
                else:
                    self.weight_maps = None

                self._check_dimensionality(self.raws, self.labels)
            else:
                # 'test' phase used only for predictions so ignore the label dataset
                self.labels = None
                self.weight_maps = None

            # build slice indices for raw and label data sets
            slice_builder = slice_builder_cls(self.raws, self.labels, self.weight_maps, patch_shape, stride_shape)
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices
            self.weight_slices = slice_builder.weight_slices

            self.patch_count = len(self.raw_slices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self._transform_patches(self.raws, raw_idx, self.raw_transform)

        if self.phase == 'test':
            # just return the transformed raw patch and the metadata
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self._transform_patches(self.labels, label_idx, self.label_transform)
            if self.weight_maps is not None:
                weight_idx = self.weight_slices[idx]
                # return the transformed weight map for a given patch together with raw and label data
                weight_patch_transformed = self._transform_patches(self.weight_maps, weight_idx, self.weight_transform)
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _calculate_mean_std(input):
        """
        Compute a mean/std of the raw stack for normalization.
        This is an in-memory implementation, override this method
        with the chunk-based computation if you're working with huge H5 files.
        :return: a tuple of (mean, std) of the raw data
        """
        return input.mean(keepdims=True), input.std(keepdims=True)

    @staticmethod
    def _check_dimensionality(raws, labels):
        for raw in raws:
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            if raw.ndim == 3:
                raw_shape = raw.shape
            else:
                raw_shape = raw.shape[1:]

        for label in labels:
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            if label.ndim == 3:
                label_shape = label.shape
            else:
                label_shape = label.shape[1:]
            assert raw_shape == label_shape, 'Raw and labels have to be of the same size'

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
        assert patch_shape[0] >= 16, 'Depth must be greater or equal 16'


def _get_slice_builder_cls(class_name):
    m = importlib.import_module('datasets.hdf5')
    clazz = getattr(m, class_name)
    return clazz


def get_brats_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset.

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger = get_logger('BraTS_Dataset')
    logger.info('Creating training and validation set loaders...')

    # get train and validation files
    train_path = loaders_config['train_path']
    val_path = loaders_config['val_path']

    # loss_file_num = 0
    # for i in train_ids:
    #     name = i + ".tfrecord.gzip"
    #     answer = search('/home/server/data/TFRecord/val', name)
    #     if answer == -1:
    #         print("查无此文件", name)
    #         loss_file_num += 1
    # print(f'loss file num is {loss_file_num}')

    logger.info(f'Loading training set from: {train_path}...')
    # train_datasets = BraTSDataset(brats, train_ids, phase='train',
    #                               transformer_config=loaders_config['transformer'],
    #                               is_mixup=loaders_config['mixup'])
    train_datasets = BratsDataset(train_path[0], doMixUp=loaders_config['mixup'],
                                  data_aug=loaders_config['data_aug'], randomCrop=[128, 128, 128], template=loaders_config['template_path'][0])

    logger.info(f'Loading validation set from: {val_path}...')
    # brats = BraTS.DataSet(brats_root=data_paths[0], year=2019).train
    # val_datasets = BraTSDataset(brats, validation_ids, phase='val',
    #                             transformer_config=loaders_config['transformer'],
    #                             is_mixup=False)
    val_datasets = BratsDataset(val_path[0], mode='validation', template=loaders_config['template_path'][0])

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val datasets: {num_workers}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints

    challengeValset = BratsDataset(loaders_config['test_path'][0], mode="validation", hasMasks=False,
                                   returnOffsets=True, template=loaders_config['template_path'][0])

    return {
        'train': DataLoader(train_datasets, batch_size=loaders_config['batch_size'], shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=num_workers),
        'challenge': DataLoader(challengeValset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    }
#
# import os
# import sys
# def search(path,name):
#
#     for root, dirs, files in os.walk(path):  # path 为根目录
#         if name in dirs or name in files:
#             flag = 1      #判断是否找到文件
#             root = str(root)
#             dirs = str(dirs)
#             return os.path.join(root, dirs)
#     return -1


def get_test_loaders(config):
    """
    Returns a list of DataLoader, one per each test file.

    :param config: a top level configuration object containing the 'datasets' key
    :return: generator of DataLoader objects
    """

    def my_collate(batch):
        error_msg = "batch must contain tensors or slice; found {}"
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        elif isinstance(batch[0], slice):
            return batch[0]
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [my_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    logger = get_logger('HDF5Dataset')

    assert 'datasets' in config, 'Could not find data sets configuration'
    datasets_config = config['datasets']

    # get train and validation files
    test_paths = datasets_config['test_path']
    assert isinstance(test_paths, list)
    # get h5 internal path
    raw_internal_path = datasets_config['raw_internal_path']
    # get train/validation patch size and stride
    patch = tuple(datasets_config['patch'])
    stride = tuple(datasets_config['stride'])
    num_workers = datasets_config.get('num_workers', 1)

    # construct datasets lazily
    datasets = (HDF5Dataset(test_path, patch, stride, phase='test', raw_internal_path=raw_internal_path,
                            transformer_config=datasets_config['transformer']) for test_path in test_paths)

    # use generator in order to create data loaders lazily one by one
    for dataset in datasets:
        logger.info(f'Loading test set from: {dataset.file_path}...')
        yield DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=my_collate)
