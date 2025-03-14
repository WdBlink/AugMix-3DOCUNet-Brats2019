# Author:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Robin Brügger
# Adopted from: https://git.ee.ethz.ch/baumgach/discriminative_learning_toolbox/blob/master/utils.py
import cv2
import numpy as np
import unet3d.utils as utils
from matplotlib import pyplot as plt
import time


def augment3DImage(img, lbl, defaultLabelValues, nnAug, do_rotate=True, rotDegrees=20, do_scale=True, scaleFactor=1.1,
                   do_flip=True, do_elasticAug=True, sigma=10, do_intensityShift=True, maxIntensityShift=0.1):
    '''
    Function for augmentation of a 3D image. It will transform the image and corresponding labels
    by a number of optional transformations.
    :param img: A numpy array of shape [X, Y, Z, nChannels]
    :param lbl: A numpy array containing a corresponding label mask
    :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
    :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                        back to the original size.
    :param do_fliplr: Perform random flips with a 50% chance in the left right direction.
    :return: Transformed images and masks.
    '''

    startTime = time.time()
    if type(img) == list:
        xSize = img[0].shape[0]
        ySize = img[0].shape[1]
        zSize = img[0].shape[2]
        defaultPerChannel = img[0][0, 0, 0, :]

        if nnAug:
            interpolationMethod = cv2.INTER_NEAREST
        else:
            interpolationMethod = cv2.INTER_LINEAR

        if do_rotate:
            random_angle = np.random.uniform(-rotDegrees, rotDegrees)
            for z in range(zSize):
                img[0][:, :, z, :] = utils.rotate_image(img[0][:, :, z, :], random_angle)
                img[1][:, :, z, :] = utils.rotate_image(img[1][:, :, z, :], random_angle)
                lbl[0][:, :, z, :] = utils.rotate_image(lbl[0][:, :, z, :], random_angle, interpolationMethod)
                lbl[1][:, :, z, :] = utils.rotate_image(lbl[1][:, :, z, :], random_angle, interpolationMethod)

        # RANDOM SCALE
        if do_scale:
            scale = np.random.uniform(1 / scaleFactor, 1 * scaleFactor)
            for z in range(zSize):
                scaledSize = [round(xSize*scale), round(ySize*scale)]
                imgScaled = utils.resize_image(img[0][:, :, z, :], scaledSize)
                lblScaled = utils.resize_image(lbl[0][:, :, z, :], scaledSize, interpolationMethod)
                if scale < 1:
                    img[0][:,:, z,:] = padToSize(imgScaled, [xSize, ySize], defaultPerChannel)
                    lbl[0][:, :, z, :] = padToSize(lblScaled, [xSize, ySize], defaultLabelValues)
                    img[1][:, :, z, :] = padToSize(imgScaled, [xSize, ySize], defaultPerChannel)
                    lbl[1][:, :, z, :] = padToSize(lblScaled, [xSize, ySize], defaultLabelValues)
                else:
                    img[0][:,:, z,:] = cropToSize(imgScaled, [xSize, ySize])
                    lbl[0][:, :, z, :] = cropToSize(lblScaled, [xSize, ySize])
                    img[1][:, :, z, :] = cropToSize(imgScaled, [xSize, ySize])
                    lbl[1][:, :, z, :] = cropToSize(lblScaled, [xSize, ySize])

        # RANDOM ELASTIC DEFOMRATIONS (like in U-NET)
        if do_elasticAug:

            mu = 0

            dx = np.random.normal(mu, sigma, 9)
            dx_mat = np.reshape(dx, (3, 3))
            dx_img = utils.resize_image(dx_mat, (xSize, ySize), interp=cv2.INTER_CUBIC)

            dy = np.random.normal(mu, sigma, 9)
            dy_mat = np.reshape(dy, (3, 3))
            dy_img = utils.resize_image(dy_mat, (xSize, ySize), interp=cv2.INTER_CUBIC)

            for z in range(zSize):
                img[0][:, :, z, :] = utils.dense_image_warp(img[0][:, :, z, :], dx_img, dy_img)
                lbl[0][:, :, z, :] = utils.dense_image_warp(lbl[0][:, :, z, :], dx_img, dy_img, interpolationMethod)
                img[1][:, :, z, :] = utils.dense_image_warp(img[1][:, :, z, :], dx_img, dy_img)
                lbl[1][:, :, z, :] = utils.dense_image_warp(lbl[1][:, :, z, :], dx_img, dy_img, interpolationMethod)

        # RANDOM INTENSITY SHIFT
        if do_intensityShift:
            for i in range(4):  # number of channels
                img[0][:, :, :, i] = img[0][:, :, :, i] + np.random.uniform(-maxIntensityShift, maxIntensityShift) #assumes unit std derivation
                img[1][:, :, :, i] = img[1][:, :, :, i] + np.random.uniform(-maxIntensityShift, maxIntensityShift)
        # RANDOM FLIP
        if do_flip:
            for i in range(3):
                if np.random.random() < 0.5:
                    img[0] = np.flip(img[0], axis=i)
                    lbl[0] = np.flip(lbl[0], axis=i)
                    img[1] = np.flip(img[1], axis=i)
                    lbl[1] = np.flip(lbl[1], axis=i)

        return img[0].copy(), lbl[0].copy(), img[1].copy(), lbl[1].copy()
        # pytorch cannot handle negative stride in view
    else:
        xSize = img.shape[0]
        ySize = img.shape[1]
        zSize = img.shape[2]
        defaultPerChannel = img[0, 0, 0, :]

        if nnAug:
            interpolationMethod = cv2.INTER_NEAREST
        else:
            interpolationMethod = cv2.INTER_LINEAR

        if do_rotate:
            random_angle = np.random.uniform(-rotDegrees, rotDegrees)
            for z in range(zSize):
                img[:, :, z, :] = utils.rotate_image(img[:, :, z, :], random_angle)
                lbl[:, :, z, :] = utils.rotate_image(lbl[:, :, z, :], random_angle, interpolationMethod)

        # RANDOM SCALE
        if do_scale:
            scale = np.random.uniform(1 / scaleFactor, 1 * scaleFactor)
            for z in range(zSize):
                scaledSize = [round(xSize*scale), round(ySize*scale)]
                imgScaled = utils.resize_image(img[:, :, z, :], scaledSize)
                lblScaled = utils.resize_image(lbl[:, :, z, :], scaledSize, interpolationMethod)
                if scale < 1:
                    img[:,:, z,:] = padToSize(imgScaled, [xSize, ySize], defaultPerChannel)
                    lbl[:, :, z, :] = padToSize(lblScaled, [xSize, ySize], defaultLabelValues)
                else:
                    img[:,:, z,:] = cropToSize(imgScaled, [xSize, ySize])
                    lbl[:, :, z, :] = cropToSize(lblScaled, [xSize, ySize])

        # RANDOM ELASTIC DEFOMRATIONS (like in U-NET)
        if do_elasticAug:

            mu = 0

            dx = np.random.normal(mu, sigma, 9)
            dx_mat = np.reshape(dx, (3, 3))
            dx_img = utils.resize_image(dx_mat, (xSize, ySize), interp=cv2.INTER_CUBIC)

            dy = np.random.normal(mu, sigma, 9)
            dy_mat = np.reshape(dy, (3, 3))
            dy_img = utils.resize_image(dy_mat, (xSize, ySize), interp=cv2.INTER_CUBIC)

            for z in range(zSize):
                img[:, :, z, :] = utils.dense_image_warp(img[:, :, z, :], dx_img, dy_img)
                lbl[:, :, z, :] = utils.dense_image_warp(lbl[:, :, z, :], dx_img, dy_img, interpolationMethod)

        # RANDOM INTENSITY SHIFT
        if do_intensityShift:
            for i in range(4):  # number of channels
                img[:, :, :, i] = img[:, :, :, i] + np.random.uniform(-maxIntensityShift, maxIntensityShift) #assumes unit std derivation

        # RANDOM FLIP
        if do_flip:
            for i in range(3):
                if np.random.random() < 0.5:
                    img = np.flip(img, axis=i)
                    lbl = np.flip(lbl, axis=i)

        return img.copy(), lbl.copy()  # pytorch cannot handle negative stride in view


def visualizeSlice(slice):
    plt.imshow(slice, interpolation='nearest')
    plt.show()

def cropToSize(image, targetSize):
    offsetX = (image.shape[0] - targetSize[0]) // 2
    endX = offsetX + targetSize[0]
    offsetY = (image.shape[1] - targetSize[1]) // 2
    endY = offsetY + targetSize[1]
    return image[offsetX:endX, offsetY:endY, :]

def padToSize(image, targetSize, backgroundColor):
    offsetX = (targetSize[0] - image.shape[0]) // 2
    endX = offsetX + image.shape[0]
    offsetY = (targetSize[1] - image.shape[1]) // 2
    endY = offsetY + image.shape[1]
    targetSize.append(image.shape[2]) #add channels to shape
    paddedImg = np.ones(targetSize, dtype=np.float32) * backgroundColor
    paddedImg[offsetX:endX, offsetY:endY, :] = image
    return paddedImg
