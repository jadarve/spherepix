"""
    spherepix.blending
    ---------------

    Spherepix image blending module

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details.
"""

import numpy as np

import spherepix.camera as scam

import spherepix.gpu.camera as gcam
import spherepix.gpu.pixelation as gpix


def blendOutput(etasOut, etasIn, imgIn, isVerticalFlipped=True):
    """Blends a set of spherepix images onto a single camera view.

    Parameters
    ----------
    etasOut : ndarray.
        Spherical coordinates grid of output view.

    etasIn : list of np.ndarray.
        List of input spherical coordinates patches.

    imgIn : list of np.ndarray.
        List of input images to blend.

    Returns
    -------
    imgOut : ndarray.
        Blended image.
    """

    gface = gpix.PixelationFace(etasOut)

    heightOut = etasOut.shape[0]
    widthOut = etasOut.shape[1]
    channels = 1 if imgIn[0].ndim == 2 else imgIn[0].shape[2]

    # image size
    ishape = imgIn[0].shape
    dtype = imgIn[0].dtype
    
    # 2D input image to GPU
    # FIXME: dtype
    gimg = gpix.PixelationFaceImage((ishape[0], ishape[1]))

    mappedImages = list()
    imgMasks = list()


    for etas, img in zip(etasIn, imgIn):

        # create camera
        cam = scam.OmnidirectionalCamera(etas, isVerticalFlipped)
        gpuCam = gcam.GPUCamera(cam)

        # image mapper
        imgMapper = gcam.GPUFaceImageMapper(gpuCam, gface, gimg)

        # compute mask
        betas = imgMapper.getInterpolationCoordinates().download()
        mask = (betas[...,0] >= 0.0) | (betas[...,1] >= 0.0)
        imgMasks.append(mask.astype(np.float32))
        

        if img.ndim == 2:

            gimg.upload(img)
            imgMapper.compute()
            imgMapped = imgMapper.getMappedImage().download()

            mappedImages.append(imgMapped)

        elif img.ndim ==3:

            imgMapped = np.zeros((heightOut, widthOut, img.shape[2]), dtype=dtype)

            # map each component
            for n in range(img.shape[2]):
                gimg.upload(np.copy(img[...,n]))
                imgMapper.compute()
                imgMapped[...,n] = imgMapper.getMappedImage().download()

            mappedImages.append(imgMapped)

        else:
            raise ValueError('input image should have 2 or 3 dimensionts, got: {0}'.format(img.ndim))

    
    # average blend
    maskSum = np.zeros_like(imgMasks[0])
    
    if channels == 1:
        imgOut = np.zeros((heightOut, widthOut), dtype=np.float32) 
    else:
        imgOut = np.zeros((heightOut, widthOut, channels), dtype=np.float32)
    
    
    for mask, img in zip(imgMasks, mappedImages):        
        imgOut += img
        maskSum += mask


    if channels == 1:
        imgOut /= maskSum
        imgOut[maskSum == 0.0] = 0.0
    else:
        for n in range(channels):
            imgOut[...,n] /= maskSum
            imgOut[maskSum == 0.0, n] = 0.0
    
    return imgOut, imgMasks

