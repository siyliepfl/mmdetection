import random
from mmdet.datasets import PIPELINES
import cv2 as cv
import math
import numpy as np

@PIPELINES.register_module()
class SampleTarget:
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self,  search_area_factor=2, output_sz=128):
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz

    def __call__(self, results):
        img = results['img']
        target_bb = results['bbox']

        im_crop_padded, resize_factor, att_mask = self.sample_target(img, target_bb,
                                                                     self.search_area_factor, output_sz=self.output_sz)

        results['img'] = im_crop_padded
        results['img_shape'] = im_crop_padded.shape
        results['ori_shape'] = im_crop_padded.shape

        return results

    def sample_target(self, im, target_bb, search_area_factor, output_sz=128):
        """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb
        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
        x2 = int(x1 + crop_sz)

        y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
        y2 = int(y1 + crop_sz)

        x1_pad = int(max(0, -x1))
        x2_pad = int(max(x2 - im.shape[1] + 1, 0))

        y1_pad = int(max(0, -y1))
        y2_pad = int(max(y2 - im.shape[0] + 1, 0))

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # Pad
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
        # deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H, W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0

        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)

        return im_crop_padded, resize_factor, att_mask

