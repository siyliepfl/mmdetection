# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
# from mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmcv.ops.nms import batched_nms
import numpy as np
import copy
import torch
from mmdet.core.visualization import imshow_det_bboxes
import mmcv
import torch.nn as nn
from mmcv.cnn import ConvModule

@DETECTORS.register_module()
class MAdet(SingleStageDetector):
    """Implementation of Adet """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MAdet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)


        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        template_list, x = self.extract_feat(img)

        losses = self.bbox_head.forward_train(x, template_list, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        template_list, x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            # template_list = self.neck(template_list)
        return template_list, x


    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        template_list, feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, template_list, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results