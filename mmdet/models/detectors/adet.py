# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
# from mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmcv.ops.nms import batched_nms
import numpy as np
import copy
import torch

@DETECTORS.register_module()
class Adet(SingleStageDetector):
    """Implementation of Adet """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 nms_cfg=None,
                 init_cfg=None):
        super(Adet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

        if nms_cfg is not None:
            self.nms_class_agnostic = nms_cfg['class_agnostic']
            self.batch_nms_cfg = nms_cfg['batch_nms_cfg']
            self.do_nms = True
            self.max_per_img = nms_cfg['max_per_img']
        else:
            self.do_nms = False

    def forward_train(self,
                      img,
                      img_metas,
                      query_img,
                      gt_bboxes,
                      gt_labels,
                      query_labels,
                      query_targets,
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
        x = self.extract_feat(query_img, img)
        losses = self.bbox_head.forward_train(x, img_metas, query_targets,
                                              query_labels, gt_bboxes_ignore)

        return losses

    def extract_feat(self, query_img, img):
        """Directly extract features from the backbone+neck."""
        template, search = self.backbone(query_img, img)
        if self.with_neck:
            x = self.neck(search)
        return x

    def simple_test(self, img, img_metas, per_class_query_imgs, avg_num, rescale=False):
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
        bbox_res_per_img = [np.empty((0, 6))] * 80
        results = []
        for i in range(avg_num):
            results.append(copy.deepcopy(bbox_res_per_img))
        repeat_imgs = img.repeat(avg_num, 1, 1, 1)
        img_metas = img_metas * avg_num
        for class_label in per_class_query_imgs:
            query_samples = per_class_query_imgs[class_label].squeeze(0)
            feat = self.extract_feat(query_samples, repeat_imgs)
            results_list = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            for i in range(avg_num):
                # add class label
                bbox_results_with_class_label = np.concatenate((bbox_results[i][0],
                                                                np.full((len(bbox_results[i][0]),1), class_label)),
                                                               axis=1)
                results[i][class_label] = bbox_results_with_class_label

        # nms
        if self.do_nms:
            for i in range(avg_num):
                all_bboxes = np.concatenate(results[i])
                boxes = torch.tensor(all_bboxes[:, :4], dtype=torch.float)
                scores = torch.tensor(all_bboxes[:, 4], dtype=torch.float)
                idxs = torch.tensor(all_bboxes[:, 5], dtype=torch.float)
                kept_bboxes, keep_index = batched_nms(boxes, scores, idxs, self.batch_nms_cfg, class_agnostic=self.nms_class_agnostic)
                kept_bboxes = kept_bboxes[:self.max_per_img]
                keep_index = keep_index[:self.max_per_img]
                kept_class_labels = idxs[keep_index]
                results[i] = bbox2result(kept_bboxes, kept_class_labels, len(self.CLASSES))

        final_result = [results]

        return final_result

