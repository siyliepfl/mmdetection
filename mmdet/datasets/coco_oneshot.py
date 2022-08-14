# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import json
import logging
import os.path
import os.path as osp
import torch
import tempfile
import warnings
from collections import OrderedDict
from einops import rearrange
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from mmcv.parallel.data_container import DataContainer
from mmdet.core import eval_map, eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
import random

import copy
from PIL import ImageFilter, Image
from mmdet.models.utils.box_ops import box_xywh_to_xyxy
from torchvision.transforms import transforms
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmcv.image.photometric import *
from mmdet.core.visualization import imshow_det_bboxes
import matplotlib.pyplot as plt

def debug_plot(data):
    img_ori = imdenormalize(np.array(data['img'].data.permute(1, 2, 0)),
                            mean=np.array([123.675, 116.28, 103.53]),
                            std=np.array([58.395, 57.12, 57.375]))
    crop_im = imshow_det_bboxes(img_ori,
                                np.array(data['query_targets'].data).reshape(-1,4),
                                np.array([0]),
                                segms=None)

    qimg_ori = imdenormalize(np.array(data['query_img'].data.permute(1, 2, 0)),
                             mean=np.array([123.675, 116.28, 103.53]),
                             std=np.array([58.395, 57.12, 57.375]))

    plt.imshow(qimg_ori/255)
    plt.show()

def get_random_patch_from_img(im_size, gt_bboxes=None, min_pixel=8, bg_gt_overlap_iou=0.3):

    """
    :param img: original image
    :param min_pixel: min pixels of the query patch
    :return: query_patch,x,y,w,h
    """
    while True:
        h, w = im_size
        min_w, max_w = min_pixel, w - min_pixel
        min_h, max_h = min_pixel, h - min_pixel
        sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
        x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0

        bbox = torch.tensor([x, y, sw, sh])
        if gt_bboxes is not None:
            ious = bbox_overlaps(np.array(gt_bboxes), np.array(box_xywh_to_xyxy(bbox)).reshape(-1, 4))
            if (ious > bg_gt_overlap_iou).any():
                continue
            else:
                return x, y, sw, sh

        return  x, y, sw, sh

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

@DATASETS.register_module()
class CocoOneShotDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def __init__(self, query_pipeline,
                 split,
                 average_num,
                 query_json=None,
                 no_test_class_present=False,
                 bg_crop_freq=0.25,
                 bg_gt_overlap_iou=0.3,
                 **kwargs):
        super().__init__( **kwargs)

        if self.test_mode and query_json is not None:
            self.class_anno_mapping, self.imid_info_mapping_dict = self.build_query_bank_from_files(query_json)
        else:
            self.class_anno_mapping = self.build_class_anno_mapping()

        self.query_json = query_json
        self.label2cat = {y:x for x,y in self.cat2label.items()}

        if self.test_mode:
            self.pipeline = Compose(kwargs['pipeline'])
        else:
            self.load_pipeline = Compose(kwargs['pipeline'][:2])
            self.transform_pipeline = Compose(kwargs['pipeline'][2:-1])
            bg_transform_pipeline = copy.deepcopy(kwargs['pipeline'][2:-1])
            bg_transform_pipeline[1] = dict(type='RandomFlip', flip_ratio=0.0)
            self.bg_transform_pipeline = Compose(bg_transform_pipeline)
            self.collect_pipeline = Compose([kwargs['pipeline'][-1]])

        # self.transform_pipeline = Compose(kwargs['pipeline'][:-1])
        # self.collect_pipeline = Compose([kwargs['pipeline'][-1]])
        #

        self.bg_query_pipeline = self.get_query_transforms()

        self.query_pipeline = Compose(query_pipeline)
        self.avg_num = average_num
        self.split = split
        self.no_test_class_present = no_test_class_present
        self.bg_crop_freq = bg_crop_freq
        self.bg_gt_overlap_iou = bg_gt_overlap_iou

        self.known_cats_labels = [cat
                                  for cat in range(0, 80)
                                  if cat % 4 != split
                                  ]
        self.unknown_cats_labels = [cat
                                    for cat in range(0, 80)
                                    if cat % 4 == split
                                    ]
        self.known_cats_ids = {
                        self.label2cat[cat_label]
                                 for cat_label in self.known_cats_labels
        }
        self.unknown_cats_ids = {
                        self.label2cat[cat_label]
                                 for cat_label in self.unknown_cats_labels
        }

    def get_query_transforms(self):
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def build_query_bank_from_files(self, query_json):

        train_query_dict = json.load(open(query_json))
        class_anno_mapping = {}

        for item in train_query_dict['annotations']:
            if item['bbox'][2] > 50 and item['bbox'][3] > 50 and item['iscrowd'] == 0:
                if item['category_id'] not in class_anno_mapping:
                    class_anno_mapping[item['category_id']] = [item]
                else:
                    class_anno_mapping[item['category_id']].append(item)

        if len(class_anno_mapping.keys()) != len(self.CLASSES):
            empty_class_id = set(self.cat_ids) - set(class_anno_mapping.keys())
            empty_class_name = [self.CLASSES[self.cat2label[cid]] for cid in empty_class_id]
            print('Empty classes', empty_class_name, 'after filtering')

        imid_info_mapping_dict = {}
        for im in train_query_dict['images']:
            if im['id'] not in imid_info_mapping_dict:
                im['filename'] = im['file_name']
                imid_info_mapping_dict[im['id']] = im

        return class_anno_mapping, imid_info_mapping_dict


    def build_class_anno_mapping(self):

        class_anno_mapping = {}

        for key, value in self.coco.anns.items():
            if value['bbox'][2] > 50 and value['bbox'][3] > 50 and value['iscrowd'] == 0:
                if value['category_id'] not in class_anno_mapping:
                    class_anno_mapping[value['category_id']] = [value]
                else:
                    class_anno_mapping[value['category_id']].append(value)

        if len(class_anno_mapping.keys()) != len(self.CLASSES):
            empty_class_id = set(self.cat_ids) - set(class_anno_mapping.keys())
            empty_class_name = [self.CLASSES[self.cat2label[cid]] for cid in empty_class_id]
            print('Empty classes', empty_class_name, 'after filtering')

        return class_anno_mapping


    def generate_oneshot_data(self, data, query_cat_id, anno_list):
        """

        Randomly sample one class that exist in the target image. Filter the ground truth.
        Load random query image based on the classes of the current search image.

        Args:
            data: dict containing the search image and its annotation

        Returns:
            diction containing data and new loaded cropped image

        """

        query_data = None
        while query_data is None:
            query_anno = anno_list[random.randint(0, len(anno_list) - 1)]
            query_img_id = query_anno['image_id']
            query_data = self.prepare_query_img(query_anno, query_img_id)

        query_label_id = self.cat2label[query_cat_id]
        data['query_img_metas'] = query_data['img_metas']
        data['query_img'] = query_data['img']
        data['query_label_id'] = DataContainer(torch.tensor(query_label_id))
        data['query_targets'] = DataContainer(data['gt_bboxes'].data[data['gt_labels'].data == query_label_id])
        data['query_labels'] = DataContainer(torch.zeros(len(data['query_targets'] )).long())

        return data

    def prepare_query_img(self, query_anno, query_img_id):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        if self.query_json is None:
            img_info = self.coco.imgs[query_img_id]
        else:
            img_info = self.imid_info_mapping_dict[query_img_id]

        results = dict(img_info=img_info)

        self.pre_pipeline(results)
        results['bbox'] = query_anno['bbox']
        query_res = self.query_pipeline(results)

        return query_res

    def load_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.load_pipeline(results)



    def check_train_img(self, idx):

        while True:
            # load annotations
            img_id = self.data_infos[idx]['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann_cat_ids = {item['category_id'] for item in ann_info}
            known_interset_anns = ann_cat_ids.intersection(self.known_cats_ids)

            # check empty image
            if not known_interset_anns:
                # print("Image ", str(self.data_infos[idx]['id']), 'has no annotations!!!!')
                idx = self._rand_another(idx)
                continue

            # check corresponding query images for the class exists
            query_cat_id = list(known_interset_anns)[random.randint(0, len(known_interset_anns) - 1)]
            if query_cat_id not in self.class_anno_mapping:
                idx = self._rand_another(idx)
                continue

            # check test classes presences
            if self.no_test_class_present:
                unknown_interset_anns = ann_cat_ids.intersection(self.unknown_cats_ids)
                if unknown_interset_anns:
                    idx = self._rand_another(idx)
                    continue

            anno_list = self.class_anno_mapping[query_cat_id]

            return idx, query_cat_id, anno_list

    def generate_self_det_query_img(self, data):


        img_info = data['img_info']
        pil_img = Image.fromarray(data['img'])
        results = dict(img_info=img_info,
                       img=data['img'])
        self.pre_pipeline(results)
        bbox = get_random_patch_from_img(im_size = data['img_shape'][:2],
                                         gt_bboxes=data['gt_bboxes'].data,
                                         bg_gt_overlap_iou=self.bg_gt_overlap_iou)
        x, y, sw, sh = bbox
        data['gt_bboxes'] = np.array(box_xywh_to_xyxy(torch.tensor(bbox, dtype=torch.float)).unsqueeze(0))
        query_img = pil_img.crop((x, y, x + sw, y + sh))
        query_img = self.bg_query_pipeline(query_img)
        data['query_img'] = DataContainer(query_img, stack=True)

        return data

    def generate_self_det_target_bboxes(self, data):

        data = self.bg_transform_pipeline(data)
        data['query_targets'] = data['gt_bboxes']
        data['query_labels'] = DataContainer(torch.zeros(1).long())

        return data

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            # perform random background crop

            if random.random() < self.bg_crop_freq:
                data = self.load_train_img(idx)
                if data is None:
                    idx = self._rand_another(idx)
                    continue
                data = self.generate_self_det_query_img(data)
                data = self.generate_self_det_target_bboxes(data)
                data = self.collect_pipeline(data)
                return data

            else:
                idx, query_cat_id, query_ann_list = self.check_train_img(idx)
                data = self.load_train_img(idx)
                if data is None:
                    idx = self._rand_another(idx)
                    continue
                data = self.transform_pipeline(data)
                data = self.generate_oneshot_data(data, query_cat_id, query_ann_list)
                data = self.collect_pipeline(data)
                return data

    def prepare_query_test_img(self, im_id, query_cat_labels, per_cat_num=2):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        random.seed(im_id)
        per_class_query_imgs = {}
        for c in query_cat_labels:
            cat_id = self.label2cat[c]
            if cat_id not in self.class_anno_mapping:
                print("No template for class id", c)
            anno_list = self.class_anno_mapping[cat_id]
            query_list = random.choices(anno_list, k=per_cat_num)
            query_data_list = []
            for query_anno in query_list:
                query_img_id = query_anno['image_id']
                # query_idx = self.img_ids.index(query_img_id)
                query_data = self.prepare_query_img(query_anno, query_img_id)
                query_img = query_data['img'].data.unsqueeze(0)
                query_data_list.append(query_img)
            query_data_tensor = torch.cat(query_data_list, dim=0)
            per_class_query_imgs[c] = query_data_tensor

        return per_class_query_imgs

    def prepare_test_img(self, idx):

        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        img_id = img_info['id']
        gt_class_labels = set(ann_info['labels'])
        per_class_query_imgs = self.prepare_query_test_img(img_id,
                                                           gt_class_labels,
                                                           per_cat_num=self.avg_num)
        results['per_class_query_imgs'] = per_class_query_imgs
        results['avg_num'] = self.avg_num
        return results

    def load_annotations(self, ann_file):
        """
        Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []

        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"

        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files[0]:
                raise KeyError(f'{metric} is not in results')


            eval_list = []
            for res_file in result_files:
                try:
                    predictions = mmcv.load(res_file[metric])
                    if iou_type == 'segm':
                        # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                        # When evaluating mask AP, if the results contain bbox,
                        # cocoapi will use the box area instead of the mask area
                        # for calculating the instance area. Though the overall AP
                        # is not affected, this leads to different
                        # small/medium/large mask AP results.
                        for x in predictions:
                            x.pop('bbox')
                        warnings.simplefilter('once')
                        warnings.warn(
                            'The key "bbox" is deleted for more accurate mask AP '
                            'of small/medium/large instances since v2.12.0. This '
                            'does not change the overall mAP calculation.',
                            UserWarning)
                    coco_det = coco_gt.loadRes(predictions)
                except IndexError:
                    print_log(
                        'The testing results of the whole dataset is empty.',
                        logger=logger,
                        level=logging.ERROR)
                    return {}
                    # break

                cocoEval = COCOeval(coco_gt, coco_det, iou_type)
                cocoEval.params.catIds = self.cat_ids
                cocoEval.params.imgIds = self.img_ids
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.params.iouThrs = iou_thrs
                # mapping of cocoEval.stats
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }
                if metric_items is not None:
                    for metric_item in metric_items:
                        if metric_item not in coco_metric_names:
                            raise KeyError(
                                f'metric item {metric_item} is not supported')

                if metric == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.evaluate()
                    cocoEval.accumulate()

                    # Save coco summarize print information to logger
                    redirect_string = io.StringIO()
                    with contextlib.redirect_stdout(redirect_string):
                        cocoEval.summarize()
                    print_log('\n' + redirect_string.getvalue(), logger=logger)

                    if metric_items is None:
                        metric_items = [
                            'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                            'AR_m@1000', 'AR_l@1000'
                        ]

                    for item in metric_items:
                        val = float(
                            f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                        eval_results[item] = val
                else:
                    cocoEval.evaluate()
                    cocoEval.accumulate()

                    # add eval to the list
                    eval_list.append(cocoEval.eval)

            sum_precision = np.zeros(eval_list[0]['precision'].shape)
            sum_recall = np.zeros(eval_list[0]['recall'].shape)
            sum_scores = np.zeros(eval_list[0]['scores'].shape)
            for item in eval_list:
                sum_precision += item['precision']
                sum_recall += item['recall']
                sum_scores += item['scores']

            avg_precision = sum_precision / self.avg_num
            avg_recall  = sum_recall / self.avg_num
            avg_scores = sum_scores / self.avg_num
            cocoEval.eval['precision'] = avg_precision
            cocoEval.eval['recall'] = avg_recall
            cocoEval.eval['scores'] = avg_scores

            ################################### OVERALL ##############################################
            print('Overall performance with known and unknown:')
            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

            ################################### Seen ##############################################
            print('Performance with known class')

            cocoEval.eval['precision'] = avg_precision[:,:,self.known_cats_labels,:,:]
            cocoEval.eval['recall'] = avg_recall[:,self.known_cats_labels,:,:]
            cocoEval.eval['scores'] = avg_scores[:,:,self.known_cats_labels,:,:]

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_known_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

            ################################### Unseen ##############################################
            print('Performance with unknown class')

            cocoEval.eval['precision'] = avg_precision[:,:,self.unknown_cats_labels,:,:]
            cocoEval.eval['recall'] = avg_recall[:, self.unknown_cats_labels,:,:]
            cocoEval.eval['scores'] = avg_scores[:,:,self.unknown_cats_labels,:,:]

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_unknown_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

        return eval_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast', 'mAP', 'recall'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.
            dataset (str): Specify which dataset to use for the evaluation.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """


        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)


        if not os.path.exists(jsonfile_prefix):
            os.mkdir(jsonfile_prefix)

        results_files_list = []
        for i in range(self.avg_num):
            tmp_result = [item[i] for item in results]
            result_files, tmp_dir = self.format_results(tmp_result, jsonfile_prefix + '/'+str(i))
            results_files_list.append(result_files)

        eval_results = self.evaluate_det_segm(results, results_files_list, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)


        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
