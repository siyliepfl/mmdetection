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
from mmdet.core import eval_map, eval_recalls, print_map_summary
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
import random

@DATASETS.register_module()
class VocMultiOneShotDataset(CustomDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    PALETTE = [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
               (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
               (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
               (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
               (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]

    def __init__(self, query_pipeline,
                 split=0,
                 average_num=1,
                 query_json=None,
                 no_test_class_present=False,
                 test_type='all',
                 **kwargs):
        """
        Args:
            query_pipeline:
            split:
            average_num:
            query_json:
            no_test_class_present:
            test_type: (str) all | unknown_only | known_only
            **kwargs:
        """
        super().__init__( **kwargs)

        if self.test_mode and query_json is not None:
            self.class_anno_mapping, self.imid_info_mapping_dict = self.build_query_bank_from_files(query_json)
        else:
            self.class_anno_mapping = self.build_class_anno_mapping()

        self.query_json = query_json
        self.label2cat = {y:x for x,y in self.cat2label.items()}
        self.transform_pipeline = Compose(kwargs['pipeline'][:-1])
        self.collect_pipeline = Compose([kwargs['pipeline'][-1]])
        self.query_pipeline = Compose(query_pipeline)
        self.avg_num = average_num
        self.split = split
        self.no_test_class_present = no_test_class_present

        self.unknown_cats_ids = [self.CLASSES.index('cow'),
                                 self.CLASSES.index('sheep'),
                                 self.CLASSES.index('cat'),
                                 self.CLASSES.index('aeroplane')]
        self.known_cats_ids = list(set(self.cat_ids) - set(self.unknown_cats_ids))
        self.known_cats_labels = self.known_cats_ids
        self.unknown_cats_labels = self.unknown_cats_ids
        self.test_type = test_type

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


    def generate_oneshot_data(self, data, query_cat_id_list, anno_list):
        """

        Randomly sample one class that exist in the target image. Filter the ground truth.
        Load random query image based on the classes of the current search image.

        Args:
            data: dict containing the search image and its annotation

        Returns:
            diction containing data and new loaded cropped image

        """

        query_data = None
        query_label_id_list = []
        query_img_list = []
        query_targets_list = []
        query_label_list = []
        query_img_meta_list = []
        for i, query_cat_id in enumerate(query_cat_id_list):

            while query_data is None:
                query_anno = anno_list[i][random.randint(0, len(anno_list[i]) - 1)]
                query_img_id = query_anno['image_id']
                query_data = self.prepare_query_img(query_anno, query_img_id)

            query_label_id = self.cat2label[query_cat_id]
            query_label_id_list.append(query_label_id)
            query_img_list.append(query_data['img'].data)
            query_img_meta_list.append(query_data['img_metas'].data)
            query_targets_list.append(data['gt_bboxes'].data[data['gt_labels'].data == query_label_id])
            query_label_list.append(torch.full((len(query_targets_list[-1]), ), i))
            query_data = None

        data['query_img_metas'] = DataContainer(query_img_meta_list)
        data['query_img_list'] = DataContainer(query_img_list)
        data['query_label_id_list'] = DataContainer(query_label_id_list)
        data['query_targets_list'] = DataContainer(query_targets_list)
        data['query_labels_list'] = DataContainer(query_label_list)

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

    def prepare_train_img(self, idx):
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
        return self.transform_pipeline(results)

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
            query_cat_id_list = []
            for query_cat_id in list(known_interset_anns):
                if query_cat_id in self.class_anno_mapping:
                    query_cat_id_list.append(query_cat_id)

            if not query_cat_id_list:
                idx = self._rand_another(idx)
                continue

            # check test classes presences
            if self.no_test_class_present:
                unknown_interset_anns = ann_cat_ids.intersection(self.unknown_cats_ids)
                if unknown_interset_anns:
                    idx = self._rand_another(idx)
                    continue

            anno_list = []
            for query_cat_id in query_cat_id_list:
                anno_list.append(self.class_anno_mapping[query_cat_id])

            return idx, query_cat_id_list, anno_list


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
            idx, query_cat_id_list, query_ann_list = self.check_train_img(idx)
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            data = self.generate_oneshot_data(data, query_cat_id_list, query_ann_list)
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
        if self.test_type == 'all':
            gt_class_labels = set(ann_info['labels'])
        elif self.test_type == 'unknown_only':
            gt_class_labels = set(self.unknown_cats_labels)
        elif self.test_type == 'known_only':
            gt_class_labels = set(self.known_cats_labels)

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

        # if self.test_type == 'all':
        #     test_cats = self.cat_ids
        # elif self.test_type == 'unknown_only':
        #     test_cats = self.unknown_cats_ids
        # elif self.test_type == 'known_only':
        #     test_cats = self.known_cats_ids
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
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        # create multiple run results list
        results_files_list = []
        for i in range(self.avg_num):
            tmp_result = [item[i] for item in results]
            results_files_list.append(tmp_result)

        different_runs_per_class_res = []
        different_runs_mean_aps = []

        for result in results_files_list:

            if metric == 'mAP':
                assert isinstance(iou_thrs, list)
                # if self.year == 2007:
                #     ds_name = 'voc07'
                # else:
                #     ds_name = self.CLASSES
                ds_name = 'voc07'
                mean_aps = []

                for iou_thr in iou_thrs:
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    # Follow the official implementation,
                    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                    # we should use the legacy coordinate system in mmdet 1.x,
                    # which means w, h should be computed as 'x2 - x1 + 1` and
                    # `y2 - y1 + 1`
                    mean_ap, per_class_eval_results = eval_map(
                        result,
                        annotations,
                        scale_ranges=None,
                        iou_thr=iou_thr,
                        dataset=ds_name,
                        logger='silent',
                        use_legacy_coordinate=True)
                    if iou_thr == 0.5:
                        different_runs_per_class_res.append(per_class_eval_results)
                        different_runs_mean_aps.append(mean_ap)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
                eval_results.move_to_end('mAP', last=False)

            elif metric == 'recall':
                gt_bboxes = [ann['bboxes'] for ann in annotations]
                recalls = eval_recalls(
                    gt_bboxes,
                    result,
                    proposal_nums,
                    iou_thrs,
                    logger=logger,
                    use_legacy_coordinate=True)
                for i, num in enumerate(proposal_nums):
                    for j, iou_thr in enumerate(iou_thrs):
                        eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
                if recalls.shape[1] > 1:
                    ar = recalls.mean(axis=1)
                    for i, num in enumerate(proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]

        avg_per_class_eval_results = [{'num_gts': 0, 'num_dets': 0,
                                       'recall': np.array([0.0]), 'ap': 0.0} for _ in range(20)]
        for result_list in different_runs_per_class_res:
            for i, item in enumerate(result_list):
                avg_per_class_eval_results[i]['num_gts'] += item['num_gts']
                avg_per_class_eval_results[i]['num_dets'] += item['num_dets']
                if item['recall'].size > 0:
                    avg_per_class_eval_results[i]['recall'] += np.array(item['recall'], ndmin=2)[:, -1]
                avg_per_class_eval_results[i]['ap'] += item['ap']

        for i, item in enumerate(avg_per_class_eval_results):
            avg_per_class_eval_results[i] = {key: value / self.avg_num
                                             for key, value in item.items()}
        mean_ap_final = sum(different_runs_mean_aps)/self.avg_num


        print_map_summary(mean_ap_final, avg_per_class_eval_results, dataset='voc07')

        return eval_results


