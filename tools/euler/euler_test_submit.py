import subprocess

# define Euler related parameters
gpu = '4'
time='4'
mem = '8000'
n_cores = '8'

# define task related parameters

name = 'TEST_adet_cvt13_2x'
config= 'configs/adet/adet_cvt13_2x.py'
# checkpoint = 'saved_models/cvt_weights/CvT-13-384x384-IN-22k-backbone.pth'
gpu_type = "NVIDIAGeForceRTX2080Ti"

dataset = 'coco'
# if dataset == 'coco':
#     ann_file = 'data/lvis/annotations/instances_train2017.json'
# elif dataset == 'lvis':
#     ann_file = 'data/lvis/annotations/lvis-0.5_coco2017_train.json'

port=22343
init_checkpoint = '1k'
batch_size=2
average_num = str(5)
lr=str(0.002)
split_list = [3]
bg_crop_freq = [0.5]
bg_gt_overlap_iou = [1]
classwise=True
no_test_class_present =str(False)

for i in split_list:
    for j in bg_crop_freq:
        for k in bg_gt_overlap_iou:

            bgf = str(j)
            bgoi = str(k)
            split = str(i)
            # one_shot_data = 'data/lvis/' + f'oneshot/train_split_{split}.txt'
            port_str = str(port)
            log_name = name + '_' + dataset + '_split_' + split + '_' \
                       + init_checkpoint + '_NoTestClass_' \
                       + no_test_class_present + '_lr_' + lr + '_bs_' \
                       + str(16) + '_bg_freq_' + bgf + '_bg_overlap_' + bgoi
            json_prefix = 'results/'+ name + '_split_'+split
            # checkpoint = f'saved_models/adet_cvt13_2xbj_coco_split_{split}_average_num_2/epoch_24.pth'
            # checkpoint = f'saved_models/adet_cvt13_2x_search_1_coco_split_{split}_1k_NoTestClass_True/epoch_24.pth'
            checkpoint = f'saved_models/adet_cvt13_2x_search_1_coco_split_{split}_1k_NoTestClass_False_lr_0.002_bs_16_bg_freq_{bgf}_bg_overlap_{bgoi}/epoch_24.pth'
            command = f"--tmpdir /scratch/{log_name}/ --eval bbox " \
                      f"--eval-options jsonfile_prefix={json_prefix} classwise={classwise} " \
                      f"--cfg-options data.test.split={split} " \
                      f"data.test.average_num={average_num} "
                      # f"data.test.classes={one_shot_data} " \

            process = subprocess.Popen(
                ['./tools/bsub_test.sh', config,checkpoint, gpu, log_name, command, time, port_str,mem,n_cores,gpu_type],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, stderr = process.communicate()
            print(stdout)
            print(stderr)
            port += 2