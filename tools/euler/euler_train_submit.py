import subprocess

# define Euler related parameters
gpu = '8'
time='24'
mem = '8000'
n_cores = '20'


# define task related parameters

name = 'adet_cvt13_2x_search_1'
config= 'configs/adet/adet_cvt13_2x.py'
gpu_type = "NVIDIAGeForceRTX2080Ti"

dataset = 'coco'
# if dataset == 'coco':
#     ann_file = 'data/lvis/annotations/instances_train2017.json'
# elif dataset == 'voc':
#     ann_file = 'data/lvis/annotations/lvis-0.5_coco2017_train.json'

port=22222

per_gpu_batch_size=2
average_num = str(2)
init_checkpoint = '1k'
no_test_class_present =str(False)
split_list = [3]
lr = str(0.002)
bg_crop_freq = [0.5]
bg_gt_overlap_iou = [1]

for i in split_list:
    for j in bg_crop_freq:
        for k in bg_gt_overlap_iou:
            split = str(i)
            port_str = str(port)
            # if j == 0 and k==1:
            #     continue
            bgf = str(j)
            bgoi = str(k)
            log_name = name + '_' + dataset + '_split_' + split + '_' \
                       + init_checkpoint + '_NoTestClass_' \
                       + no_test_class_present + '_lr_' + lr + '_bs_' \
                       + str(per_gpu_batch_size * int(gpu)) + '_bg_freq_' + bgf + '_bg_overlap_' + bgoi
            json_prefix = 'results/' + name + '_split_' + split
            checkpoint = f'saved_models/cvt_weights/CvT-13-384x384-IN-{init_checkpoint}-backbone.pth'

            command = f" --work-dir saved_models/{log_name}/ --resume-from saved_models/adet_cvt13_2x_search_1_coco_split_3_1k_NoTestClass_False_lr_0.002_bs_16_bg_freq_0.5_bg_overlap_1/epoch_16.pth " \
                      f"--cfg-options load_from={checkpoint} data.samples_per_gpu={per_gpu_batch_size} " \
                      f"data.train.split={split} data.val.split={split} data.test.split={split} " \
                      f"data.train.average_num={average_num} data.val.average_num={average_num} data.test.average_num={average_num} " \
                      f"data.train.no_test_class_present={no_test_class_present} evaluation.jsonfile_prefix={json_prefix} " \
                      f"optimizer.lr={lr} data.train.bg_crop_freq={bgf} data.train.bg_gt_overlap_iou={bgoi} " \

            process = subprocess.Popen(
                ['./tools/bsub_train.sh', config, gpu, log_name, command, time, port_str, mem, n_cores, gpu_type],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            stdout, stderr = process.communicate()
            print(stdout)
            print(stderr)
            port += 2