import subprocess

# define Euler related parameters
gpu = '4'
time='4'
mem = '8000'
n_cores = '8'


# define task related parameters

name = 'adet_cvt13_2x_search_1'
# config= 'configs/adet/conv_adet_cvt13_2x_voc.py'
config= 'configs/adet/adet_cvt13_2x_voc.py'
gpu_type = "NVIDIAGeForceRTX2080Ti"

dataset = 'voc'

port=22266

per_gpu_batch_size=2
average_num = str(2)
init_checkpoint = '1k'
no_test_class_present =str(False)
split_list = [0]
att_type = 'all'
lr = str(0.001)
bg_crop_freq = [0.25, 0.5]
bg_gt_overlap_iou = [0.3, 1]

for i in split_list:
    for j in bg_crop_freq:
        for k in bg_gt_overlap_iou:
            split = str(i)
            port_str = str(port)
            bgf = str(j)
            bgoi = str(k)
            # if j == 0 and k==0.3:
            #     continue
            log_name = name + '_' + dataset +  '_split_' + split + '_' \
                       + init_checkpoint + '_NoTestClass_' \
                       + no_test_class_present + '_lr_' + lr + '_bs_' \
                       + str(per_gpu_batch_size*int(gpu)) + '_bg_freq_' + bgf + '_bg_overlap_' + bgoi
            json_prefix = 'results/'+ log_name + '_split_'+split
            checkpoint = f'saved_models/cvt_weights/CvT-13-384x384-IN-{init_checkpoint}-backbone.pth'


            command = f" --work-dir saved_models/{log_name}/ --auto-resume " \
                      f"--cfg-options load_from={checkpoint} data.samples_per_gpu={per_gpu_batch_size} " \
                      f"data.train.split={split} data.val.split={split} data.test.split={split} " \
                      f"data.train.average_num={average_num} data.val.average_num={average_num} data.test.average_num={average_num} " \
                      f"data.train.no_test_class_present={no_test_class_present} " \
                      f"optimizer.lr={lr} data.train.bg_crop_freq={bgf} data.train.bg_gt_overlap_iou={bgoi} " \
                      # f"evaluation.jsonfile_prefix={json_prefix} "

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