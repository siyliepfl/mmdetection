import subprocess

# define Euler related parameters
gpu = '2'
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

port=22222

batch_size=2
average_num = str(5)
# lr=0.001
split_list = [0,1,2, 3]
classwise=True
no_test_class_present =str(True)

for i in split_list:
    split = str(i)
    # one_shot_data = 'data/lvis/' + f'oneshot/train_split_{split}.txt'
    port_str = str(port)
    log_name = name + '_' + dataset +  '_split_' \
            + split + '_average_num_'+str(average_num) +'_NoTestClass_' + no_test_class_present
    json_prefix = 'results/'+ name + '_split_'+split
    # checkpoint = f'saved_models/adet_cvt13_2x_coco_split_{split}_average_num_2/epoch_24.pth'
    checkpoint = f'saved_models/adet_cvt13_2x_search_1_coco_split_{split}_1k_NoTestClass_True/epoch_24.pth'
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