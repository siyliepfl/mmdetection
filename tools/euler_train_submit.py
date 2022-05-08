import subprocess

# define Euler related parameters
gpu = '4'
time='24'
mem = '8000'
n_cores = '8'


# define task related parameters

name = 'voc_adet_cvt13_2x_search_1'
config= 'configs/adet/adet_cvt13_2x_voc.py'
gpu_type = "NVIDIAGeForceRTX2080Ti"

dataset = 'voc'
# if dataset == 'coco':
#     ann_file = 'data/lvis/annotations/instances_train2017.json'
# elif dataset == 'voc':
#     ann_file = 'data/lvis/annotations/lvis-0.5_coco2017_train.json'

port=22222

batch_size=2
average_num = str(2)
init_checkpoint = '1k'
no_test_class_present =str(False)
split_list = [0]
lr = str(0.001)
for i in split_list:
    split = str(i)
    # one_shot_data = 'data/lvis/' + f'oneshot/train_split_{split}.txt'
    port_str = str(port)
    log_name = name + '_' + dataset +  '_split_' + split + '_' + init_checkpoint + '_NoTestClass_' + no_test_class_present
    json_prefix = 'results/'+ name + '_split_'+split
    checkpoint = f'saved_models/cvt_weights/CvT-13-384x384-IN-{init_checkpoint}-backbone.pth'


    command = f" --work-dir saved_models/{log_name}/ " \
              f"--cfg-options load_from={checkpoint} data.samples_per_gpu={batch_size} " \
              f"data.train.split={split} data.val.split={split} data.test.split={split} " \
              f"data.train.average_num={average_num} data.val.average_num={average_num} data.test.average_num={average_num} " \
              f"data.train.no_test_class_present={no_test_class_present} " \
              f"optimizer.lr={lr} " \
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