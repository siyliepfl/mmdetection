import subprocess

# define Euler related parameters
gpu = '8'
time='120'
mem = '8000'
n_cores = '16'


# define task related parameters

name = 'adet_cvt13_150e'
config= 'configs/adet/adet_cvt13_150e.py'
checkpoint = 'saved_models/cvt_weights/CvT-13-384x384-IN-22k-backbone.pth'
gpu_type = "NVIDIAGeForceRTX2080Ti"

dataset = 'coco'
if dataset == 'coco':
    ann_file = 'data/lvis/annotations/instances_train2017.json'
elif dataset == 'lvis':
    ann_file = 'data/lvis/annotations/lvis-0.5_coco2017_train.json'

port=22222

batch_size=2
average_num = str(1)
# lr=0.001
split_list = [0, 1, 2, 3]

for i in split_list:
    split = str(i)
    one_shot_data = 'data/lvis/' + f'oneshot/train_split_{split}.txt'
    port_str = str(port)
    log_name = name + '_' + dataset +  '_split_' + split + '_average_num_'+str(average_num)
    json_prefix = 'results/'+ name + '_split_'+split
    command = f" --work-dir saved_models/{log_name}/ " \
              f"--cfg-options load_from={checkpoint} data.samples_per_gpu={batch_size} " \
              f"evaluation.jsonfile_prefix={json_prefix} data.train.split={split} data.val.split={split} data.test.split={split} " \
              f"data.train.average_num={average_num} data.val.average_num={average_num} data.test.average_num={average_num} " \
              f"data.train.classes={one_shot_data} " \


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