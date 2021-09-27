PATH=$PATH:/opt/conda/bin:/opt/conda/condabin:/userhome/cbh/DOWNLOAD/cuda/cuda/bin:/userhome/cbh/DOWNLOAD/cuda/cuda/bin; source activate;

# tag=p16-1_g0-3
# tag=p16-1_g4-7
# tag=p16-1_g8-11
# tag=p16-1_g12-15

# tag=p16-2_g0-3
# tag=p16-2_g4-7
# tag=p16-2_g8-11
# tag=p16-2_g12-15

# tag=p16-3_g0-3
# tag=p16-3_g4-7
# tag=p16-3_g8-11
# tag=p16-3_g12-15


# tag=p16-4_g0-3
# tag=p16-4_g4-7
tag=p16-4_g8-11
# tag=p16-4_g12-15

# tag=p4-2_g0-3


gpu_num=2
data_loader_workers=4
data_transforms=imagenet1k_mobile

# batch_size=32
batch_size=256
# batch_size=512

# dataset_dir=/dev/shm/DATASET/ImageNet10
dataset_dir=/dev/shm/DATASET/ImageNet100
# dataset_dir=/dev/shm/DATASET/ImageNet

# n_classes=10
# n_classes=100
n_classes=1000

# epochs=10
epochs=250


print_freq=100
valid_all_freq=250
valid_last_epoch=2
valid_topk=5

optimizer=sgd
momentum=0.9
nesterov=True
lr_scheduler=cosine
reset_parameters=True

random_seed=0
# random_seed=2000




converge_to_pool=exp_rank

# lr=0.1
# lr=0.2
# lr=0.3
lr=0.4
# lr=0.5
# lr=0.6





# --width_mult_range 0.75 1.0 \
# --resolution_range 224 96 \
# --resolution_step 8 \

#* mbv1
model_name=mobilenet_v1

##* latency
# infer_metric_type=latency
# --infer_metric_target_range 60 570 \
# --infer_metric_target_step 10 \
lut_dir=/userhome/cbh/EXP/21.03/0802/22:22:35-734674940_mobilenet_v1_gpu_iters=1000/lut_ckpt/full.pkl
# lut_dir=/userhome/cbh/EXP/21.03/0802/23:17:03-318177652_mobilenet_v1_cpu_iters=1000/lut_ckpt/full.pkl

##* flops
infer_metric_type=flops
# --infer_metric_target_range 60 570 \
# --infer_metric_target_step 10 \




#* mbv2
# model_name=mobilenet_v2

##* latency
# infer_metric_type=latency
# --infer_metric_target_range 60 570 \
# --infer_metric_target_step 10 \
# lut_dir=/userhome/cbh/EXP/21.03/0802/22:23:38-543150450_mobilenet_v2_gpu_iters=1000/lut_ckpt/full.pkl
# lut_dir=/userhome/cbh/EXP/21.03/0802/23:17:29-221131591_mobilenet_v2_cpu_iters=1000/lut_ckpt/full.pkl

##* flops
# infer_metric_type=flops
# --infer_metric_target_range 40 300 \
# --infer_metric_target_step 10 \



sample_type=max_randc_min
num_subnet_training=2

weight_decay=1e-4

data_backend=pytorch


# test_only=True
# pretrained='/userhome/cbh/EXP/21.03/0702/0702-23:15:41-344694168_MuFix_ds=im1000_e=25_bs=_tag=p16-1_g8-9/imagenet1k-mobilenet_v2/checkpoint_024.pt'
# gpu_num=1
# data_loader_workers=4

# --test_only $test_only \
# --pretrained $pretrained \

wandb offline
wandb_project=test
# job_dir=/userhome/cbh/EXP/tmp/$(date +%m%d/%m%d-%T-%N)
job_dir=`pwd`
########################################################
# wandb online
# wandb_project=aaai_22_munet
# date=$(date +%m%d/%T-%N)
# job_dir=/userhome/cbh/EXP/21.03/$date\_MuFix_ds=im$n_classes\_e=$epochs\_bs=$batch_size\_db=$data_backend\_lr=$lr\_wd=$weight_decay\_tag=$tag
# git add -A
# git cmm "[run_id] $date"
# git pm
########################################################
echo $job_dir >> job_dir.log

sampler_metric_target_map_path=$job_dir/metric_target.map

python train.py \
--data_transforms $data_transforms \
--data_loader_workers $data_loader_workers \
--dataset_dir $dataset_dir \
--n_classes $n_classes \
--epochs $epochs \
--print_freq $print_freq \
--valid_all_freq $valid_all_freq \
--optimizer $optimizer \
--momentum $momentum \
--weight_decay $weight_decay \
--nesterov $nesterov \
--lr_scheduler $lr_scheduler \
--random_seed $random_seed \
--job_dir $job_dir \
--model_name $model_name \
--reset_parameters $reset_parameters \
--lr $lr \
--batch_size $batch_size \
--gpu_num $gpu_num \
--sampler_metric_target_map_path $sampler_metric_target_map_path \
--converge_to_pool $converge_to_pool \
--num_subnet_training $num_subnet_training \
--sample_type $sample_type \
--data_backend $data_backend \
--valid_last_epoch $valid_last_epoch \
--wandb_project $wandb_project \
--valid_topk $valid_topk \
--infer_metric_type $infer_metric_type \
--lut_dir $lut_dir \
--profiling gpu \
--infer_metric_target_range 60 570 \
--infer_metric_target_step 10 \
--width_mult_range 0.75 1.0 \
--resolution_range 224 96 \
--resolution_step 8

rm -rf $job_dir/code/.git
