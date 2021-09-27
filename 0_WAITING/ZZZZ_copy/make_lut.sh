# model_name=mobilenet_v1
model_name=mobilenet_v2

# device=gpu
device=cpu

# gpu_id=0
gpu_id=1

warmup=100
iters=1000


# job_dir=/userhome/cbh/EXP/tmp/$(date +%m%d/%m%d-%T-%N)
##################################################
job_dir=/userhome/cbh/EXP/21.03/$(date +%m%d/%T-%N)\_$model_name\_$device\_iters=$iters
echo $job_dir >> job_dir.log

mkdir -p $job_dir
rm -rf $job_dir/code
cp -r ./ $job_dir/code


taskset -c 85 python $job_dir/code/make_lut.py \
--model_name $model_name \
--device $device \
--warmup $warmup \
--iters $iters \
--gpu_id $gpu_id \
--job_dir $job_dir \
2>&1 \
| tee $job_dir/stdout_full.log






