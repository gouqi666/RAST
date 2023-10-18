'''
SQuAD_1.1_split1
'''
# is_use_skeleton,output_file,flag
mode  
1. vanilla qg  
   is_use_skeleton = False
   flag = processed
   output_file = experiments_vanilla
   is_use_skeleton=False
2. init training
   is_use_skeleton = True
   flag = processed_with_null_skeleton
   output_file = experiments_nucleus_sampling
   is_use_skeleton=True
SQuAD_1.1_split_1   
```
home=${home_path}
max_source_length=128
dataset=SQuAD_1.1_split1
output_file=experiments_nucleus_sampling
xpython=${home_path}/env/v100/bin/python
file=$home/qg/run_qg.py
flag=processed
$xpython    $file \
--train_data_dir=$home/data/${dataset}/${flag}/train.jsonl \
--valid_data_dir=$home/data/${dataset}/${flag}/dev.jsonl \
--test_data_dir=$home/data/${dataset}/${flag}/test.jsonl \
--output_dir=$home/output/${dataset}/QG/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=16 \
--per_device_train_batch_size=64 \
--do_sample=True \
--top_p=0.9 \
--top_k=50 \
--eval_beams=5 \
--num_beam_groups=1 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--num_return_sequences=5 \
--is_use_skeleton=False \
--model_name_or_path=t5-base \
--warmup_ratio=0.1 \
--gradient_accumulation_steps=1 \
--task=QG \
--n_train=-1 \
--n_val=-1 \
--temperature=1.8

```




SQuAD_1.1_split2
```
# is_use_skeleton,output_file,flag
export CUDA_VISIBLE_DEVICES=0
home=${home_path}
max_source_length=256
dataset=SQuAD_1.1_split2
output_file=experiments_nucleus_sampling
xpython=${home_path}/env/v100/bin/python
file=$home/qg/run_qg.py
flag=processed
$xpython    $file \
--train_data_dir=$home/data/${dataset}/${flag}/train.jsonl \
--valid_data_dir=$home/data/${dataset}/${flag}/dev.jsonl \
--test_data_dir=$home/data/${dataset}/${flag}/test.jsonl \
--output_dir=$home/output/${dataset}/QG/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=8 \
--per_device_train_batch_size=32 \
--do_sample=True \
--top_p=0.9 \
--top_k=50 \
--eval_beams=5 \
--num_beam_groups=1 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--num_return_sequences=5 \
--is_use_skeleton=False \
--model_name_or_path=t5-base \
--warmup_ratio=0.1 \
--gradient_accumulation_steps=1 \
--task=QG \
--n_train=-1 \
--n_val=-1 \
--temperature=1.8
```


newsqa
```
export CUDA_VISIBLE_DEVICES=1
home=${home_path}
max_source_length=1250
dataset=newsqa
output_file=experiments_nucleus_sampling_v2
xpython=$home/env/v100/bin/python
file=$home/qg/run_qg.py
flag=processed
$xpython    $file \
--train_data_dir=$home/data/${dataset}/${flag}/train.jsonl \
--valid_data_dir=$home/data/${dataset}/${flag}/dev.jsonl \
--test_data_dir=$home/data/${dataset}/${flag}/test.jsonl \
--output_dir=$home/output/${dataset}/QG/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=2 \
--per_device_train_batch_size=6 \
--do_sample=True \
--top_p=0.9 \
--top_k=50 \
--eval_beams=5 \
--num_beam_groups=1 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--num_return_sequences=5 \
--is_use_skeleton=True \
--model_name_or_path=t5-base \
--warmup_ratio=0.1 \
--gradient_accumulation_steps=1 \
--task=QG \
--n_train=-1 \
--n_val=-1 \
--temperature=1.5
```