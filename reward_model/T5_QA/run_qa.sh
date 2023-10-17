# python -m torch.distributed.launch \
# --nproc_per_node=4 \
# --nnodes=1 \


cuda_visible_devices=2
home=/home/student2020/gouqi/RAST
if [ $dataset == SQuAD_1.1_split1 ] 
then 
    max_source_length=128
    batch_size=64
    model_name_or_path=t5-base
    gradient_accumulation_steps=1
elif [ $dataset == SQuAD_1.1_split2 ] 
then
    max_source_length=512
    batch_size=16 
    model_name_or_path=t5-base
    gradient_accumulation_steps=1
elif [ $dataset == newsqa ] 
then
    max_source_length=1250
    batch_size=16 
    model_name_or_path=t5-base
    gradient_accumulation_steps=1 # 16
fi
xpython=$home/env/v100/bin/python
file=$home/reward_model/T5_QA/run_qa.py
$xpython    $file \
--num_train_epochs=$epoch \
--learning_rate=$learning_rate \
--add_extra_data=$add_extra_data \
--train_data_dir=$home/data/${dataset}/processed/train.jsonl \
--valid_data_dir=$home/data/${dataset}/processed/dev.jsonl \
--test_data_dir=$home/data/${dataset}/processed/test.jsonl \
--output_dir=$home/output/${dataset}/QA/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=${batch_size} \
--per_device_train_batch_size=${batch_size} \
--model_name_or_path=${model_name_or_path} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \