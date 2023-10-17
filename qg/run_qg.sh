if [ $dataset == SQuAD_1.1_split1 ] 
then 
    max_source_length=128
    eval_batch_size=16
    model_name_or_path=t5-base
    gradient_accumulation_steps=1
    num_return_sequences=5
    task=QG
    learning_rate=5e-5
elif [ $dataset == SQuAD_1.1_split2 ] 
then
    max_source_length=512
    batch_size=16
    eval_batch_size=4
    model_name_or_path=t5-base
    gradient_accumulation_steps=1
    num_return_sequences=5
    task=QG
    learning_rate=5e-5
elif [ $dataset == newsqa ] 
then
    max_source_length=1250
    batch_size=16 # 16
    eval_batch_size=4
    model_name_or_path=t5-base
    gradient_accumulation_steps=16
    num_return_sequences=5
    task=QG
    learning_rate=5e-5
elif [ $dataset == data_qqp ] 
then
    max_source_length=64
    batch_size=32
    eval_batch_size=16
    model_name_or_path=t5-base
    gradient_accumulation_steps=1
    num_return_sequences=3
    task=paraphrase_generation
elif [ $dataset == data_mscoco ] 
then
    max_source_length=64
    batch_size=32
    eval_batch_size=16
    model_name_or_path=t5-base
    gradient_accumulation_steps=1
    num_return_sequences=3
fi
xpython=python
file=/mnt/ruzhen/ruzhen/RQG/qg/run_qg.py
if [ $use_dlc == True ] 
then 
    xpython=/root/data/ruzhen/envs/a100/bin/python
    file=/root/data/ruzhen/RQG/qg/run_qg.py
fi
$xpython    $file \
--use_dlc=$use_dlc \
--train_data_dir=RQG/data/${dataset}/${flag}/train.jsonl \
--valid_data_dir=RQG/data/${dataset}/${flag}/dev.jsonl \
--test_data_dir=RQG/data/${dataset}/${flag}/test.jsonl \
--output_dir=RQG/output/${dataset}/${task}/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=${eval_batch_size} \
--per_device_train_batch_size=${batch_size} \
--do_sample=${do_sample} \
--top_p=${top_p} \
--top_k=${top_k} \
--eval_beams=${num_beams} \
--num_beam_groups=${num_beam_groups} \
--learning_rate=${learning_rate} \
--num_train_epochs=${epochs} \
--num_return_sequences=${num_return_sequences} \
--is_use_skeleton=${is_use_skeleton} \
--model_name_or_path=${model_name_or_path} \
--warmup_ratio=${warmup_ratio} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--task=${task} \
--n_train=-1 \
--n_val=-1 \
--temperature=${temperature} \