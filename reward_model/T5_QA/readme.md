
### SQuAD_1.1_split1
```
CUDA_VISIBLE_DEVICES='2'
home=${home_path}
max_source_length=128
batch_size=64
model_name_or_path=t5-base
gradient_accumulation_steps=1
dataset=SQuAD_1.1_split1
epoch=5
output_file=experiments
learning_rate=2e-4

xpython=$home/env/v100/bin/python
file=$home/reward_model/T5_QA/run_qa.py

$xpython    $file \
--num_train_epochs=$epoch \
--learning_rate=$learning_rate \
--train_data_dir=$home/data/${dataset}/processed/train.jsonl \
--valid_data_dir=$home/data/${dataset}/processed/dev.jsonl \
--test_data_dir=$home/data/${dataset}/processed/test.jsonl \
--output_dir=$home/output/${dataset}/QA/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=${batch_size} \
--per_device_train_batch_size=${batch_size} \
--model_name_or_path=${model_name_or_path} \
--gradient_accumulation_steps=${gradient_accumulation_steps}
```



### SQuAD_1.1_split2
```
export CUDA_VISIBLE_DEVICES=0
home=${home_path}
max_source_length=512
batch_size=16
model_name_or_path=t5-base
gradient_accumulation_steps=1
dataset=SQuAD_1.1_split2
epoch=5
output_file=experiments
learning_rate=2e-4

xpython=$home/env/v100/bin/python
file=$home/reward_model/T5_QA/run_qa.py

$xpython    $file \
--num_train_epochs=$epoch \
--learning_rate=$learning_rate \
--train_data_dir=$home/data/${dataset}/processed/train.jsonl \
--valid_data_dir=$home/data/${dataset}/processed/dev.jsonl \
--test_data_dir=$home/data/${dataset}/processed/test.jsonl \
--output_dir=$home/output/${dataset}/QA/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=${batch_size} \
--per_device_train_batch_size=${batch_size} \
--model_name_or_path=${model_name_or_path} \
--gradient_accumulation_steps=${gradient_accumulation_steps}
```

### newsqa
```
export CUDA_VISIBLE_DEVICES=0
home=${home_path}
max_source_length=1250
batch_size=8
model_name_or_path=t5-base
gradient_accumulation_steps=1
dataset=newsqa
epoch=5
output_file=experiments
learning_rate=2e-4

xpython=$home/env/v100/bin/python
file=$home/reward_model/T5_QA/run_qa.py

$xpython    $file \
--num_train_epochs=$epoch \
--learning_rate=$learning_rate \
--train_data_dir=$home/data/${dataset}/processed/train.jsonl \
--valid_data_dir=$home/data/${dataset}/processed/dev.jsonl \
--test_data_dir=$home/data/${dataset}/processed/test.jsonl \
--output_dir=$home/output/${dataset}/QA/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=${batch_size} \
--per_device_train_batch_size=${batch_size} \
--model_name_or_path=${model_name_or_path} \
--gradient_accumulation_steps=${gradient_accumulation_steps}
```



### eval qa and f1 result of RAST  
```
# 先用convert_rast_result2qa.py 将output_data.json转化为qa.json,然后train,valid,test dir全部改成对应路径，do_train=False
export CUDA_VISIBLE_DEVICES=2
home=${home_path}
max_source_length=128
batch_size=64
model_name_or_path=${home_path}/RAST/output/SQuAD_1.1_split1/QA/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b64-beam_search5/0.7866302071131831
gradient_accumulation_steps=1
dataset=SQuAD_1.1_split1
epoch=5
output_file=experiments_test_v23
learning_rate=2e-4

xpython=$home/env/v100/bin/python
file=$home/reward_model/T5_QA/run_qa.py

$xpython    $file \
--num_train_epochs=$epoch \
--learning_rate=$learning_rate \
--train_data_dir=${home_path}/RAST/output/SQuAD_1.1_split1/RAG/experiments_v23/15.582719976811617/qa_data.jsonl \
--valid_data_dir=${home_path}/RAST/output/SQuAD_1.1_split1/RAG/experiments_v23/15.582719976811617/qa_data.jsonl \
--test_data_dir=${home_path}/RAST/output/SQuAD_1.1_split1/RAG/experiments_v23/15.582719976811617/qa_data.jsonl \
--output_dir=$home/output/${dataset}/QA/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=${batch_size} \
--per_device_train_batch_size=${batch_size} \
--model_name_or_path=${model_name_or_path} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--do_train=False
```

### experiment of QG for QA data augmentation

1. predict the output of training data -> output_data.json  
2. train for qa

```
export CUDA_VISIBLE_DEVICES=2
home=${home_path}
max_source_length=150
batch_size=64
model_name_or_path=${home_path}/RAST/output/SQuAD_1.1_split1/QA/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b64-beam_search5/0.7866302071131831
gradient_accumulation_steps=1
dataset=SQuAD_1.1_split1
epoch=5
output_file=experiments_DA_train_output_0.2_v4
learning_rate=2e-4

xpython=$home/env/v100/bin/python
file=$home/reward_model/T5_QA/run_qa.py

$xpython    $file \
--num_train_epochs=$epoch \
--learning_rate=$learning_rate \
--train_data_dir=$home/data/${dataset}/processed/train.jsonl \
--valid_data_dir=$home/data/${dataset}/processed/dev.jsonl \
--test_data_dir=$home/data/${dataset}/processed/test.jsonl \
--output_dir=$home/output/${dataset}/QA/${output_file} \
--max_source_length=${max_source_length} \
--per_device_eval_batch_size=${batch_size} \
--per_device_train_batch_size=${batch_size} \
--model_name_or_path=${model_name_or_path} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--add_extra_data=True
```