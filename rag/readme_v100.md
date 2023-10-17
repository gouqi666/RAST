### SQuAD_1.1_split1
params:

```
export CUDA_VISIBLE_DEVICES=1
home=${home_path}
external_dataset=SQuAD_1.1_split1
dataset=SQuAD_1.1_split1
generator_name_or_path=$home/output/SQuAD_1.1_split1/QG/experiments_with_null_skeleton/51.801828729175455 \
reward_model_name_or_path=$home/output/SQuAD_1.1_split1/QA/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b64-beam_search5/0.7866302071131831 \
vanilla_qg_model_name_or_path=$home/output/SQuAD_1.1_split1/QG/experiments/t5-base-GPUNums1-len128-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b64-top_k50-top_p0.9-temperature1.0/20.827964423502124 \
question_encoder_name_or_path=facebook/dpr-question_encoder-single-nq-base
eval_model_name_or_path=None
ctx_retrieve_key=question_skeleton
task=question_generation

xpython=$home/env/v100/bin/python
file=$home/rag/run_rag.py
$xpython    $file \
--generator_name_or_path=${generator_name_or_path} \
--reward_model_name_or_path=${reward_model_name_or_path} \
--vanilla_qg_model_name_or_path=${vanilla_qg_model_name_or_path} \
--dataset_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_${ctx_retrieve_key} \
--index_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_hnsw_index_${ctx_retrieve_key}.faiss \
--eval_model_name_or_path=${eval_model_name_or_path} \
--data_dir=$home/data/${dataset}/processed \
--output_dir=$home/output/${dataset}/RAG/experiments_v24/ \
--max_source_length=128 \
--max_combined_length=128 \
--train_dpr=True \
--set_context_encoder_for_training=False \
--do_train=True \
--do_eval=False \
--num_train_epochs=7 \
--retrieve_key=${ctx_retrieve_key} \
--ctx_retrieve_key=${ctx_retrieve_key} \
--n_questions=3 \
--eval_n_questions=5 \
--train_retrieve_num=100 \
--eval_retrieve_num=500 \
--per_device_train_batch_size=12 \
--per_device_eval_batch_size=10 \
--reward_type=qa  \
--add_rl_base_reward=True \
--sm_reward_type=jaccard \
--add_sm_base_reward=False \
--whiten=False \
--learning_rate=1e-5 \
--dpr_learning_rate=1e-6 \
--lamda=0.5  \
--sm_coef=0.1 \
--gradient_accumulation_steps=1 \
--reward_batch_normalization=True \
--eval_steps=2000 \
--train_filter_type=v3 \
--eval_filter_type=v0_1 \
--task=question_generation \
--n_train=-1 \
--n_val=-1 \
--do_sample=True \
--use_bm25=False \
--question_encoder_name_or_path=${question_encoder_name_or_path} \
--seed=42
```

### SQuAD_1.1_split2
```
export CUDA_VISIBLE_DEVICES=0
home=${home_path}
external_dataset=SQuAD_1.1_split2
dataset=SQuAD_1.1_split2
generator_name_or_path=$home/output/SQuAD_1.1_split2/QG/experiments_with_null_skeleton/t5-base-GPUNums1-len256-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b32-top_k50-top_p0.9-temperature1.0/48.92814408153402 \
reward_model_name_or_path=$home/output/SQuAD_1.1_split2/QA/experiments/t5-base-GPUNums1-len512-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b16-beam_search5/0.832362581054675 \
vanilla_qg_model_name_or_path=$home/output/SQuAD_1.1_split2/QG/experiments/t5-base-GPUNums1-len256-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b32-top_k50-top_p0.9-temperature1.0/22.752691870599858 \
question_encoder_name_or_path=facebook/dpr-question_encoder-single-nq-base
eval_model_name_or_path=None
ctx_retrieve_key=question_skeleton
task=question_generation

xpython=$home/env/v100/bin/python
file=$home/rag/run_rag.py
$xpython    $file \
--generator_name_or_path=${generator_name_or_path} \
--reward_model_name_or_path=${reward_model_name_or_path} \
--vanilla_qg_model_name_or_path=${vanilla_qg_model_name_or_path} \
--dataset_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_${ctx_retrieve_key} \
--index_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_hnsw_index_${ctx_retrieve_key}.faiss \
--eval_model_name_or_path=${eval_model_name_or_path} \
--data_dir=$home/data/${dataset}/processed \
--output_dir=$home/output/${dataset}/RAG/experiments_v4/ \
--max_source_length=256 \
--max_combined_length=256 \
--train_dpr=True \
--set_context_encoder_for_training=False \
--do_train=True \
--do_eval=False \
--num_train_epochs=7 \
--retrieve_key=question_skeleton  \
--ctx_retrieve_key=question_skeleton \
--n_questions=3 \
--eval_n_questions=5 \
--train_retrieve_num=100 \
--eval_retrieve_num=500 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--reward_type=qa  \
--add_rl_base_reward=True \
--sm_reward_type=jaccard \
--add_sm_base_reward=False \
--whiten=False \
--learning_rate=1e-5 \
--dpr_learning_rate=1e-7 \
--lamda=0.5  \
--sm_coef=0.1 \
--gradient_accumulation_steps=1 \
--reward_batch_normalization=True \
--eval_steps=2000 \
--train_filter_type=v3 \
--eval_filter_type=v0_1 \
--task=question_generation \
--n_train=-1 \
--n_val=-1 \
--do_sample=True \
--use_bm25=False \
--question_encoder_name_or_path=${question_encoder_name_or_path} \
--seed=42
```


newsqa
```
export CUDA_VISIBLE_DEVICES=0
home=${home_path}
external_dataset=newsqa
dataset=newsqa
generator_name_or_path=$home/output/newsqa/QG/experiments_with_null_skeleton/44.43476673263473 \
reward_model_name_or_path=$home/output/newsqa/QA/experiments/t5-base-GPUNums1-len1250-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b8-beam_search5/0.5208000588588081 \
vanilla_qg_model_name_or_path=$home/output/newsqa/QG/experiments/t5-base-GPUNums1-len1250-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b8-beam_search5/14.993048431084343 \
question_encoder_name_or_path=facebook/dpr-question_encoder-single-nq-base
eval_model_name_or_path=None
ctx_retrieve_key=question_skeleton
task=question_generation

xpython=$home/env/v100/bin/python
file=$home/rag/run_rag.py
$xpython    $file \
--generator_name_or_path=${generator_name_or_path} \
--reward_model_name_or_path=${reward_model_name_or_path} \
--vanilla_qg_model_name_or_path=${vanilla_qg_model_name_or_path} \
--dataset_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_${ctx_retrieve_key} \
--index_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_hnsw_index_${ctx_retrieve_key}.faiss \
--eval_model_name_or_path=${eval_model_name_or_path} \
--data_dir=$home/data/${dataset}/processed \
--output_dir=$home/output/${dataset}/RAG/experiments_v3/ \
--max_source_length=256 \
--max_combined_length=1250 \
--train_dpr=True \
--set_context_encoder_for_training=False \
--do_train=True \
--do_eval=False \
--num_train_epochs=7 \
--retrieve_key=question_skeleton  \
--ctx_retrieve_key=question_skeleton \
--n_questions=2 \
--eval_n_questions=5 \
--train_retrieve_num=100 \
--eval_retrieve_num=500 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--reward_type=qa  \
--add_rl_base_reward=True \
--sm_reward_type=jaccard \
--add_sm_base_reward=False \
--whiten=False \
--learning_rate=1e-6 \
--dpr_learning_rate=1e-7 \
--lamda=0.5  \
--sm_coef=0.05 \
--gradient_accumulation_steps=1 \
--reward_batch_normalization=True \
--eval_steps=2000 \
--train_filter_type=v3 \
--eval_filter_type=v0_1 \
--task=question_generation \
--n_train=-1 \
--n_val=-1 \
--do_sample=True \
--use_bm25=False \
--question_encoder_name_or_path=${question_encoder_name_or_path} \
--seed=42
```


获得squad1的training data对应的predicted question，再作为DA增强到QA model 训练中去
```
export CUDA_VISIBLE_DEVICES=1
home=${home_path}
external_dataset=SQuAD_1.1_split1
dataset=SQuAD_1.1_split1
generator_name_or_path=$home/output/SQuAD_1.1_split1/QG/experiments_with_null_skeleton/51.801828729175455 \
reward_model_name_or_path=$home/output/SQuAD_1.1_split1/QA/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b64-beam_search5/0.7866302071131831 \
vanilla_qg_model_name_or_path=$home/output/SQuAD_1.1_split1/QG/experiments/t5-base-GPUNums1-len128-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b64-top_k50-top_p0.9-temperature1.0/20.827964423502124 \
question_encoder_name_or_path=facebook/dpr-question_encoder-single-nq-base
eval_model_name_or_path=${home_path}/RAST/output/SQuAD_1.1_split1/RAG/experiments_v17/17.30754486328647
ctx_retrieve_key=question_skeleton
task=question_generation

xpython=$home/env/v100/bin/python
file=$home/rag/run_rag.py
$xpython    $file \
--generator_name_or_path=${generator_name_or_path} \
--reward_model_name_or_path=${reward_model_name_or_path} \
--vanilla_qg_model_name_or_path=${vanilla_qg_model_name_or_path} \
--dataset_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_${ctx_retrieve_key} \
--index_path=$home/data/${external_dataset}/processed/my_knowledge_dataset_hnsw_index_${ctx_retrieve_key}.faiss \
--eval_model_name_or_path=${eval_model_name_or_path} \
--data_dir=$home/data/${dataset}/processed \
--output_dir=$home/output/${dataset}/RAG/experiments_train_output/ \
--max_source_length=128 \
--max_combined_length=128 \
--train_dpr=True \
--set_context_encoder_for_training=False \
--do_train=False \
--do_eval=True \
--num_train_epochs=7 \
--retrieve_key=${ctx_retrieve_key} \
--ctx_retrieve_key=${ctx_retrieve_key} \
--n_questions=3 \
--eval_n_questions=5 \
--train_retrieve_num=100 \
--eval_retrieve_num=500 \
--per_device_train_batch_size=12 \
--per_device_eval_batch_size=10 \
--reward_type=qa  \
--add_rl_base_reward=True \
--sm_reward_type=jaccard \
--add_sm_base_reward=False \
--whiten=False \
--learning_rate=1e-5 \
--dpr_learning_rate=1e-6 \
--lamda=0.5  \
--sm_coef=0.1 \
--gradient_accumulation_steps=1 \
--reward_batch_normalization=True \
--eval_steps=2000 \
--train_filter_type=v3 \
--eval_filter_type=v0_1 \
--task=question_generation \
--n_train=-1 \
--n_val=-1 \
--do_sample=True \
--use_bm25=False \
--question_encoder_name_or_path=${question_encoder_name_or_path} \
--seed=42
```