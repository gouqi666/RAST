#!/bin/bash -e
if [ $use_all_corpus == True ] 
then
    external_dataset=all_datasets
else
    external_dataset=$dataset
fi
if [ $dataset == SQuAD_1.1_split1 ] 
then
    max_combined_length=128
    pre_model_batch_size=32
    batch_size=16
    eval_batch_size=16
    n_questions=3
    learning_rate=5e-6
    dpr_learning_rate=1e-7
    forward_batch_size=${batch_size}
    generator_name_or_path=RQG/output/SQuAD_1.1_split1/QG/experiments_with_null_skeleton/t5-base-GPUNums1-len128-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b32-top_k30-top_p0.9-temperature1.0 \
    reward_model_name_or_path=/home/student2020/gouqi/RAST/output/SQuAD_1.1_split1/QA/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b64-beam_search5/ \
    vanilla_qg_model_name_or_path=RQG/output/SQuAD_1.1_split1/QG/experiments/t5-base-GPUNums1-len128-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b32-beam_search5 \
    gradient_accumulation_steps=1
    task=question_generation
    eval_n_questions=5 # 5
    output_dir=RQG/output/${dataset}/${output_file}/
elif [ $dataset == SQuAD_1.1_split2 ] 
then
    max_combined_length=512
    pre_model_batch_size=32
    batch_size=8
    eval_batch_size=8
    n_questions=3
    learning_rate=1e-6
    dpr_learning_rate=1e-7
    forward_batch_size=${batch_size}
    generator_name_or_path=RQG/output/SQuAD_1.1_split2/QG/experiments_with_null_skeleton/t5-base-GPUNums1-len512-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b16-top_k30-top_p0.9-temperature1.0 \
    reward_model_name_or_path=RQG/output/SQuAD_1.1_split2/QA/experiments/t5-base-GPUNums1-len512-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b16-beam_search5 \
    vanilla_qg_model_name_or_path=RQG/output/SQuAD_1.1_split2/QG/experiments/t5-base-GPUNums1-len512-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b16-beam_search5 \
    gradient_accumulation_steps=1
    task=question_generation
    eval_n_questions=5
    output_dir=RQG/output/${dataset}/${output_file}/
elif [ $dataset == newsqa ] 
then
    max_combined_length=1250
    pre_model_batch_size=16
    batch_size=4
    eval_batch_size=2
    n_questions=2
    learning_rate=1e-6
    dpr_learning_rate=1e-7
    forward_batch_size=8
    generator_name_or_path=RQG/output/newsqa/QG/experiments_with_null_skeleton/t5-base-GPUNums1-len1250-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b16-top_k30-top_p0.9-temperature1.0 \
    reward_model_name_or_path=RQG/output/newsqa/QA/experiments/t5-base-GPUNums1-len1250-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b16-beam_search5 \
    vanilla_qg_model_name_or_path=RQG/output/newsqa/QG/experiments/t5-base-GPUNums1-len1250-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr5e-05-b16-beam_search5 \
    task=question_generation
    gradient_accumulation_steps=1
    eval_n_questions=5
    output_dir=RQG/output/${dataset}/${output_file}/
fi
xpython=python
file=/mnt/ruzhen/ruzhen/RQG/rag/run_rag.py
if [ $use_dlc == True ] 
then 
    xpython=/root/data/ruzhen/envs/a100/bin/python
    file=/root/data/ruzhen/RQG/rag/run_rag.py
fi

$xpython    $file \
--use_dlc=$use_dlc \
--generator_name_or_path=${generator_name_or_path} \
--reward_model_name_or_path=${reward_model_name_or_path} \
--vanilla_qg_model_name_or_path=${vanilla_qg_model_name_or_path} \
--dataset_path=RQG/data/${external_dataset}/${flag}/my_knowledge_dataset_${ctx_retrieve_key} \
--index_path=RQG/data/${external_dataset}/${flag}/my_knowledge_dataset_hnsw_index_${ctx_retrieve_key}.faiss \
--data_dir=RQG/data/${dataset}/${flag} \
--output_dir=${output_dir} \
--max_source_length=128 \
--max_combined_length=${max_combined_length} \
--train_dpr=${train_dpr} \
--set_context_encoder_for_training=False \
--do_train=False \
--do_eval=True \
--num_train_epochs=${epoch} \
--retrieve_key=${retrieve_key}  \
--ctx_retrieve_key=${ctx_retrieve_key} \
--n_questions=${n_questions} \
--eval_n_questions=${eval_n_questions} \
--train_retrieve_num=${train_retrieve_num} \
--eval_retrieve_num=${eval_retrieve_num} \
--per_device_train_batch_size=${batch_size} \
--per_device_eval_batch_size=${eval_batch_size} \
--reward_type=qa  \
--add_rl_base_reward=True \
--sm_reward_type=jaccard \
--add_sm_base_reward=${add_sm_base_reward} \
--whiten=False \
--learning_rate=${learning_rate} \
--dpr_learning_rate=${dpr_learning_rate} \
--lamda=${lamda} \
--sm_coef=${sm_coef} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--reward_batch_normalization=True \
--eval_steps=2000 \
--train_filter_type=${train_filter_type} \
--eval_filter_type=${eval_filter_type} \
--task=${task} \
--n_train=0.3 \
--n_val=-1 \
--do_sample=${do_sample} \
--use_bm25=${use_bm25} \
--question_encoder_name_or_path=${question_encoder_name_or_path} \
--seed=$seed