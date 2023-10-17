# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
# print(sys.path)
from collections import defaultdict
import numpy as np
# from gensim.summarization.bm25 import BM25
import argparse
from functools import partial
import configparser
import itertools
from prepare_dataset import save as embeds_update

import json
import copy
from tqdm import tqdm
import logging
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, T5Tokenizer, Trainer, \
    BartTokenizer, T5ForConditionalGeneration, BartConfig, T5Config, DPRConfig, RagConfig, AutoModel, \
    BertForSequenceClassification, set_seed
from rag_arguments import ModelArguments, DataTrainingArguments, TrainingArguments, ReinforcementArguments
from datasets import load_dataset, load_from_disk, Features, Sequence, Value, load_dataset
import torch.distributed as dist
from ragRetriever import MyRagRetriever
from ragModel import MyRagModel
from ragSequenceForGeneration import MyRagSequenceForGeneration
from RAGTrainer import RAGTrainer
from utils import calculate_bleu, get_question_skeleton_for_qg
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def settle_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def cleanup():
    dist.destroy_process_group()


def highlight_context(example):
    contexts, answers = example['context'], example['answer']
    res = {"HL_context": []}
    for context, answer in zip(contexts, answers):
        if not answer:
            res['HL_context'].append(context)
        else:
            answer_start = context.find(answer)
            assert answer_start != -1
            HL_context = context[:answer_start - 1] + ' <HL> ' + answer + \
                         ' <HL> ' + context[answer_start + len(answer):]
            res['HL_context'].append(HL_context)
    return res


def mask_context(context, answer):
    res = []
    for c, a in zip(context, answer):
        answer_start = c.find(a)
        assert answer_start != -1
        masked_context = c[:answer_start - 1] + ' [MASK] ' + c[answer_start + len(a):]
        res.append(masked_context)
    return res


class Collator:
    def __init__(self, tokenizer, padding, max_length):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        features = {}
        features['input_ids'] = torch.tensor([item['input_ids'] for item in batch])
        features['attention_mask'] = torch.tensor([item['attention_mask'] for item in batch])

        # features = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     return_tensors='pt',
        # )
        contexts = [item['HL_context'] for item in batch]
        labels = [item['question'] for item in batch]
        return {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask'], 'contexts': contexts,
                'labels': labels}


class T5QARewardModel():
    def __init__(self, device, reward_model_name_or_path, max_source_length):
        self.device = device
        self.max_source_length = max_source_length
        self.config = AutoConfig.from_pretrained(
            reward_model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_name_or_path,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            reward_model_name_or_path,
        )

    def get_rewards(self, hl_context, prediction):
        # split hl_context to context and answer
        context = []
        answer = []
        for hl_c in hl_context:
            hl_c_list = hl_c.split(' ')
            start, end = -1, -1
            for i in range(len(hl_c_list)):
                if start != -1 and hl_c_list[i] == '<HL>':
                    end = i
                    break
                if hl_c_list[i] == '<HL>':
                    start = i
            else:
                end = len(hl_c_list)
            answer.append(' '.join(hl_c_list[start + 1:end]))
            context.append(' '.join(hl_c_list[:start] + hl_c_list[start + 1:end] + hl_c_list[end + 1:]))

        # alter input format for qa
        inputs = []
        for con, pred in zip(context, prediction):
            inputs.append(con + ' <sep> ' + pred)

        input_tok = self.tokenizer.batch_encode_plus(
            list(inputs),
            max_length=self.max_source_length,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        output_tok = self.tokenizer.batch_encode_plus(
            list(answer),
            max_length=64,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        output_tok['input_ids'] = output_tok['input_ids'].masked_fill(output_tok['input_ids'] == 0,
                                                                      -100)  # label 0 -> -100
        self.model.eval()
        self.model = self.model.to(self.device)
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=output_tok['input_ids'])
        rewards = []

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.model.eval()
        output = self.model(
            input_ids=input_tok['input_ids'].to(self.device),
            attention_mask=input_tok['attention_mask'].to(self.device),
            labels=output_tok['input_ids'].to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device)
        )
        for logit, label in zip(output.logits, output_tok['input_ids'].to(self.device)):
            loss = loss_fct(logit, label)
            logp = -loss
            rewards.append(np.exp(logp.data.item()))
            # rewards.append(logp.data.item())
        return np.array(rewards)


class MatchRewardModel():
    def __init__(self, reward_model_name_or_path, max_source_length):
        self.config = AutoConfig.from_pretrained(
            reward_model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_name_or_path,
        )
        self.model = BertForSequenceClassification.from_pretrained(
            reward_model_name_or_path,
        )
        self.max_source_length = max_source_length

    def get_rewards(self, text1, text2):
        inputs = []
        for con, pred in zip(text1, text2):
            inputs.append(con + ' <sep> ' + pred)

        input_tok = self.tokenizer.batch_encode_plus(
            list(inputs),
            max_length=self.max_source_length,
            return_tensors='pt',
            padding='longest',
            truncation=True,
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        rewards = []
        self.model.eval()
        output = self.model(
            input_ids=input_tok['input_ids'].to(self.device),
            attention_mask=input_tok['attention_mask'].to(self.device),
        )
        logits = torch.nn.functional.softmax(output['logits'], dim=1)[:, 1]
        rewards = logits.cpu().detach().numpy()
        return rewards


class SMRewardModel2():
    def __init__(self, reward_model_name_or_path):
        self.reward_model_name_or_path = reward_model_name_or_path

    def get_rewards(self, question, skeleton):
        rewards = []
        # gen_skeleton = get_question_skeleton_for_qg(question,contexts)
        gen_skeleton = question
        for q, s in zip(gen_skeleton, skeleton):
            if not s:
                rewards.append(0)
                continue
            s = s.replace('[MASK]', '')
            words_1 = set(q.lower().split())
            words_2 = set(s.lower().split())
            intersection = words_1.intersection(words_2)
            union = words_1.union(words_2)
            if len(union) == 0:
                rewards.append(0)
                print(f'gen_skeleton:{gen_skeleton},retrieve_skeleton:{skeleton}')
            else:
                rewards.append(float(len(intersection) / len(union)))
        return np.array(rewards)


'''
def embeds_update(suffix,ctx_encoder):
    if suffix == 'masked_context':
        question_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    else:
        question_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(args.dpr_question_encoder_model_name)
    
    dataset = load_dataset("/mnt/workspace/gouqi/code/RQG/data/SQuAD_1.1_split1/processed", split="train")

    new_features = Features(
        {"context": Value("string"),"text": Value("string"), "answer": Value("string"),"question": Value("string"),"question_skeleton":Value("string"), "embeddings": Sequence(Value("float32")),
        "title":Value("string"),"HL_context":Value("string"),"masked_context":Value("string")}
    )  # optional, save as float32 instead of float64 to save space

    def embed(qrys: dict, question_encoder, question_tokenizer,suffix) -> dict:
        """Compute the DPR embeddings of document passages"""
        HL_context = highlight_context(qrys['context'], qrys['answer'])
        masked_context = mask_context(qrys['context'], qrys['answer'])
        if suffix == 'masked_context':
            input_ids = question_tokenizer(
                masked_context, truncation=True, padding="longest", return_tensors="pt"
            )["input_ids"]
        else:
            input_ids = question_tokenizer(
                qrys[suffix], truncation=True, padding="longest", return_tensors="pt"
            )["input_ids"]  
        embeddings = torch.nn.functional.normalize(question_encoder(input_ids.to(device=device), return_dict=True).pooler_output,p=2,dim=1)
        return {"embeddings": embeddings.detach().cpu().numpy(),'title':qrys['answer'],'text':qrys['context'],'HL_context':HL_context,'masked_context':masked_context}
    dataset = dataset.map(
        partial(embed, question_encoder=ctx_encoder, question_tokenizer=question_tokenizer,suffix=suffix),
        batched=True,
        batch_size=64,
        features=new_features,
    )
    
    passages_path = os.path.join(args.output_dir, "my_knowledge_dataset" + f"_{suffix}")
    dataset.save_to_disk(passages_path)
    # from datasets import load_from_disk
    # dataset = load_from_disk(passages_path)  # to reload the dataset



    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(args.hnsw_d, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(args.output_dir, "my_knowledge_dataset_hnsw_index" + f"_{suffix}.faiss")
    dataset.get_index("embeddings").save(index_path)
'''


def main():
    # assert torch.cuda.is_available(), 'CUDA not available'
    # parse arguments

    # parse remaining arguments and divide them into three categories

    parser = argparse.ArgumentParser()
    _, cmd_args = parser.parse_known_args()
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ReinforcementArguments))
    model_args, data_args, training_args, rl_args = second_parser.parse_args_into_dataclasses(cmd_args)
    device = training_args.device
    set_seed(training_args.seed)

    # construct name for the output directory
    # for example: conll04-t5-base-ep200-len256-ratio0-b4-train


    # output_dir = training_args.output_dir
    # if training_args.train_dpr and training_args.do_train:
    #     output_dir += '_end2end'  # 联合训练
    # elif training_args.do_train == True:
    #     output_dir += '_rl'  # 只训练generator
    # else:
    #     output_dir += '_base'
    # output_dir += f'_retrieve_key{training_args.retrieve_key}_ctx_retrieve_key{training_args.ctx_retrieve_key}'
    # output_dir += f'train_filter{training_args.train_filter_type}_num_{training_args.train_retrieve_num}_eval_filter{training_args.eval_filter_type}_num{training_args.eval_retrieve_num}_{training_args.n_questions}'
    # output_dir = os.path.join(
    #     output_dir,
    #     f'{model_args.model_name_or_path.split("/")[-1]}'
    #     f'-len{data_args.max_source_length}'
    #     f'--warm{training_args.warmup_ratio}'
    #     f'--warmSteps{training_args.warmup_steps}'
    #     f'--weightDecay{training_args.weight_decay}'
    #     f'-lr{training_args.learning_rate}'
    #     f'-b{training_args.per_device_train_batch_size}'
    #     f'-gamma{rl_args.gamma}'
    #     f'-lamda{rl_args.lamda}'
    #     f'-whiten{rl_args.whiten}'
    #     f'-normalize_reward{training_args.reward_batch_normalization}'
    #     f'reward_type{training_args.reward_type}'
    #     f'sm_reward_type{training_args.sm_reward_type}_{rl_args.sm_coef}'
    #     f'rl_base{training_args.add_rl_base_reward}_sm_base{training_args.add_sm_base_reward}'
    #     f'do_sample{data_args.do_sample}'
    # )
    #
    # training_args.output_dir = output_dir


    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    # setup logging
    filename = training_args.output_dir + '/' + __name__ + '.log'
    logging.basicConfig(
        filename=filename,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
        filemode='w+'
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.warning(f'device:{device}')
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)
    logger.info(rl_args)
    # dataset and index 
    # dataset = load_from_disk(data_args.dataset_path)
    # dataset.load_faiss_index("embeddings", data_args.index_path)
    train_dataset = load_dataset(data_args.data_dir, split="train")
    eval_dataset = load_dataset(data_args.data_dir, split="validation")
    test_dataset = load_dataset(data_args.data_dir, split="test")
    if os.path.exists(model_args.eval_model_name_or_path) :  # 加载指定模型进行
        print('加载eval的model')
        eval_model_path = model_args.eval_model_name_or_path
        rag_config = RagConfig.from_pretrained(eval_model_path)
        # 重新设置index和datasets,适配dsw和dlc
        rag_config.passages_path = data_args.dataset_path
        rag_config.index_path = data_args.index_path
        question_encoder_tokenizer = AutoTokenizer.from_pretrained(f'{eval_model_path}/question_encoder_tokenizer')
        generator_tokenizer = AutoTokenizer.from_pretrained(f'{eval_model_path}/generator_tokenizer')
        retriever = MyRagRetriever(config=rag_config,
                                   question_encoder_tokenizer=question_encoder_tokenizer,
                                   generator_tokenizer=generator_tokenizer,
                                   )
        model = MyRagSequenceForGeneration.from_pretrained(eval_model_path, retriever=retriever)
    else:  # create model config
        question_encoder_config = AutoConfig.from_pretrained(model_args.question_encoder_name_or_path)

        generator_config = AutoConfig.from_pretrained(model_args.generator_name_or_path)
        rag_config = RagConfig.from_pretrained(model_args.model_name_or_path)

        rag_config.index_name = 'custom'
        rag_config.passages_path = data_args.dataset_path
        rag_config.index_path = data_args.index_path
        rag_config.use_dummy_dataset = False

        rag_config.generator = generator_config
        rag_config.question_encoder = question_encoder_config
        rag_config.n_docs = training_args.n_questions
        rag_config.max_combined_length = data_args.max_combined_length

        # tokenizer
        question_encoder_tokenizer = AutoTokenizer.from_pretrained(model_args.question_encoder_name_or_path)
        question_encoder_tokenizer.add_tokens(['<HL>', '[MASK]', '<sep>'], special_tokens=True)

        generator_tokenizer = AutoTokenizer.from_pretrained(model_args.generator_name_or_path)
        # generator_tokenizer.add_tokens(['<HL>','[MASK]','<sep>'], special_tokens=True) 

        ctx_encoder_tokenizer = AutoTokenizer.from_pretrained(model_args.question_encoder_name_or_path)
        ctx_encoder_tokenizer.add_tokens(['<HL>', '[MASK]', '<sep>'], special_tokens=True)

        # model
        question_encoder = AutoModel.from_pretrained(model_args.question_encoder_name_or_path,
                                                     config=question_encoder_config)
        question_encoder.resize_token_embeddings(len(question_encoder_tokenizer))

        generator = AutoModelForSeq2SeqLM.from_pretrained(model_args.generator_name_or_path, config=generator_config)
        generator.resize_token_embeddings(len(generator_tokenizer))
        retriever = MyRagRetriever(config=rag_config,
                                   question_encoder_tokenizer=question_encoder_tokenizer,
                                   generator_tokenizer=generator_tokenizer,
                                   )
        retriever.set_ctx_encoder_tokenizer(ctx_encoder_tokenizer)
        model = MyRagSequenceForGeneration(config=rag_config, question_encoder=question_encoder, generator=generator,
                                           retriever=retriever)
    if training_args.do_train and training_args.set_context_encoder_for_training:
        ctx_encoder = copy.deepcopy(question_encoder)
        model.set_context_encoder_for_training(ctx_encoder)
        model.rag.retrieve_key = training_args.retrieve_key
    # dlc save
    # question_encoder = AutoModel.from_pretrained('bert-base-cased')
    # question_encoder_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    # question_encoder_config = AutoConfig.from_pretrained('bert-base-cased')
    # question_encoder.save_pretrained(f'/mnt/data/ruzhen/model/bert-base-cased')
    # question_encoder_tokenizer.save_pretrained(f'/mnt/data/ruzhen/model/bert-base-cased')
    # question_encoder_config.save_pretrained(f'/mnt/data/ruzhen/model/bert-base-cased')

    # rag_config.save_pretrained(f'/mnt/data/ruzhen/model/{model_args.model_name_or_path}')

    # (note that the episode index is used as the random seed, so that each episode is reproducible)
    evaluation_results = defaultdict(list)

    training_args.logging_dir = training_args.output_dir + '/' + 'tensorboard/'  # trainer's tensorboard file location
    try:
        os.makedirs(training_args.logging_dir)
    except FileExistsError:
        pass

    tensorboard_dir = os.path.join(training_args.output_dir, 'tensorboard')

    # tensorboard_writer = SummaryWriter(tensorboard_dir)
    try:
        os.makedirs(tensorboard_dir)
    except FileExistsError:
        pass

    # load pretrained model
    model_ref = None
    model_ref = copy.deepcopy(model).to(device)

    # process data
    if data_args.n_train != -1:
        train_dataset = train_dataset.train_test_split(test_size=data_args.n_train)['test']
    if data_args.n_val != -1:
        eval_dataset = eval_dataset.train_test_split(test_size=data_args.n_val)['test']
        test_dataset = test_dataset.train_test_split(test_size=data_args.n_val)['test']
    collator = Collator(model.retriever.question_encoder_tokenizer, padding='max_length',
                        max_length=data_args.max_source_length)
    if data_args.task == 'question_generation':
        new_features = Features(
            {"HL_context": Value("string"), 'input_ids': Sequence(Value("int32")),
             'attention_mask': Sequence(Value("int32")), "context": Value("string"), "answer": Value("string"),
             "question": Value("string"), "question_skeleton": Value("string"),
             "token_type_ids": Sequence(Value("int32")),
             })  # optional, save as float32 instead of float64 to save space
        if training_args.retrieve_key == 'masked_context':
            train_dataset = train_dataset.map(
                lambda examples: question_encoder_tokenizer(mask_context(examples['context'], examples['answer']),
                                                            padding='max_length',
                                                            max_length=data_args.max_source_length, truncation=True),
                batched=True)
            eval_dataset = eval_dataset.map(
                lambda examples: question_encoder_tokenizer(mask_context(examples['context'], examples['answer']),
                                                            padding='max_length',
                                                            max_length=data_args.max_source_length, truncation=True),
                batched=True)
            test_dataset = test_dataset.map(
                lambda examples: question_encoder_tokenizer(mask_context(examples['context'], examples['answer']),
                                                            padding='max_length',
                                                            max_length=data_args.max_source_length, truncation=True),
                batched=True)
        else:
            train_dataset = train_dataset.map(
                lambda examples: question_encoder_tokenizer(examples[training_args.retrieve_key], padding='max_length',
                                                            max_length=data_args.max_source_length, truncation=True),
                batched=True)
            eval_dataset = eval_dataset.map(
                lambda examples: question_encoder_tokenizer(examples[training_args.retrieve_key], padding='max_length',
                                                            max_length=data_args.max_source_length, truncation=True),
                batched=True)
            test_dataset = test_dataset.map(
                lambda examples: question_encoder_tokenizer(examples[training_args.retrieve_key], padding='max_length',
                                                            max_length=data_args.max_source_length, truncation=True),
                batched=True)

        if 'newsqa' in data_args.data_dir:
            train_dataset = train_dataset.remove_columns(["answer_start", "answer_end"])
            eval_dataset = eval_dataset.remove_columns(["answer_start", "answer_end"])
            test_dataset = test_dataset.remove_columns(["answer_start", "answer_end"])
        train_dataset = train_dataset.map(lambda example: highlight_context(example), batched=True,
                                          features=new_features)
        train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size,
                                      collate_fn=collator, shuffle=True)
        eval_dataset = eval_dataset.map(lambda example: highlight_context(example), batched=True, features=new_features)
        test_dataset = test_dataset.map(lambda example: highlight_context(example), batched=True, features=new_features)

    if training_args.use_bm25:
        assert training_args.retrieve_key == 'question_skeleton'
        candidates = []
        for item in retriever.index.dataset:
            candidates.append(item['question_skeleton'].replace('[MASK]', '').strip().lower().split())
        # retriever.bm25 = BM25(candidates)

    rag_trainer = RAGTrainer(
        ref_model=model_ref,
        eval_data_args=data_args,
        rl_args=rl_args,
        model=model,
        ref_model_device=device,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        eval_collate_fn=collator,
        # compute_metrics=compute_metrics,
        tokenizer=generator_tokenizer,
        logger=logger,
    )

    logger.info(rag_trainer.args)
    if training_args.do_eval:
        rag_trainer.evaluate_(n_docs=training_args.eval_n_questions,
                              vanilla_qg_model_name_or_path=model_args.vanilla_qg_model_name_or_path,
                              retrieve_key=training_args.retrieve_key, train_dataloader=train_dataloader)
    # fine-tune the model
    if training_args.do_train:
        reward_model = T5QARewardModel(device=device, reward_model_name_or_path=model_args.reward_model_name_or_path,
                                       max_source_length=data_args.max_combined_length)
        sm_reward_model = SMRewardModel2(reward_model_name_or_path=model_args.sm_reward_model_name_or_path)
        for epoch_id in range(1, int(training_args.num_train_epochs) + 1):
            for step, batch in enumerate(tqdm(train_dataloader, desc=f'rl train...epoch:{epoch_id}')):
                rag_trainer.model.train()

                predictions, retrieve_candidates, question_candidates, retrieved_context, doc_scores, sequence_scores, greedy_search_sequences = model.generate(
                    batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    contexts=batch['contexts'],
                    max_length=data_args.val_max_target_length,
                    num_beams=1,
                    do_sample=True,
                    top_p=0.95,
                    top_k=30,
                    n_docs=training_args.n_questions,
                    add_rl_base_reward=training_args.add_rl_base_reward,
                    retrieve_num=training_args.train_retrieve_num,
                    filter_type=training_args.train_filter_type,
                    use_bm25=training_args.use_bm25,
                    data_args=data_args
                    # num_return_sequences=5,
                )
                responses = [retriever.generator_tokenizer.decode(predictions[k, :],
                                                                  skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)
                             for k in range(len(predictions))]
                # obtain rewards 
                # original reward
                direct_rewards = []
                label_ids = retriever.generator_tokenizer.batch_encode_plus(
                    list(batch['labels']),
                    max_length=data_args.max_target_length,
                    return_tensors='pt',
                    padding='longest',
                    truncation=True,
                )
                contexts = [c for c in batch['contexts'] for i in range(len(doc_scores) // len(batch['input_ids']))]
                labels = [l for l in label_ids['input_ids'] for i in range(len(doc_scores) // len(batch['input_ids']))]
                # bleu reward
                # t5 qa reward
                # 检索reward
                retriever_reward = 0
                if data_args.task == 'question_generation':
                    consistency_rewards = reward_model.get_rewards(prediction=responses, hl_context=contexts)
                elif data_args.task == 'paraphrase_generation':
                    original_input = [x for x in batch['contexts'] for _ in range(training_args.n_questions)]
                    consistency_rewards = reward_model.get_rewards(text1=original_input, text2=responses)
                retriever_reward = consistency_rewards
                if training_args.sm_reward_type != None:
                    sm_rewards = sm_reward_model.get_rewards(responses, retrieve_candidates)
                retriever_reward += rag_trainer.rf_params['sm_coef'] * sm_rewards
                # ground_rewards = reward_model.get_rewards_from(hl_context = contexts, prediction = batch['labels'], labels = labels)
                if training_args.add_rl_base_reward:  # add_rl_base_reward = True
                    base_responses = [retriever.generator_tokenizer.decode(greedy_search_sequences[k, :],
                                                                           skip_special_tokens=True,
                                                                           clean_up_tokenization_spaces=False)
                                      for k in range(len(greedy_search_sequences))]
                    if data_args.task == 'question_generation':
                        consistency_base_rewards = reward_model.get_rewards(prediction=base_responses,
                                                                            hl_context=contexts)
                    elif data_args.task == 'paraphrase_generation':
                        original_input = [x for x in batch['contexts'] for _ in range(training_args.n_questions)]
                        consistency_base_rewards = reward_model.get_rewards(text1=original_input, text2=base_responses)
                    consistency_rewards -= consistency_base_rewards
                    if training_args.sm_reward_type != None and training_args.add_sm_base_reward:
                        sm_base_rewards = sm_reward_model.get_rewards(base_responses, retrieve_candidates)
                        sm_rewards -= sm_base_rewards
                rag_trainer.step(contexts, retrieve_candidates, predictions.to(device), doc_scores, consistency_rewards,
                                 sm_rewards, retriever_reward)
                if epoch_id > 7 and rag_trainer.global_steps != 0 and rag_trainer.global_steps % training_args.eval_steps == 0:
                    dev_res = rag_trainer.evaluate_(n_docs=training_args.eval_n_questions,
                                                    vanilla_qg_model_name_or_path=model_args.vanilla_qg_model_name_or_path,
                                                    retrieve_key=training_args.retrieve_key)
            if training_args.set_context_encoder_for_training:
                embeds_update(training_args.ctx_retrieve_key, model.rag.ctx_encoder, use_dlc=True)

            if epoch_id <= 7:
                dev_res = rag_trainer.evaluate_(n_docs=training_args.eval_n_questions,
                                                vanilla_qg_model_name_or_path=model_args.vanilla_qg_model_name_or_path,
                                                retrieve_key=training_args.retrieve_key)

            # frezze dpr after one epoch
            # rag_trainer.args.train_dpr = False
            # 重新加载index
            # rag_trainer.model.rag.retriever.re_load()

            # rag_trainer.model.rag.retriever.init_retrieval()
            # if step != 0 and step % 500 == 0:
            #     dev_res = rag_trainer.evaluate_(n_docs = rag_config.n_docs,vanilla_qg_model_name_or_path,retrieve_key=training_args.retrieve_key)

            # eval every epoch
            # dev_res = rag_trainer.evaluate_(n_docs = training_args.eval_n_questions,vanilla_qg_model_name_or_path = model_args.vanilla_qg_model_name_or_path,retrieve_key=training_args.retrieve_key)

            # save_model(ppo_trainer.model, episode_output_dir + '/' + str(best_f1) + '/')
            # tokenizer.save_pretrained(episode_output_dir + '/' + str(best_f1) + '/', legacy_format=False)
            # save_model(ppo_trainer.model, episode_output_dir + '/')
            # tokenizer.save_pretrained(episode_output_dir + '/', legacy_format=False)
            # rag_trainer.model.save_pretrained(episode_output_dir + '/' + str(best_f1) + '/')
        rag_trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == "__main__":

    GpuNum = torch.cuda.device_count()
    print("GPU:", GpuNum)
    main()
