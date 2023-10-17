import datasets
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
import json
import torch
from tqdm import tqdm
from datasets import Features, Sequence, Value, load_dataset,load_from_disk,concatenate_datasets
import numpy as np
import faiss
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoderTokenizerFast,
    DPRQuestionEncoder,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
    AutoModel,
    AutoTokenizer
)

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Args:
    dpr_question_encoder_model_name: str = field(default='facebook/dpr-question_encoder-multiset-base')
    batch_size: int = field(default=64)
    output_dir: str = field(default='/mnt/data/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split1/processed')
    hnsw_d:int = field(default=768)
    hnsw_m:int = field(default=128)
    
args = Args()


def highlight_context(context,answer):
    res = []
    for c,a in zip(context,answer):
        answer_start = c.find(a)
        assert answer_start != -1
        HL_context = c[:answer_start-1] + ' <HL> ' + a + \
                ' <HL> ' + c[answer_start + len(a):]
        res.append(HL_context)
    return res

def mask_context(context,answer):
    res = []
    for c,a in zip(context,answer):
        answer_start = c.find(a)
        assert answer_start != -1
        masked_context = c[:answer_start-1] + ' [mask] ' + c[answer_start + len(a):]
        res.append(masked_context)
    return res
def embed(qrys: dict, question_encoder, question_tokenizer,suffix) -> dict:
    """Compute the DPR embeddings of document passages"""
    if qrys['answer']:
        HL_context = qrys['context']
        masked_context = qrys['context']
    else:
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


def embed_for_paraphrase(qrys: dict, question_encoder, question_tokenizer,suffix) -> dict:
    """Compute the DPR embeddings of document passages"""
    # question_skeleton, original_context_skeleton
    if suffix == 'masked_context':
        input_ids = question_tokenizer(
            qrys['original_context_skeleton'], truncation=True, padding="longest", return_tensors="pt"
        )["input_ids"]
    else:
        input_ids = question_tokenizer(
            qrys[suffix], truncation=True, padding="longest", return_tensors="pt"
        )["input_ids"]  
    embeddings = torch.nn.functional.normalize(question_encoder(input_ids.to(device=device), return_dict=True).pooler_output,p=2,dim=1)
    return {"embeddings": embeddings.detach().cpu().numpy(),'context': qrys['src'],'masked_context':qrys['original_context_skeleton'],"question":qrys['tgt'],"title":qrys['src'],"text":qrys['src']}

def save(suffix,dataset_path= "/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split2/processed_v2/index_dataset.json",question_encoder = None,use_dlc=False):
    if suffix == 'masked_context':
        args.dpr_question_encoder_model_name='bert-base-cased'
    # print(args.dpr_question_encoder_model_name)
    question_tokenizer = AutoTokenizer.from_pretrained(args.dpr_question_encoder_model_name) # DPRQuestionEncoderTokenizerFast
    if question_encoder is None:
        question_encoder = AutoModel.from_pretrained(args.dpr_question_encoder_model_name).to(device=device) # DPRQuestionEncoder
    if use_dlc : # 说明是训练过程中保存index,保存在processed_ctx中
        dataset_path = dataset_path.replace('/mnt/ruzhen/ruzhen/', '/root/data/ruzhen/')
    dataset = load_dataset("json", data_files=dataset_path,field='data',split='train')
    if 'SQuAD_1.1_split1' in dataset_path or 'SQuAD_1.1_split2' in dataset_path or 'newsqa' in dataset_path or 'all_datasets' in dataset_path:
        if 'newsqa' in dataset_path:
            dataset = dataset.remove_columns(["answer_start", "answer_end"])
        new_features = Features(
            {"context": Value("string"),"text": Value("string"), "answer": Value("string"),"question": Value("string"),"question_skeleton":Value("string"), "embeddings": Sequence(Value("float32")),
            "title":Value("string"),"masked_context":Value("string"),"HL_context":Value("string")}
        )  # optional, save as float32 instead of float64 to save space

        dataset = dataset.map(
            partial(embed, question_encoder=question_encoder, question_tokenizer=question_tokenizer,suffix=suffix),
            batched=True,
            batch_size=args.batch_size,
            features=new_features,
        )
    elif 'data_qqp' in dataset_path or 'data_mscoco' in dataset_path:
        new_features = Features(
            {"context": Value("string"),"question_skeleton":Value("string"), "embeddings": Sequence(Value("float32")),
            "masked_context":Value("string"),'title':Value("string"),"text":Value("string"),"question":Value("string")}
        )  # optional, save as float32 instead of float64 to save space
        dataset = dataset.map(
            partial(embed_for_paraphrase, question_encoder=question_encoder, question_tokenizer=question_tokenizer,suffix=suffix),
            batched=True,
            batch_size=args.batch_size,
            remove_columns=['ref','tgt','src','original_context_skeleton'],
            features=new_features,
        )
    args.output_dir = '/' + '/'.join(dataset_path.split('/')[1:-1])
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
    # dataset.load_faiss_index("embeddings", index_path)  # to reload the index

def caculate_jaccard(s1,s2):
    words_1 = set(s1.lower().split())
    words_2 = set(s2.lower().split())
    intersection = words_1.intersection(words_2)
    union = words_1.union(words_2)
    return float(len(intersection) / len(union))
def filter(threshold=0.6,dataset_path="/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split2/processed_v2/train.jsonl"): # 先去除相同的skeleton，再用Jaccard相似度过滤掉特别相似的skeleton
    unique_data = []
    unique_skeleton = set()
    with open(dataset_path,'r') as fp:
        for line in fp.readlines():
            data = json.loads(line)
            if not data['question_skeleton'].endswith('?'):
                continue
            if data['question_skeleton'].lower() not in unique_skeleton:
                unique_skeleton.add(data['question_skeleton'].lower())
                unique_data.append(data)
    # 
    # similarity = np.ones((len(data))))
    new_data = []
    deleted_idx = set()
    for i in tqdm(range(len(unique_data))):
        if i in deleted_idx:
            continue
        new_data.append(unique_data[i])
        for j in range(i+1,len(unique_data)):
            if j in deleted_idx:
                continue
            sim = caculate_jaccard(unique_data[i]['question_skeleton'],unique_data[j]['question_skeleton'])
            if sim > threshold:
                deleted_idx.add(j)
            
    print(len(new_data))
    save_data_path = dataset_path.replace('train.jsonl','index_dataset.json')
    res = {}
    res['version'] = 0.1
    res['data'] = new_data
    with open(save_data_path,'w')as fp:
        json.dump(res,fp,indent=2)
        # for d in unique_data:
        #     json.dump(d, fp)
        #     fp.write('\n')
    

if __name__ == "__main__":
    suffix = 'question_skeleton' # 'question','question_skeleton',masked_context, for paraphrase_generation:question_skeleton, original_context_skeleton
    home = '/home/student2021/gouqi/RAST/'
    dataset = home + "data/newsqa/processed/train.jsonl"
    filter(dataset_path=dataset) # 过滤掉相似的skeleton，得到index_datasets.jsonl
    #
    dataset = dataset.replace('train.jsonl','index_dataset.json')
    save(suffix,dataset_path = dataset)