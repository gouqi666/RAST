import torch
import json
from torch.utils.data import Dataset, DataLoader
import random
import copy
import numpy as np
    
def read_raw_files(path):
    with open(path,'r') as fp:
        data = json.load(fp)
        data = data['data']
        res = []
        all_context = []
        for item in data:
            para = item['paragraphs']
            for qas in para:
                all_context.append({'pid': len(all_context), 'title': item['title'], 'context': qas['context']})
                for q in qas['qas']:
                    if q['is_impossible']:
                        continue
                    res.append({
                        'context_id': len(all_context) - 1,
                        'question': q['question'],
                        'answers': q['answers'],
                    })
        return res, all_context
def read_jsonls_file(data_path):
    data = []
    with open(data_path,'r') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            data.append(item)
    return data

class SQuAD_QG_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                is_use_skeleton = True,
                seed = 42,
                returns='pt'):
        self.is_use_skeleton = is_use_skeleton
        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data = read_jsonls_file(data_dir)
        self.data = data
        if self.n_obs != -1:
            self.data = data[:int(self.n_obs * len(data))]
        if 'train' in self.data_dir:
            random.shuffle(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        try :
            answer = res['answer']
        except:
            answer = res['ans']
        context = res['context']
        question = res['question']
        if answer is not None:
            answer_start = context.find(answer)
            # if answer_start == -1:
            #     print(answer)
            #     print(context)
            assert answer_start != -1
            HL_context = context[:answer_start-1] + ' <HL> ' + answer + \
                    ' <HL> ' + context[answer_start + len(answer):]
        else:
            HL_context = context
        if self.is_use_skeleton:
            if res['question_skeleton']:
                HL_context  = HL_context + ' <sep> ' + res['question_skeleton']
        if self.returns == 'dict':
            return {'context':HL_context,'question':question,'answer':answer}
        else:
            ctx_tensor = self.tokenizer(HL_context, max_length=self.max_source_length,
                                        truncation=True)
            qry_id = self.tokenizer(question, max_length=self.max_target_length,
                                        truncation=True)
            qry_id = qry_id['input_ids']
            return {"input_ids":ctx_tensor['input_ids'],"attention_mask":ctx_tensor['attention_mask'],'labels':qry_id}
        

class Newsqa_QG_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                is_use_skeleton = True,
                seed = 42,
                returns='pt'):
        self.is_use_skeleton = is_use_skeleton
        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data = read_jsonls_file(data_dir)
        self.data = data
        if self.n_obs != -1:
            self.data = data[:int(self.n_obs * len(data))]
        if self.seed is not None:
            random.shuffle(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        try :
            answer = res['answer']
        except:
            answer = res['ans']
        context = res['context']
        question = res['question']
        if answer is not None:
            answer_start = context.find(answer)
            # if answer_start == -1:
            #     print(answer)
            #     print(context)
            assert answer_start != -1
            HL_context = context[:answer_start-1] + ' <HL> ' + answer + \
                    ' <HL> ' + context[answer_start + len(answer):]
        else:
            HL_context = context
        if self.is_use_skeleton:
            if res['question_skeleton']:
                HL_context  = res['question_skeleton'] + ' <sep> ' + HL_context
        if self.returns == 'dict':
            return {'context':HL_context,'question':question,'answer':answer}
        else:
            ctx_tensor = self.tokenizer(HL_context, max_length=self.max_source_length,
                                        truncation=True)
            qry_id = self.tokenizer(question, max_length=self.max_target_length,
                                        truncation=True)
            qry_id = qry_id['input_ids']
            return {"input_ids":ctx_tensor['input_ids'],"attention_mask":ctx_tensor['attention_mask'],'labels':qry_id}
        
        
class SQuAD_SM_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                seed = 42,
                returns='pt'):

        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data = read_jsonls_file(data_dir)
        self.data = data
        if self.seed is not None:
            random.shuffle(self.data)
        if self.n_obs != -1:
            self.data = self.data[:int(self.n_obs * len(self.data))]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        question = res['question']
        question_skeleton = res['question_skeleton']
        if not res.get('label',None):
            res['label'] = 0
        label = res['label']
        q_s = question + ' <sep> ' + question_skeleton
        if self.returns == 'dict':
            return {'q_s':q_s,'question_skeleton':question_skeleton,'question':question,'label':label}
        else:
            ctx_tensor = self.tokenizer(q_s, max_length=self.max_source_length,
                                          truncation=True, padding='max_length')
            return {"input_ids":ctx_tensor['input_ids'],"attention_mask":ctx_tensor['attention_mask'],'labels':label}


class CQ_Match_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                seed = 42,
                returns='pt'):

        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data = read_jsonls_file(data_dir)
        self.data = data
        if self.seed is not None:
            random.shuffle(self.data)
        if self.n_obs != -1:
            self.data = self.data[:int(self.n_obs * len(self.data))]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        question = res['question']
        context = res['context']
        label = res['label']
        if self.returns == 'dict':
            return {'context':context,'question':question,'label':label}
        else:
            c_q = context + ' <sep> ' + question
            ctx_tensor = self.tokenizer(c_q, max_length=self.max_source_length,
                                          truncation=True, padding='max_length')
            return {"input_ids":ctx_tensor['input_ids'],"attention_mask":ctx_tensor['attention_mask'],'labels':label}

        
class SQuAD_QA_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                seed = 42,
                returns='pt',
                extra_data = False
                ):

        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.seed = seed
        self.extra_data = extra_data
        if seed is not None:
            random.seed(seed)
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data = read_jsonls_file(data_dir)
        self.data = data
        if self.extra_data:
            extra_data_path = '**' #
            with open(extra_data_path,'r') as fp:
                data = json.load(fp)
            for item in data:
                context = item['context'].replace('<HL>','')
                answer = item['context'].split('<HL>')[1]
                # for x in item['predicted']:
                if random.random() < 0.1:
                    self.data.append({'context':context,'answer':answer,'question':item['best_predicted']})
        if self.n_obs != -1:
            self.data = data[:int(self.n_obs * len(data))]
        if self.seed is not None:
            random.shuffle(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        answer = res['answer']
        context = res['context']
        question = res['question']

        enhanced_context = context + ' <sep> ' + question
        if self.returns == 'dict':
            return {'context':context,'question':question,'answer':answer}
        else:
            ctx_tensor = self.tokenizer(enhanced_context, max_length=self.max_source_length,
                                          truncation=True)
            ans = self.tokenizer(answer, max_length=self.max_target_length,
                                          truncation=True)
            return {"input_ids":ctx_tensor['input_ids'],"attention_mask":ctx_tensor['attention_mask'],'labels':ans['input_ids']}
        
class SQuAD_Raw_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                returns='pt'):
        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data,context = read_raw_files(data_dir)
        self.data = data
        if self.n_obs != -1:
            self.data = data[:int(self.n_obs * len(data))]
        self.context = context
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        answer = res['answers'][0]
        context = self.context[res['context_id']]['context']
        HL_context = context[:answer['answer_start']-1] + ' <HL> ' + \
                context[answer['answer_start']: answer['answer_start'] + len(answer['text'])] + \
                ' <HL> ' + context[answer['answer_start'] + len(answer['text']):]

        if self.returns == 'dict':
            return {'context':HL_context,'question':question}
        else:
            ctx_tensor = self.tokenizer(HL_context, max_length=self.max_source_length,
                                          truncation=True)
            qry_id = self.tokenizer(res['question'], max_length=self.max_target_length,
                                          truncation=True)
            qry_id = qry_id['input_ids']
            return{"input_ids":ctx_tensor['input_ids'],"attention_mask":ctx_tensor['attention_mask'],'labels':qry_id}

class SQuAD_DataLoader(DataLoader):

    def __init__(self,dataset,tokenizer, batch_size,max_target_length,max_source_length):
        super().__init__(dataset,batch_size,collate_fn = self.collate_wrapper)
        self.tokenizer = tokenizer

    def collate_wrapper(self,batch):
        ctx,qry = zip(*batch)
        input_tok = self.tokenizer.batch_encode_plus(
            list(ctx),
            max_length=256,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        output_tok = self.tokenizer.batch_encode_plus(
            list(qry),
            max_length=256,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        return input_tok['input_ids'], input_tok['attention_mask'], output_tok['input_ids']

class Paraphrase_Generation_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                is_use_skeleton = True,
                seed = None,
                returns='pt'):
        self.is_use_skeleton = is_use_skeleton
        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.seed = seed
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data = read_jsonls_file(data_dir)
        self.data = data
        if self.n_obs != -1:
            self.data = data[:int(self.n_obs * len(data))]
        if self.seed is not None:
            random.shuffle(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        context = res['src'] 
        question = res['tgt']
        if self.is_use_skeleton:
            context  = context + ' <sep> ' + res['question_skeleton']
        if self.returns == 'dict':
            return {'context':context,'question':question}
        else:
            ctx_tensor = self.tokenizer(context, max_length=self.max_source_length,
                                        truncation=True)
            qry_id = self.tokenizer(question, max_length=self.max_target_length,
                                        truncation=True)
            qry_id = qry_id['input_ids']
            return {"input_ids":ctx_tensor['input_ids'],"attention_mask":ctx_tensor['attention_mask'],'labels':qry_id}


class Paraphrase_generation_Classify_Dataset(Dataset):
    def __init__(self,
                tokenizer, 
                data_dir,
                n_obs,
                max_target_length,
                max_source_length,
                prefix,
                seed = 42,
                returns='pt'):

        self.data_dir = data_dir
        self.data = []
        self.context = []
        self.tokenizer = tokenizer
        self.n_obs = n_obs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.returns = returns
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.load_dataset(data_dir)
    def load_dataset(self,data_dir):
        data = read_jsonls_file(data_dir)
        self.data = data
        if self.seed is not None:
            random.shuffle(self.data)
        if self.n_obs != -1:
            self.data = self.data[:int(self.n_obs * len(self.data))]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        res = self.data[item]
        # highlight answer in context
        q1 = res['q1']
        q2 = res['q2']
        if not res.get('label',None):
            res['label'] = 0
        label = res['label']
        inputs = q1 + ' <sep> ' + q2
        if self.returns == 'dict':
            return {'inputs':inputs,'label':label}
        else:
            input_tensor = self.tokenizer(inputs, max_length=self.max_source_length,
                                          truncation=True, padding='max_length')
            return {"input_ids":input_tensor['input_ids'],"attention_mask":input_tensor['attention_mask'],'labels':label}
