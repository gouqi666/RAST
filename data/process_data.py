import numpy as np
import pandas as pd
from tqdm import tqdm
np.random.seed(42)
import spacy
import json
import copy
import os
from datasets import load_from_disk,concatenate_datasets
from multiprocessing import Pool
import argparse
nlp = spacy.load("en_core_web_sm")
import sys
retain_words = {'what','?','who','how','much','many','where','which','when','why','do','did','does',
                'mean', 'long', 'take', 'get', 'good', 'best', 'find', 'anyone', 'get', 'difference',
                'best', 'better', 'and', 'or', 'people', 'happen', 'happens', 'would', 'affect', 'effect',
                'effects', 'get', 'way', 'make', 'best', 'know', 'think', 'would', 'like', 'anyone','is',
                'was','be','can','must','could','may','maybe','have','has','should','will','shall',
                'might', 'need', 'ought to', 'dare','dared','whether','whatsoever','whose','whom','whither',
                'whence'

}
def read_stopwords(home):
    stopwords = []
    with open(home+ '/data/stopwords.txt', 'r') as fp:
        for line in fp.readlines():
            stopwords.append(line.strip())
    return stopwords



def process_raw_squad(root_dir = '/mnt/workspace/gouqi/code/RQG/data/SQuAD_1.1_split2/'):
    for mode in ['train','dev','test']:
        file_path = root_dir + mode + '.json'
        # read raw data
        with open(file_path, 'r') as fp:
            data = json.load(fp)
        res = []
        for item in data:
            para = item['paragraphs']
            for qas in para:
                context = qas['context']
                for q in qas['qas']:
                    res.append({
                        'context' : context,
                        'question': q['question'],
                        'answer': q['answers'][0]['text'],
                    })
        save_dir = root_dir + 'processed2/' + mode + '.jsonl'
        with open(save_dir, 'w') as fp:
            for item in res:
                json.dump(item, fp)
                fp.write('\n')

def process_squad_split1(root_dir = '/mnt/workspace/gouqi/code/RQG/data/SQuAD_1.1_split1/'):
    for mode in ['train','dev','test']:
        file_path = root_dir + mode + '.jsonl'
        # read raw data
        res = []
        with open(file_path, 'r') as fp:
            for line in fp.readlines():
                item = json.loads(line)
                item['answer'] = item['ans']
                del item['ans']
                res.append(item)
        save_dir = root_dir + 'processed/' + mode + '.jsonl'
        with open(save_dir, 'w') as fp:
            for item in res:
                json.dump(item, fp)
                fp.write('\n')

def process_squad_split2(root_dir = '/mnt/workspace/gouqi/code/RQG/data/SQuAD_1.1_split2/'):
    for mode in ['train','dev','test']:
        src_file_path = root_dir + 'src-' + mode + '.txt'
        tgt_file_path = root_dir + 'tgt-' + mode + '.txt'
        # read raw data
        with open(src_file_path, 'r') as fp:
            src_data = fp.readlines()
        with open(tgt_file_path, 'r') as fp:
            tgt_data = fp.readlines()
        res = []



        for src,tgt in zip(src_data,tgt_data):
            res.append({
                'context' : src.strip(),
                'question': tgt.strip(),
                'answer': None,
            })
        save_dir = root_dir + 'processed/' + mode + '.jsonl'
        with open(save_dir, 'w') as fp:
            for item in res:
                json.dump(item, fp)
                fp.write('\n')

def process_raw_newsqa(root_dir= '/mnt/ruzhen/ruzhen/RQG/data/newsqa/split_data/'):
    for mode in ['train','dev','test']:
        df = pd.read_csv(f'{root_dir}{mode}.csv')
        json_data = []
        for context,question,answer_se in zip(df['story_text'],df['question'],df['answer_token_ranges']):
            item = {}
            item['context'] = context
            item['question'] = question
            answer_start,answer_end = map(int,answer_se.split(',')[0].split(':'))
            item['answer_start'] = answer_start
            item['answer_end'] = answer_end
            item['answer'] = ' '.join(context.split(' ')[answer_start:answer_end])
            json_data.append(item)
        with open(f'/mnt/ruzhen/ruzhen/RQG/data/newsqa/processed/{mode}.jsonl','w') as fp:
            for item in json_data:
                json.dump(item,fp)
                fp.write('\n')



def child_process(line):
    item = json.loads(line)
    question = nlp(item['question'])
    context = nlp(item['context'])
    answer_set = set()
    if item['answer']:
        answer = nlp(item['answer'])
        answer_set = set([t for t in answer] + [t.lemma_ for t in answer])
    context_set = set([t for t in context] + [t.lemma_ for t in context])
    
    new_token_list = []
    for token in question:
        if str(token) in stop_words or str(token).lower() in stop_words or \
            ((token not in context_set and token not in answer_set and token.lemma_ not in context_set and token.lemma_ not in answer_set) \
        and str(token.tag_) != 'NN' and str(token.tag_) != 'NN' and str(token.tag_) != 'NNP' and str(token.tag_) != 'NNPS') :
            new_token_list.append(str(token))
        elif len(new_token_list) == 0 or new_token_list[-1] != '[MASK]':
            new_token_list.append('[MASK]')
    item['question_skeleton'] = ' '.join(new_token_list)
    return item
def process_v3(root_dir='/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split1/processed/'): 
    # retain stopwords

    for mode in ['train','dev','test']:
        file_path = root_dir + mode + '.jsonl'
        data = []
        with open(file_path,'r') as fp:
            lines = fp.readlines()
            with Pool(20) as pool:
                data = list(tqdm(pool.imap(child_process,lines),total=len(lines)))
        save_dir = root_dir + mode + '.jsonl'
        with open(save_dir,'w') as fp:
            for item in data:
                json.dump(item,fp)
                fp.write('\n')


def get_question_skeleton(questions,hl_contexts):
    stop_words = read_stopwords()
    stop_words.extend(list(retain_words))
    stop_words = set(stop_words)
    contexts = []
    answers = []
    for hl_c in hl_context:
        hl_c_list = hl_c.split(' ')
        start,end = -1,-1
        for i in range(len(hl_c_list)):
            if start != -1 and hl_c_list[i] == '<HL>':
                end = i
                break
            if hl_c_list[i] == '<HL>':
                start = i
        else:
            end = len(hl_c_list)
        answers.append(' '.join(hl_c_list[start+1:end]))
        contexts.append(' '.join(hl_c_list[:start] + hl_c_list[start+1:end] + hl_c_list[end+1:]))
        for question,context,answer in zip(questions,contexts,answers):
            question = nlp(question)
            context = nlp(context)
            answer = nlp(answer)
        context_set = set([t for t in context] + [t.lemma_ for t in context])
        answer_set = set([t for t in answer] + [t.lemma_ for t in answer])
        # save entity
        new_token_list = []
        entity = []
        question_skeleton = []
        for token in question:
            if str(token) in stop_words or str(token).lower() in stop_words or \
                ((token not in context_set and token not in answer_set and token.lemma_ not in context_set and token.lemma_ not in answer_set) \
            and str(token.tag_) != 'NN' and str(token.tag_) != 'NN' and str(token.tag_) != 'NNP' and str(token.tag_) != 'NNPS') :
                new_token_list.append(str(token))
            elif len(new_token_list) == 0 or new_token_list[-1] != '[MASK]':
                new_token_list.append('[MASK]')
            question_skeleton.append(' '.join(new_token_list))
        return question_skeleton

def corrupt(root_dir = '/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split1/processed/'):
    all_entities = set()
    data = []
    file_path = root_dir + 'train' + '.jsonl'
    with open(file_path,'r') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            data.append(item)
            try:
                question = nlp(item['question'])
            except:
                question = nlp(item['tgt'])
            for ents in question.ents:
                all_entities.add(str(ents.text))
                
    all_entities = list(all_entities)
    new_data = []
    for k in range(len(data)): # corrupt train data
        prob = np.random.rand()
        word_list = data[k]['question_skeleton'].strip().split(' ')
        if prob < 0.4: # replace mask to another noun
            mask_positions = []
            for i in range(len(word_list)):
                if word_list[i] == '[MASK]':
                    mask_positions.append(i)
            if len(mask_positions) == 0: # add
                word_list.insert(np.random.randint(len(word_list)),all_entities[np.random.randint(len(all_entities))])
            elif len(mask_positions) == 1:
                word_list[mask_positions[0]] = all_entities[np.random.randint(len(all_entities))]
            else:
                word_list[mask_positions[np.random.randint(len(mask_positions))]] = all_entities[np.random.randint(len(all_entities))]
        elif prob < 0.6: # add noun span
            word_list.insert(np.random.randint(len(word_list)),all_entities[np.random.randint(len(all_entities))])
        elif prob < 0.7: # direct replace skeleton 
            word_list = data[np.random.randint(len(data))]['question_skeleton'].split(' ')
        elif prob < 0.8: # delete  token
            idx = np.random.randint(len(word_list))
            while word_list[idx] == '[MASK]' and len(word_list) != 1:
                idx = np.random.randint(len(word_list))
            word_list.pop(idx)
        data[k]['question_skeleton'] = ' '.join(word_list)
    save_dir = '/' + '/'.join(root_dir.split('/')[:-2]) + '/corrupt/' + 'train.jsonl'
    with open(save_dir,'w') as fp:
        for item in data:
            json.dump(item,fp)
            fp.write('\n')



def caculate_length(root_dir='/mnt/ruzhen/ruzhen/RQG/data/newsqa/processed/'):
    length = []
    with open(root_dir + 'train.jsonl','r') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            length.append(len(item['context'].split(' ')))
    down_1,down_2 = 0,0,
    for l in length:
        if l <= 512:
            down_1 += 1
        if l <= 128:
            down_2 += 1
    print(len(length))
    print(f'down_1:{down_1}')
    print(f'down_2:{down_2}')

def process_quora(root_dir='/mnt/ruzhen/ruzhen/RQG/data/data_qqp/qqp-splitforgeneval/'):
    for mode in ['train', 'dev','test']:
        new_data = []
        with open(root_dir + mode + '.jsonl','r') as fp:
            for line in fp.readlines():
                data = json.loads(line)
                item = {}
                item['src'] = data['sem_input']
                item['tgt'] = data['tgt']
                item['ref'] = data['paras']
                new_data.append(item)
        with open(f'/mnt/ruzhen/ruzhen/RQG/data/data_qqp/processed/{mode}.jsonl','w') as fp:
            for item in new_data:
                json.dump(item,fp)
                fp.write('\n')
def process_mscoco(root_dir='/mnt/ruzhen/ruzhen/RQG/data/data_mscoco/mscoco-eval/'):
    for mode in ['train', 'dev','test']:
        new_data = []
        with open(root_dir + mode + '.jsonl','r') as fp:
            for line in fp.readlines():
                data = json.loads(line)
                item = {}
                item['src'] = data['sem_input']
                item['tgt'] = data['tgt']
                item['ref'] = data['paras']
                new_data.append(item)
        with open(f'/mnt/ruzhen/ruzhen/RQG/data/data_mscoco/processed/{mode}.jsonl','w') as fp:
            for item in new_data:
                json.dump(item,fp)
                fp.write('\n')

def process_v3_for_paraphrase_generation(root_dir='/mnt/ruzhen/ruzhen/RQG/data/data_qqp/processed/'): 
    # retain stopwords
    stop_words = read_stopwords()
    stop_words.extend(list(retain_words))
    stop_words = set(stop_words)
    for mode in ['train','dev','test']:
        file_path = root_dir + mode + '.jsonl'
        data = []
        with open(file_path,'r') as fp:
            for line in fp.readlines():
                item = json.loads(line)
                question = nlp(item['tgt'])
                context = nlp(item['src'])
                context_set = set([t for t in context] + [t.lemma_ for t in context])
                question_set = set([t for t in question] + [t.lemma_ for t in question])
                
                # save entity
                new_token_list = []
                for token in question:
                    if str(token) in stop_words or str(token).lower() in stop_words or \
                     ((token not in context_set and token.lemma_ not in context_set) \
                    and str(token.tag_) != 'NN' and str(token.tag_) != 'NN' and str(token.tag_) != 'NNP' and str(token.tag_) != 'NNPS') :
                        new_token_list.append(str(token))
                    elif len(new_token_list) == 0 or new_token_list[-1] != '[MASK]':
                        new_token_list.append('[MASK]')
                item['question_skeleton'] = ' '.join(new_token_list)
                # for the original context
                new_token_list = []
                for token in context:
                    if str(token) in stop_words or str(token).lower() in stop_words or \
                     (token not in question_set and token.lemma_ not in question_set) \
                    and str(token.tag_) != 'NN' and str(token.tag_) != 'NN' and str(token.tag_) != 'NNP' and str(token.tag_) != 'NNPS' :
                        new_token_list.append(str(token))
                    elif len(new_token_list) == 0 or new_token_list[-1] != '[MASK]':
                        new_token_list.append('[MASK]')
                item['original_context_skeleton'] = ' '.join(new_token_list)
                data.append(item)
        save_dir = root_dir + mode + '.jsonl'
        with open(save_dir,'w') as fp:
            for item in data:
                json.dump(item,fp)
                fp.write('\n')
def process_for_pair_dataset(root_dir='/mnt/ruzhen/ruzhen/RQG/data/data_qqp/training-triples/qqp-clusters-chunk-extendstop-realexemplars-resample-drop30-N5-R100/'):
    for mode in ['train', 'dev', 'test']:
        file_path = root_dir + mode + '.jsonl'
        data = []
        with open(file_path,'r') as fp:
            for line in fp.readlines():
                item = json.loads(line)
                data.append({'q1':item['sem_input'],'q2':item['tgt'],'label':1})
                data.append({'q1':item['syn_input'],'q2':item['tgt'],'label':0})
        dataset = 'data_qqp' if 'qqp' in root_dir else 'data_mscoco'
        with open(f'/mnt/ruzhen/ruzhen/RQG/data/{dataset}/paraphrase_pairs/{mode}.jsonl','w') as fp:
            for item in data:
                json.dump(item, fp)
                fp.write('\n')

def process_for_with_null_skeleton(path='/mnt/ruzhen/ruzhen/RQG/data/newsqa/processed/'):
    for mode in ['train','dev','test']:            
        file_path = f'{path}{mode}.jsonl'
        if mode == 'train':
            file_path = file_path.replace('processed', 'corrupt')
        new_data = []
        with open(file_path,'r') as fp:
            lines = fp.readlines()
            for line in lines:
                item = json.loads(line)
                new_data.append(item)
                if np.random.rand() < 0.5:
                    new_item = copy.deepcopy(item)
                    new_item['question_skeleton'] = None
                    new_data.append(new_item)
        save_path = path.replace('processed', 'processed_with_null_skeleton')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(f'{save_path}{mode}.jsonl','w') as fp:
            for item in new_data:
                json.dump(item, fp)
                fp.write('\n')

def get_whole_dataset(path='/mnt/ruzhen/ruzhen/RQG/data/all_datasets/'):
    all_datasets = []
    for data_name in ['mrqa','narrativeqa','boolq','quoref','drop','mctest',]:
        data_path = os.path.join(path,data_name)
        dataset = load_from_disk(data_path)
        all_datasets.append(dataset)
    train_datasets = [d["train"].shuffle()
                    for d in all_datasets if "train" in d.keys()]
    train_dataset = concatenate_datasets(train_datasets)
    with open(f'/mnt/ruzhen/ruzhen/RQG/data/all_datasets/processed/train.jsonl','w') as fp:
        for item in train_dataset:
            json.dump(item, fp)
            fp.write('\n')
if __name__ == '__main__':
    # s = 'When did Tugh Temur die?\n'
    # result = nlp(s)
    # tag = [[r,r.tag_] for r in result]
    # print(tag)
    # caculate_length('D:/research/RAST/data/SQuAD_1.1_split1/processed/')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='newsqa')
    parser.add_argument('--home', type=str, default='/home/student2020/gouqi/RAST')

    args = parser.parse_args()

    stop_words = read_stopwords(args.home)
    stop_words.extend(list(retain_words))
    stop_words = set(stop_words)


    if args.dataset == 'SQuAD_1.1_split1':
        # process_squad_split1('/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split1/')
        # process_v3('/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split1/processed/') # 提取skeleton
        process_for_with_null_skeleton(f'{args.home}/data/SQuAD_1.1_split1/processed/')

        # corrupt('/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split1/processed/')
    if args.dataset == 'SQuAD_1.1_split2':
        process_for_with_null_skeleton(f'{args.home}/data/SQuAD_1.1_split2/processed/')
        # process_raw_squad('/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split2/') # 我们使用的是answer-aware的
        # process_squad_split2('/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split2/') # 这个是answer-unaware的，没有用
        # process_v3('/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split2/processed_v2/')
        # corrupt('/mnt/ruzhen/ruzhen/RQG/data/SQuAD_1.1_split2/processed/')
    elif args.dataset == 'newsqa':
        process_for_with_null_skeleton(f'{args.home}/data/newsqa/processed/')
        # process_raw_newsqa('/mnt/ruzhen/ruzhen/RQG/data/newsqa/split_data/')
        # process_v3('/mnt/ruzhen/ruzhen/RQG/data/newsqa/processed/')
        # corrupt(f'{args.home}/data/newsqa/processed/')
    elif args.dataset == 'all':
        # get_whole_dataset()
        process_v3(args.home + '/data/all_datasets/processed/')
