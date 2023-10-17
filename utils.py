import nltk
from rouge import Rouge
import torch
from torch import nn
import numpy as np
import spacy
import os
import json
from collections import Counter,defaultdict
# from multiprocessing import Pool
from itertools import chain
from datasets import load_metric
# from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
from nltk.translate import meteor
# from nltk import word_tokenize
# cm = SmoothingFunction()
from sacrebleu.metrics import BLEU, CHRF, TER
import sacrebleu
bleu = BLEU(effective_order=True)
def flatten(list_of_lists):
    return list(chain(*list_of_lists))


def get_sent_bleu(h_r):
    h, r = h_r
    # smoothing method: Chin-Yew Lin and Franz Josef Och (COLING 2004)
    return bleu.corpus_score(h, [r]).score


def get_sent_bleu_list(hyps, ref, n_process=4):
    sent_bleu_list = []
    for item in hyps:
        sent_bleu_list.append(bleu.corpus_score([item],[ref]).score)
        # sent_bleu_list.append(sacrebleu.corpus_bleu([item], [ref],
        #                                smooth_method="exp",
        #                                smooth_value=0.0,
        #                                force=False,
        #                                lowercase=False,
        #                                tokenize="intl",
        #                                use_effective_order=True).score)

    # with Pool(n_process) as pool:
    #     sent_bleu_list = [get_sent_bleu(item) for item in zip(hyp,ref)]

    return sent_bleu_list


def oracle_bleu(hyp_list, ref, n_process=4):
    assert len(set([len(h) for h in hyp_list])) == 1

    all_hyp_sentence_bleu_list = []
    for i in range(len(ref)):
        hyp = [item[i] for item in hyp_list]
        all_hyp_sentence_bleu_list.append(get_sent_bleu_list(hyp, [ref[i]], n_process=n_process))


    max_hyp_index_list = [np.argmax(item) for item in zip(all_hyp_sentence_bleu_list)]
    # if n_process > len(hyp_list[0]):
    #     n_process = len(hyp_list[0])

    # with Pool(n_process) as pool:
    #     max_hyp_index_list = list(tqdm(pool.imap(np.argmax, zip(*all_hyp_sentence_bleu_list)),
    #                                    total=len(all_hyp_sentence_bleu_list)))

    best_hyp_list = []
    for i, max_hyp_index in enumerate(max_hyp_index_list):
        best_hyp = hyp_list[max_hyp_index][i]
        best_hyp_list.append(best_hyp)

    return bleu.corpus_score(best_hyp_list, [ref]).score
    # return sacrebleu.corpus_bleu(best_hyp_list, [ref],
    #                                    smooth_method="exp",
    #                                    smooth_value=0.0,
    #                                    force=False,
    #                                    lowercase=False,
    #                                    tokenize="intl",
    #                                    use_effective_order=False).score


def avg_bleu(hyp_list, ref):
    all_ref = []
    for i in range(len(hyp_list)):
        all_ref.extend(ref)
    return sacrebleu.corpus_bleu(flatten(hyp_list), [all_ref],
                                       smooth_method="exp",
                                       smooth_value=0.0,
                                       force=False,
                                       lowercase=False,
                                       tokenize="intl",
                                       use_effective_order=True).score
    # return bleu.corpus_score(flatten(hyp_list),[all_ref]).score


def get_qg_pairwise_bleu(hyps):
    sent_bleu_list = []
    corpus_bleu_list = []
    for i in range(len(hyps)):
        ref = hyps[:i] + hyps[i + 1:]
        sent_bleu_list.append(bleu.sentence_score(hyps[i],ref).score)
        # corpus_bleu_list.append(bleu.corpus_score([hyps[i]],[[r] for r in ref]).score)
    return np.mean(sent_bleu_list)


def qg_pairwise_bleu(hyp_list, n_process=4):

    assert len(set([len(h) for h in hyp_list])) == 1


    pairwise_bleu_list = [get_qg_pairwise_bleu(item) for item in zip(*hyp_list)]
    # if n_process > len(hyp_list):
    #     n_process = len(hyp_list)
    
    # with Pool(n_process) as pool:
    #     pairwise_bleu_list = list(tqdm(pool.imap(get_qg_pairwise_bleu, zip(*hyp_list)),
    #                                total=len(hyp_list)))

    return np.mean(pairwise_bleu_list)

def get_paraphrase_pairwise_bleu(hyps):
    sent_bleu_list = []
    corpus_bleu_list = []
    for i in range(len(hyps)):
        ref = hyps[:i] + hyps[i + 1:]
        for r in ref:
            sent_bleu_list.append(bleu.corpus_score([hyps[i]], [[r]]).score)
        sent_bleu_list.append(bleu.sentence_score(hyps[i],ref).score)
        # corpus_bleu_list.append(bleu.corpus_score([hyps[i]],[[r] for r in ref]).score)

    return np.mean(sent_bleu_list)
def paraphrase_pairwise_bleu(hyp_list, n_process=4):

    assert len(set([len(h) for h in hyp_list])) == 1


    pairwise_bleu_list = [get_paraphrase_pairwise_bleu(item) for item in zip(*hyp_list)]
    # if n_process > len(hyp_list):
    #     n_process = len(hyp_list)

    # with Pool(n_process) as pool:
    #     pairwise_bleu_list = list(tqdm(pool.imap(get_qg_pairwise_bleu, zip(*hyp_list)),
    #                                total=len(hyp_list)))

    return np.mean(pairwise_bleu_list)
def calculate_bleu(output_data):
    ref = [item['ground_truth'] for item in output_data]
    best_predict = [item['best_predicted'] for item in output_data]
    all_predict = list(zip(*[item['predicted'] for item in output_data]))
    # top_1_bleu_score = bleu.corpus_score(best_predict,[ref]).score
    top_1_bleu_score = sacrebleu.corpus_bleu(best_predict, [ref],
                                       smooth_method="exp",
                                       smooth_value=0.0,
                                       force=False,
                                       lowercase=False,
                                       tokenize="intl",
                                       use_effective_order=True).score
    oracle_bleu_score = oracle_bleu(all_predict, ref)
    pairwise_bleu_score = qg_pairwise_bleu(all_predict)
    average_bleu_score = avg_bleu(all_predict, ref)
    # oracle bleu
    return top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score
def calculate_rouge(output_data,use_dlc=None):

    
    preds,labels = map(list,zip(*[[item['best_predicted'], item['ground_truth']] for item in output_data]))
    
    # rouge.Rouge
    # rouge = Rouge()
    # score = rouge.get_scores(preds, labels, avg=True)
    # return score['rouge-l']['f']
    
    # hugginface rouge
    root_dir = '/mnt/workspace/gouqi/code/RQG'
    if use_dlc:
        root_dir = '/root/data/ruzhen/RQG'
    rouge = load_metric(root_dir + "/rouge.py")
    rouge_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    rouge_labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    result = rouge.compute(
        predictions=rouge_preds, 
        references=rouge_labels,
        use_stemmer=True
    )
    return result['rougeL'].mid.fmeasure
def calculate_ppl(model,data_loader,device,label_pad_token_id=-100):
    ppl = 0
    total = 0
    for inputs in data_loader:
        input_ids= inputs['input_ids'].unsqueeze(1).to(device)
        attention_mask = inputs['attention_mask'].unsqueeze(1).to(device)
        labels = inputs['labels'].unsqueeze(1).to(device)
        decoder_input_ids = inputs['decoder_input_ids'].unsqueeze(1).to(device)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_input_ids=decoder_input_ids
                )
        for logit,label in zip(output.logits,labels):
            loss = loss_fct(logit, label)
            logp = -loss
            ppl += np.exp(logp.data.item())
            total += 1
    return ppl / total
def calculate_dist(output_best_ids):
    output_best_ids = [[id for id in ids.cpu().detach().numpy() if id != 2 and id != 0 and id != 1] for ids in output_best_ids]
    dist_1_gram = defaultdict(int)
    dist_2_gram = defaultdict(int)
    for item in output_best_ids:
        for w_id in item:
            dist_1_gram[w_id] += 1
        for i in range(0,len(item)-2):
            dist_2_gram[tuple(item[i:i+2])] += 1 
    return len(dist_1_gram) / sum(dist_1_gram.values()), len(dist_2_gram) / sum(dist_2_gram.values())


def calculate_bleu_for_paraphrase(output_data,prefix,is_quora,use_dlc):
    # ref 有多个， #for 
    dataset = 'data_qqp' if is_quora else 'data_mscoco'
    if use_dlc:
        data_path = f'/root/data/ruzhen/RQG/data/{dataset}/processed/{prefix}.jsonl'
    else:
        data_path = f'/mnt/ruzhen/ruzhen/RQG/data/{dataset}/processed/{prefix}.jsonl'
    data = []
    with open(data_path,'r') as fp:
        for line in fp.readlines():
            item = json.loads(line)
            data.append(item)
    refs = [item['ref'] for item in data]
    max_num_refs = max([len(x) for x in refs])
    refs_padded = [x + [x[0]] * (max_num_refs - len(x)) for x in refs]

    origin_input = [item['context'] for item in output_data]
    best_predict = [item['best_predicted'] for item in output_data]
    all_predict = list(zip(*[item['predicted'] for item in output_data]))

    top_1_bleu_score = sacrebleu.corpus_bleu(best_predict,list(zip(*refs_padded)), lowercase=True).score
    self_bleu_score = sacrebleu.corpus_bleu(best_predict, [origin_input], lowercase=True).score
    pairwise_bleu_score = paraphrase_pairwise_bleu(all_predict)
    alpha = 0.7
    ibleu_7 = alpha * top_1_bleu_score - (1 - alpha) * self_bleu_score
    alpha = 0.8
    ibleu_8 = alpha * top_1_bleu_score - (1 - alpha) * self_bleu_score
    alpha = 0.9
    ibleu_9 = alpha * top_1_bleu_score - (1 - alpha) * self_bleu_score
    mean_ibleu = []
    alpha = 0.8
    for each_predict in all_predict:
        each_bleu_score = sacrebleu.corpus_bleu(each_predict,list(zip(*refs_padded)), lowercase=True).score
        each_self_bleu_score = sacrebleu.corpus_bleu(each_predict, [origin_input], lowercase=True).score
        mean_ibleu.append(alpha * each_bleu_score - (1 - alpha) * each_self_bleu_score)
    mean_ibleu = sum(mean_ibleu) / len(mean_ibleu)
    # oracle_bleu_score = oracle_bleu(all_predict, ref)
    # average_bleu_score = avg_bleu(all_predict, ref)
    # oracle bleu
    return top_1_bleu_score,self_bleu_score,ibleu_7,ibleu_8,ibleu_9,mean_ibleu,pairwise_bleu_score


def calculate_meteor(output_data):
    preds,labels = map(list,zip(*[[item['best_predicted'], item['ground_truth']] for item in output_data]))
    labels = [[word_tokenize(r)] for r in labels]
    preds = [word_tokenize(r) for r in preds]
    return round(meteor(labels, preds),4)




  
def read_stopwords(root_dir = '/home/student2021/gouqi/RAST'):
    stopwords = []
    with open(os.path.join(root_dir,'data/stopwords.txt'), 'r') as fp:
        for line in fp.readlines():
            stopwords.append(line.strip())
    return stopwords

def get_question_skeleton_for_qg(questions,hl_contexts,use_dlc=False):
    nlp = spacy.load("en_core_web_sm")
    retain_words = {'what','?','who','how','much','many','where','which','when','why','do','did','does',
                'mean', 'long', 'take', 'get', 'good', 'best', 'find', 'anyone', 'get', 'difference',
                'best', 'better', 'and', 'or', 'people', 'happen', 'happens', 'would', 'affect', 'effect',
                'effects', 'get', 'way', 'make', 'best', 'know', 'think', 'would', 'like', 'anyone',
    }
    if not use_dlc:
        stop_words = read_stopwords()
    else:
        stop_words = read_stopwords('/root/data/ruzhen/')
    stop_words.extend(list(retain_words))
    stop_words = set(stop_words)
    contexts = []
    answers = []
    for hl_c in hl_contexts:
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
    question_skeleton = []
    for question,context,answer in zip(questions,contexts,answers):
        question = nlp(question)
        context = nlp(context)
        answer = nlp(answer)
        context_set = set([t for t in context] + [t.lemma_ for t in context])
        answer_set = set([t for t in answer] + [t.lemma_ for t in answer])
        new_token_list = []
        for token in question:
            if str(token) in stop_words or str(token).lower() in stop_words or \
                ((token not in context_set and token not in answer_set and token.lemma_ not in context_set and token.lemma_ not in answer_set) \
            and str(token.tag_) != 'NN' and str(token.tag_) != 'NN' and str(token.tag_) != 'NNP' and str(token.tag_) != 'NNPS') :
                new_token_list.append(str(token))
            elif len(new_token_list) == 0 or new_token_list[-1] != '[mask]':
                new_token_list.append('[mask]')
        question_skeleton.append(' '.join(new_token_list))
    return question_skeleton


def get_question_skeleton_for_paraphrase(questions,contexts,use_dlc=False):
    nlp = spacy.load("en_core_web_sm")
    retain_words = {'what','?','who','how','much','many','where','which','when','why','do','did','does',
                'mean', 'long', 'take', 'get', 'good', 'best', 'find', 'anyone', 'get', 'difference',
                'best', 'better', 'and', 'or', 'people', 'happen', 'happens', 'would', 'affect', 'effect',
                'effects', 'get', 'way', 'make', 'best', 'know', 'think', 'would', 'like', 'anyone',
    }
    if not use_dlc:
        stop_words = read_stopwords()
    else:
        stop_words = read_stopwords('/root/data/ruzhen/')
    stop_words.extend(list(retain_words))
    stop_words = set(stop_words)
    question_skeleton = []
    for question,context in zip(questions,contexts):
        question = nlp(question)
        context = nlp(context)
        context_set = set([t for t in context] + [t.lemma_ for t in context])
        new_token_list = []
        for token in question:
            if str(token) in stop_words or str(token).lower() in stop_words or \
                ((token not in context_set and token.lemma_ not in context_set) \
            and str(token.tag_) != 'NN' and str(token.tag_) != 'NN' and str(token.tag_) != 'NNP' and str(token.tag_) != 'NNPS') :
                new_token_list.append(str(token))
            elif len(new_token_list) == 0 or new_token_list[-1] != '[mask]':
                new_token_list.append('[mask]')
        question_skeleton.append(' '.join(new_token_list))
    return question_skeleton

def caculate_jaccard_similarity(s1,s2):
    words_1 = set(s1.lower().split())
    words_2 = set(s2.lower().split())
    intersection = words_1.intersection(words_2)
    union = words_1.union(words_2)
    return float(len(intersection) / len(union))

def caculate_jaccard_distance(s1,s2):
    words_1 = set(s1.lower().split())
    words_2 = set(s2.lower().split())
    intersection = words_1.intersection(words_2)
    union = words_1.union(words_2)
    return 1 - float(len(intersection) / len(union))


if __name__ == '__main__':

    # 使用sacrebleu计算bleu值
    # bleu.corpus_score（hyp,ref）输入格式： 
    # hyp: list[str]
    # ref: list[list[str]]
    #
    # bleu.sentence_score(jyp,ref),输入格式：
    # hyp:str
    # ref:list[str]
    hyp1 = ['the dog is found under the big funny bed','how are you?']
    hyp2 = ['the cat was under the big funny bed','hwo do you do? ']
    hyp3 = ['the dragon was found under the big funny bed','how about you?']

    ref = ['that cat was under the bed.','how about you?']
    bleu.sentence_score(hyp1[0], ref)
    print('Corpus BLEU with hyp 1')
    print(bleu.corpus_score(hyp1, [ref]))
    print('Corpus BLEU with hyp 2')
    print(bleu.corpus_score(hyp2, [ref]))
    print('Corpus BLEU with hyp 3')
    print(bleu.corpus_score(hyp3, [ref]))
    print('Oracle BLEU')
    print(oracle_bleu([hyp1, hyp2, hyp3], ref))
    print('\nAverage BLEU')
    print(avg_bleu([hyp1, hyp2, hyp3], ref))
    print('SELF BLEU')

    print(qg_pairwise_bleu([hyp1, hyp2, hyp3]))
