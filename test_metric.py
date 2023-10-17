import nltk
import json
from tqdm import tqdm
import torch
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, T5Tokenizer, Trainer
import numpy as np
from eval_squad import compute_exact, compute_f1
bleu = BLEU()
def read_zehua(json_path_or_file):
    if isinstance(json_path_or_file, str):
        with open(json_path_or_file, "r") as f:
            data = json.load(f)[1:]
    else:
        data = json_path_or_file
    golds = []
    preds = []
    sys = [sen['pred'] for sen in data]
    ref = [sen['gold'] for sen in data]
    return bleu.corpus_score(sys, [ref])

def get_qg_pairwise_bleu(hyps):
    sent_bleu_list = []
    corpus_bleu_list = []
    for i in range(len(hyps)):
        ref = hyps[:i] + hyps[i + 1:]
        # sent_bleu_list.append(bleu.sentence_score(hyps[i],ref).score)
        sent_bleu_list.append(sacrebleu.sentence_bleu(hyps[i],ref).score)
        # sent_bleu_list.append(sacrebleu.corpus_bleu([hyps[i]], [[r] for r in ref],
        #                                smooth_method="exp",
        #                                smooth_value=0.0,
        #                                force=False,
        #                                lowercase=False,
        #                                tokenize="intl",
        #                                use_effective_order=False).score)
        # corpus_bleu_list.append(bleu.corpus_score([hyps[i]],[[r] for r in ref]).score)
        # for r in ref:
        #     sent_bleu_list.append(bleu.corpus_score([hyps[i]], [[r]]).score)
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

def nltk_bleu_rag(json_path_or_file): # nltk bleu
    if isinstance(json_path_or_file, str):
        with open(json_path_or_file, "r") as f:
            data = list(f.readlines())
    else:
        data = json_path_or_file

    length = len(data)
    golds = []
    preds = []
    for i in tqdm(range(length//26), desc="Reading rag nltk"):
        gold = nltk.word_tokenize(data[i*26+3].split(':')[1].strip())
        pred = []
        for j in range(5):
            pre = nltk.word_tokenize(data[i*26+9 + j * 4].split(':')[1].strip())
            pred.append(pre)
        preds.append(pred)
        golds.append([gold])   
    all_pred = list(zip(*preds))
    top_1_pred = all_pred[0]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(golds, top_1_pred)
    from nltk_bleu import self_bleu
    self_bleu_score = self_bleu(all_pred)
    return bleu_score,self_bleu_score

def nltk_bleu_qg(json_path_or_file): # nltk bleu
    if isinstance(json_path_or_file, str):
        with open(json_path_or_file, "r") as f:
            data = list(f.readlines())
    else:
        data = json_path_or_file

    length = len(data)
    golds = []
    preds = []
    for i in tqdm(range(length//9), desc="Reading qg nltk"):
        gold = nltk.word_tokenize(data[i*9+1].split(':')[1].strip())
        pred = []
        for j in range(5):
            pre = nltk.word_tokenize(data[i*9+3+j].strip())
            pred.append(pre)
        preds.append(pred)
        golds.append([gold])   
    all_pred = list(zip(*preds))
    top_1_pred = all_pred[0]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(golds, top_1_pred)
    from nltk_bleu import self_bleu
    self_bleu_score = self_bleu(all_pred)
    return bleu_score,self_bleu_score

def sacrebleu_rag(json_path_or_file): # sacrebleu
    if isinstance(json_path_or_file, str):
        with open(json_path_or_file, "r") as f:
            data = list(f.readlines())
    else:
        data = json_path_or_file
    length = len(data)
    output_data = []
    for i in tqdm(range(length//26), desc="Reading rag sacrebleu"):
        item = {}
        item['ground_truth'] =  data[i*26+3].split(':')[1].strip()
        item['predicted'] = []
        for j in range(5):
            item['predicted'].append(data[i*26+9+j*4].split(':')[1].strip())
        item['best_predicted'] = item['predicted'][0]
        output_data.append(item)
    from utils import  calculate_bleu
    top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score = calculate_bleu(output_data)
    return top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score




def sacrebleu_qg(json_path_or_file): # sacrebleu
    if isinstance(json_path_or_file, str):
        with open(json_path_or_file, "r") as f:
            data = list(f.readlines())
    else:
        data = json_path_or_file

    length = len(data)
    golds = []
    preds = []
    best_preds = []
    output_data = []
    for i in tqdm(range(length//9), desc="Reading qg sacrebleu"):
        item = {}
        item['ground_truth']=data[i*9+1].split(':')[1].strip()
        item['predicted'] = []
        pred = []
        for j in range(5):
            item['predicted'].append(data[i*9+3+j].strip())
        item['best_predicted']=item['predicted'][0]
        output_data.append(item)
    from utils import  calculate_bleu
    top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score = calculate_bleu(output_data)
    return top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score


class T5QARewardModel():
    def __init__(self,device,reward_model_name_or_path,max_source_length):
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
    def get_scores(self,hl_context, prediction):
        # split hl_context to context and answer
        context = []
        answer = []
        for hl_c in hl_context:
            answer.append(hl_c.split('<HL>')[1])
            context.append(hl_c.replace('<HL>',''))
        
        # alter input format for qa
        inputs = []
        for con,pred in zip(context,prediction):
            inputs.append(con+ ' <sep> ' + pred)

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
        output_tok['input_ids'] = output_tok['input_ids'].masked_fill(output_tok['input_ids'] == 0,-100) # label 0 -> -100
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
        for logit,label in zip(output.logits,output_tok['input_ids'].to(self.device)):
            loss = loss_fct(logit, label)
            logp = -loss
            rewards.append(np.exp(logp.data.item()))
            # rewards.append(logp.data.item())

        # em, f1
        output = self.model.generate(
                    input_ids=input_tok['input_ids'].to(self.device),
                    attention_mask=input_tok['attention_mask'].to(self.device),
                    num_beams = 5,
                    max_length = 64,
                    do_sample = False,
                    num_return_sequences = 1,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        em = [compute_exact(a,o) for a,o in zip(answer,output)]
        f1 = [compute_f1(a,o) for a,o in zip(answer,output)]
        return np.mean(em),np.mean(f1),np.array(rewards)


def main():
    rag = r"/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAG_origin_lr_1e-6_pregenertor/experiments_end2end_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeleton_train_context_encoderFalsereward_typeqa/sm_reward_typejaccard_0.6_rl_baseTrue_sm_baseFalse/train_retrieve_num50_eval_retrieve_num300_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTrue/18.171641865526595/output_data.txt"
    qg = r"/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1_pre/QG/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr5e-05-b32-beam_search5/output_data.txt"
    # top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score = sacrebleu_qg(qg)
    # print(f"sacrebleu  bleu socre: {bleu_score},self_bleu:{self_bleu_score}")
    # bleu_score,self_bleu_score = nltk_bleu_qg(qg)
    # print(f"nltk bleu socre: {bleu_score},self_bleu:{self_bleu_score}")
    
    top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score = sacrebleu_rag(rag)
    print(f"sacrebleu  top1 bleu socre: {top_1_bleu_score},self_bleu:{pairwise_bleu_score},oracle_bleu:{oracle_bleu_score}")
    bleu_score,self_bleu_score = nltk_bleu_rag(rag)
    print(f"nltk bleu socre: {bleu_score},self_bleu:{self_bleu_score}")

def test_google_result(file_name='/mnt/ruzhen/ruzhen/frost_composition_sampling/frost_composition_sampling/all_predictions/qgen_du_all_test_predictions.json'):

    with open(file_name,'r') as fp:
        data = json.load(fp)
    frost = []
    frost__ = []
    output_data = []
    for item in tqdm(data.values()):
        new_item= {}
        new_item['ground_truth'] = item['target']
        new_item['predicted'] = []
        for each in item['composition(frost)'][0]:
            new_item['predicted'].append(each.split('[SUMMARY]')[-1].strip())
        new_item['best_predicted'] = new_item['predicted'][0]   
        output_data.append(new_item) 
    return output_data
    # from utils import calculate_bleu
    # top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score = calculate_bleu(output_data)
    # print(top_1_bleu_score,pairwise_bleu_score,oracle_bleu_score,average_bleu_score)

def read_ours(path,reward_model):
    if '.txt' in path:
        data = []
        with open(path) as fp:
            lines = fp.readlines()
            for i in range(len(lines) // 9):
                item = {'predicted':[]}
                item['context'] = lines[i*9].split('context:')[1]
                for j in range(3,8):
                    item['predicted'].append(lines[i*9+j])
                data.append(item)
    elif '.jsonl' in path:
        with open(path) as fp:
            raw_data = fp.read()
            raw_data = raw_data.replace('}\n{', '}<SPLIT>{')
            data_list = raw_data.split('<SPLIT>')
        data = [json.loads(item) for item in data_list]
    elif '.json' in path:
        with open(path) as fp:
            data = json.load(fp)
    rewards = []
    ems = []
    f1s = []
    wrong_num = 0
    for item in tqdm(data):
        prediction = item['predicted']
        hl_context = [item['context'] for _ in range(5)]
        if '<HL>' not in item['context']:
            wrong_num += 1
            continue
        em,f1,reward = reward_model.get_scores(prediction = prediction, hl_context = hl_context)
        rewards.append(np.mean(reward))
        ems.append(em)
        f1s.append(f1)
    print('rewards:',np.mean(rewards))
    print('EM:',np.mean(ems))
    print('F1:',np.mean(f1s))
    print('wrong_num:',wrong_num)
if __name__ == "__main__":

    # json_path = '/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/6/_end2end_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeletontrain_filterv3_num_100_eval_filterv6_num300_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.6rl_baseTrue_sm_baseFalsedo_sampleTrue/19.60683538204789/output_data.json'
    # with open(json_path) as fp:
    #     data = json.load(fp)
    # with open(json_path,'w') as fp:
    #     json.dump(data,fp,indent=2)

    reward_model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_model_name_or_path = "/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/QA/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b32-beam_search5"
    max_combined_length=128
    reward_model = T5QARewardModel(device=reward_model_device,reward_model_name_or_path=reward_model_name_or_path,max_source_length=max_combined_length)
    print('question')
    read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/26/_end2end_retrieve_keyquestion_ctx_retrieve_keyquestiontrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.3rl_baseTrue_sm_baseFalsedo_sampleTrue/20.203616858908255/output_data.json', reward_model)
    print('masked_context')
    read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/26/_end2end_retrieve_keymasked_context_ctx_retrieve_keymasked_contexttrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.3rl_baseTrue_sm_baseFalsedo_sampleTrue/20.29143825783134/output_data.json', reward_model)
    print('w/o diversity-driven sampling')
    read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/27/_end2end_retrieve_keyquestion_ctx_retrieve_keyquestiontrain_filterv2_num_100_eval_filterv1_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.3rl_baseTrue_sm_baseFalsedo_sampleTrue/19.10960620502821/output_data.json', reward_model)
    # print('w/o e2e')
    # read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/21/_rl_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeletontrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.3rl_baseTrue_sm_baseFalsedo_sampleTrue/20.324685285426376/output_data.json', reward_model)
    # print('0.2:\n')
    # read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/15/_end2end_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeletontrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.2rl_baseTrue_sm_baseFalsedo_sampleTrue/20.36757624304246/output_data.json', reward_model)
    # print('0.3:\n')
    # read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/20/_end2end_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeletontrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.3rl_baseTrue_sm_baseFalsedo_sampleTrue/20.332077074841713/output_data.json',reward_model)
    # print('0.4:\n')
    # read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/15/_end2end_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeletontrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.4rl_baseTrue_sm_baseFalsedo_sampleTrue/20.013275448396843/output_data.json',reward_model)
    # print('0.5:\n')
    # read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/13/_end2end_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeletontrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.5rl_baseTrue_sm_baseFalsedo_sampleTrue/19.792295474864638/output_data.json', reward_model)
    # print('0.6:\n')
    # read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/RAST/12/_end2end_retrieve_keyquestion_skeleton_ctx_retrieve_keyquestion_skeletontrain_filterv3_num_100_eval_filterv6_num500_3/rag-sequence-base-len128--warm0.2--warmSteps0--weightDecay0.1-lr5e-06-b16-gamma1.0-lamda0.6-whitenFalse-normalize_rewardTruereward_typeqasm_reward_typejaccard_0.6rl_baseTrue_sm_baseFalsedo_sampleTrue/19.67885774719949/output_data.json', reward_model)
    
    # read_ours('/mnt/ruzhen/ruzhen/RQG/output/SQuAD_1.1_split1/QG/experiments/t5-base-GPUNums1-len128-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b32-top_k30-top_p0.9-temperature2.0/12.226229404503394/output_data.json', reward_model)
    # test_google_result('/mnt/ruzhen/ruzhen/frost_composition_sampling/frost_composition_sampling/all_predictions/qgen_du_all_test_predictions.json')
    
    # main()
    # refs = [  
    #       ['The dog bit the man.','you are a teacher']
    #     ]
    # sys = ['The dog bit the man.','you are a teacher you']



    # BLEU = 29.44 82.4/42.9/27.3/12.5 (BP = 0.889 ratio = 0.895 hyp_len = 17 ref_len = 19)

    # print(bleu.get_signature())
    # nrefs:var|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0