__all__ = ['AdaptiveKLController', 'FixedKLController', 'RAGTrainer']

# Cell
import math
from dataclasses import asdict, dataclass
import numpy as np
import datetime
import torch.nn.functional as F
from torch.optim import Adam
from transformers import TrainingArguments,Seq2SeqTrainer,AutoModelForSeq2SeqLM,AdamW
import torch
import collections
import time
import random
import logging
import copy
from tqdm import tqdm
import os
from utils import *
from torch.utils.tensorboard import SummaryWriter

from core import (logprobs_from_logits,
                   whiten,
                   clip_by_value,
                   entropy_from_logits,
                   flatten_dict,
                   average_torch_dicts,
                   stats_to_np,
                   stack_dicts,
                   add_suffix,
                   counter_pad_tokens, )
# Cell

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


# Cell

class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


# Cell
def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)    
    return (data - min)/(max-min + 1e-8)
class RAGTrainer(Seq2SeqTrainer):
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "adap_kl_ctrl": False,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 0.9,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": .1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "lamda":0.3
    }

    def __init__(self,
                ref_model,
                ref_model_device,
                eval_data_args,
                rl_args,
                eval_collate_fn,
                logger=None,
                **kwargs
                ):
        """
        Initialize PPOTrainer.

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000

        """

        self.eval_data_args = eval_data_args
        self.logger = logger
        self.best_bleu = -float('inf')
        self.ref_model_device = ref_model_device
        self.ref_model = ref_model
        self.ref_model.to(ref_model_device)
        test_dataset = kwargs.pop('test_dataset')
        self.test_dataset = test_dataset
        self.eval_collate_fn = eval_collate_fn
        super().__init__(**kwargs)
        self.model = self.model.to(ref_model_device)
        self.writer =  SummaryWriter(self.args.logging_dir)
        self.rf_params = self.default_params
        self.rf_params.update(asdict(rl_args))
        self.global_steps = 0
        self.rf_params['batch_size'] = self.args.per_device_train_batch_size

        self.calculate_max_train_step()
        self.create_optimizer()
        # self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
        self.create_scheduler(num_training_steps=self.args.max_steps, optimizer=self.optimizer)
        self.dpr_parameters = [p for n, p in self.model.named_parameters() if  n.startswith('rag.question_encoder')]
        if self.rf_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.rf_params['init_kl_coef'],
                                               self.rf_params['target'],
                                               self.rf_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.rf_params['init_kl_coef'])
    
        self.best_bleu = -float('inf')
    def create_optimizer(self):
        def get_parameter_names(model, forbidden_layer_types):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
            result += list(model._parameters.keys())
            return result
        opt_model = self.model_wrapped
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and n.startswith('rag.question_encoder')],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.dpr_learning_rate
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and not n.startswith('rag.question_encoder')],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and n.startswith('rag.question_encoder')],
                    "weight_decay": 0.0,
                    "lr": self.args.dpr_learning_rate
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and not n.startswith('rag.question_encoder')],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    def step(self, contexts, retrieve_candidates, predictions,doc_scores,qa_rewards,sm_rewards,retriever_reward):
        """
        Run a RF optimisation step.
        args:
            query (torch.tensor): tensor containing the encoded queries,
                                  shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses vocab_ids, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size, response_length]

        returns:
            train_stats (dict): a summary of the training statistics
        """
        direct_rewards = qa_rewards + self.rf_params['sm_coef'] * sm_rewards
        if self.args.reward_batch_normalization:
            mean, var = np.mean(direct_rewards), np.var(direct_rewards)
            direct_rewards = (direct_rewards - mean) / np.sqrt(var + 1e-8)
            
        predictions = predictions[:,1:]  # 对t5 generate的id进行处理，删去开始的0
        bs = len(predictions)
        pad_num = counter_pad_tokens(predictions)
        logprobs, ref_logprobs,batch_kl = self.batched_forward_pass(contexts, retrieve_candidates, predictions,pad_num=pad_num)


        self.train_minibatch(logprobs, direct_rewards, predictions,doc_scores=doc_scores,batch_kl=batch_kl,accumulation_steps = len(contexts) // self.rf_params['forward_batch_size'],retriever_reward = retriever_reward)


        # for i in range(len(contexts) // self.rf_params['forward_batch_size']):
            # forward_batch_predictions = predictions[i*self.rf_params['forward_batch_size']:(i+1) * self.rf_params['forward_batch_size']]
            # forward_batch_contexts = contexts[i*self.rf_params['forward_batch_size']:(i+1) * self.rf_params['forward_batch_size']]
            # forward_batch_candidates = retrieve_candidates[i*self.rf_params['forward_batch_size']:(i+1) * self.rf_params['forward_batch_size']]
            # forward_batch_doc_scores = doc_scores[i*self.rf_params['forward_batch_size']:(i+1) * self.rf_params['forward_batch_size']]
            # forward_batch_rewards = rewards[i*self.rf_params['forward_batch_size']:(i+1) * self.rf_params['forward_batch_size']]

            # forward_batch_predictions = forward_batch_predictions[:,1:]  # 对t5 generate的id进行处理，删去开始的0
            # bs = len(forward_batch_predictions)
            # pad_num = counter_pad_tokens(forward_batch_predictions)
            # logprobs, ref_logprobs,batch_kl = self.batched_forward_pass(forward_batch_contexts,forward_batch_candidates, forward_batch_predictions,pad_num=pad_num)

            # # rewards, batch_mean_rewards, batch_kl_mean = self.compute_rewards(scores, logprobs, ref_logprobs, pad_num)

            # self.train_minibatch(logprobs, forward_batch_rewards, forward_batch_predictions,doc_scores=forward_batch_doc_scores,batch_kl=batch_kl,accumulation_steps = len(contexts) // self.rf_params['forward_batch_size'])

        self.kl_ctl.update(batch_kl.cpu().detach().numpy(), bs)
        self.writer.add_scalar('train/rewards', sum(direct_rewards) / len(direct_rewards), self.global_steps)
        self.writer.add_scalar('train/qa_rewards', sum(qa_rewards) / len(qa_rewards), self.global_steps)
        self.writer.add_scalar('train/sm_rewards', sum(sm_rewards) / len(sm_rewards), self.global_steps)
        self.writer.add_scalar('train/kl_value', batch_kl, self.global_steps)
        self.writer.add_scalar('train/kl_ctrl', self.kl_ctl.value, self.global_steps)
        self.writer.flush()
        self.global_steps += 1
    def postprocess_docs(self, retrieve_skeleton,n_docs, contexts,return_tensors="pt", prefix=None):
        def cat_input_and_doc(question_skeleton, context, prefix):
            if question_skeleton:
                if question_skeleton.startswith('"'):
                    question_skeleton = question_skeleton[1:]
                if question_skeleton.endswith('"'):
                    question_skeleton = question_skeleton[:-1]
                if prefix is None:
                    prefix = ""
                if 'newsqa' in self.eval_data_args.data_dir:
                    out = (prefix + question_skeleton + ' <sep> ' + context).replace(
                        "  ", " "
                    )
                else:
                    out = (prefix + context + ' <sep> ' + question_skeleton).replace(
                        "  ", " "
                    )
            else:
                out = context
            return out

        rag_input_strings = [
            cat_input_and_doc(
                skeleton,
                context,
                prefix,
            ) for context,skeleton in zip(contexts,retrieve_skeleton)
        ]

        contextualized_inputs = self.tokenizer.batch_encode_plus( # ragTrainer的tokenizer是generator的tokenizer
            rag_input_strings,
            max_length=self.model.retriever.config.max_combined_length,
            return_tensors=return_tensors,
            padding="longest",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]
    def batched_forward_pass(self, contexts, skeleton, batch_output_ids,pad_num):
        """Calculate model outputs in multiple batches."""
        logprobs = []
        ref_logprobs = []
        decoder_input_ids = self.model.generator.prepare_decoder_input_ids_from_labels(labels=batch_output_ids).to(batch_output_ids)
        # forward pass
        input_ids, attention_mask = self.postprocess_docs(skeleton,self.model.config.n_docs, contexts)
        input_ids = input_ids.to(batch_output_ids)
        attention_mask = attention_mask.to(batch_output_ids)

        forward_batch_outputs = self.model.generator(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=decoder_input_ids)

        ref_forward_batch_outputs = self.ref_model.generator(input_ids=input_ids.to(self.ref_model_device),attention_mask=attention_mask.to(self.ref_model_device),decoder_input_ids=decoder_input_ids.to(self.ref_model_device))

        logits = forward_batch_outputs.logits
        ref_logits = ref_forward_batch_outputs.logits.detach().to(logits)
        # kl divergence

        logprobs,raw_logprobs = logprobs_from_logits(logits, batch_output_ids)
        ref_logprobs,raw_ref_logprobs = logprobs_from_logits(ref_logits, batch_output_ids)

        batch_kl = 0
        gen_len = batch_output_ids.size(1) - pad_num
        for logp_list,ref_logp_list,num in zip(raw_logprobs,raw_ref_logprobs,gen_len):
            for i,(logp, ref_logp) in enumerate(zip(logp_list,ref_logp_list)):
                if i >= num:
                    break
                batch_kl += torch.sum(torch.exp(logp) * (logp - ref_logp),dim=-1)
        batch_kl = batch_kl / len(batch_output_ids)
        # shape:[batch size, generation length]
        return logprobs, ref_logprobs, batch_kl

    def compute_rewards(self, scores, logprobs, ref_logprobs, end_token_idxs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs
        batch_kl = torch.tensor([0.0] * self.rf_params["batch_size"])
        gen_len = kl.shape[-1]
        non_score_reward = -self.kl_ctl.value * kl
        # rewards = non_score_reward.clone().detach() + scores

        batch_rewards = []
        for i in range(self.rf_params["batch_size"]):
            end_idx = gen_len - end_token_idxs[i] - 1
            # rewards[i, end_idx:] = scores[i]
            batch_kl[i] = kl[i, :end_idx + 1].sum()
            rewards_i = rewards[i, :end_idx + 1].sum(dim=-1)
            batch_rewards.append(rewards_i / (end_idx + 1))

        batch_mean_rewards = torch.tensor(batch_rewards).mean(dim=-1)

        return rewards, batch_mean_rewards, batch_kl.mean()

    def train_minibatch(self, logprobs, rewards, predictions,doc_scores,batch_kl,accumulation_steps,retriever_reward):
        """Train one PPO minibatch"""

        loss_list = []
        pg_loss = 0
        for j in range(len(rewards)):
            loss_list.append(self.caculate_loss(logprobs[j:j + 1], rewards[j:j + 1],predictions[j:j+1]))

        if self.args.train_dpr:
            for each_doc_loss,score,reward in zip(loss_list,doc_scores,retriever_reward):
                pg_loss += each_doc_loss + (- torch.log(score) * reward)
        else:
            pg_loss = sum(loss_list)
        loss = pg_loss + self.rf_params['lamda'] * batch_kl

        self.writer.add_scalar('train/reinforce-loss', pg_loss, self.global_steps)
        self.writer.add_scalar('train/total loss', loss, self.global_steps)
        self.writer.add_scalar('train/learning rate', self.optimizer.state_dict()['param_groups'][0]['lr'], self.global_steps)
        # loss = loss / accumulation_steps
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # return train_stats

    def caculate_loss(self, old_logprobs, reward, prediction):
        """Calculate policy and value losses."""
        pad_tokens_num = counter_pad_tokens(prediction)[0]
        # gen_len = response.shape[-1] - 1
        gen_len = prediction.shape[-1] - pad_tokens_num
        lastgaelam = reward[0]
        advantages_reversed = []
        old_logprobs = old_logprobs[:, :gen_len]
        for t in reversed(range(gen_len)):
            advantages_reversed.append(lastgaelam)
            lastgaelam = self.rf_params['gamma'] * lastgaelam
        advantages = torch.tensor(advantages_reversed[::-1]).to(self.args.device)
        # returns = advantages + values
        # if self.rf_params['whiten']:
        #     advantages = whiten(advantages)
        # advantages = minmaxscaler(advantages)
        advantages = advantages.detach()

        pg_losses = -advantages * old_logprobs
        # pg_losses2 = -advantages * torch.clamp(old_logprobs,
        #                                        1.0 - self.rf_params['cliprange'],
        #                                        1.0 + self.rf_params['cliprange'])
        #
        # pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))

        # pg_loss = pg_losses.mean()
        return pg_losses.mean()

    def calculate_max_train_step(self):
        if self.args.max_steps > 0:
            return
        else:
            assert len(self.train_dataset) > 0
            len_dataset = len(self.train_dataset) 
            total_train_batch_size = self.args.per_device_train_batch_size * \
                                                        self.args.gradient_accumulation_steps * \
                                                        self.args.world_size
            num_update_steps_per_epoch = len_dataset // total_train_batch_size
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
            self.args.max_steps = max_steps


    
    def evaluate_(self,ignore_keys=-100,n_docs = 5,vanilla_qg_model_name_or_path=None,retrieve_key='question',train_dataloader=None):
        metrics = {}
        model = self._wrap_model(self.model, training=False)
        model.eval()
        vanilla_qg_model = None
        if retrieve_key != 'masked_context':
            vanilla_qg_model = AutoModelForSeq2SeqLM.from_pretrained(vanilla_qg_model_name_or_path)
            vanilla_qg_model = vanilla_qg_model.to(self.args.device)
        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.per_device_eval_batch_size, collate_fn=self.eval_collate_fn)

        # 如果是DA，就把test_dataloader换成train_dataloader
        test_metrics, test_output_data = self.caculate_metrics(data_loader=test_dataloader,is_output_file=True,prefix='test',n_docs = n_docs,vanilla_qg_model=vanilla_qg_model,retrieve_key=retrieve_key)
        self.logger.info(f'-----test:{test_metrics}')
        for k,v in test_metrics.items():
            metrics['test/'+ k] = v
        # eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.args.per_device_eval_batch_size, collate_fn=self.eval_collate_fn)
        # eval_metrics, eval_output_data = self.caculate_metrics(data_loader=eval_dataloader,n_docs = n_docs,prefix='dev',is_output_file=False,vanilla_qg_model=vanilla_qg_model,retrieve_key=retrieve_key)
        # self.logger.info(f'-----eval:{eval_metrics}')
        # for k,v in eval_metrics.items():
        #     metrics['eval/'+ k] = v
        if metrics['test/overall_bleu'] > self.best_bleu:
            current_bleu = metrics['test/top_1_bleu']
            save_dir = self.args.output_dir + '/' + str(current_bleu)
            self.save_model(save_dir)
        if self.model.rag.ctx_encoder:
            self.model.rag.ctx_encoder.save_pretrained(save_dir + '/ctx_encoder')
        self.model.retriever.generator_tokenizer.save_pretrained(save_dir + '/generator_tokenizer')
        self.model.retriever.question_encoder_tokenizer.save_pretrained(save_dir + '/question_encoder_tokenizer')
        # 保存测试集输出
        self.save_data(current_bleu,test_output_data)
        
        # tensorboard
        for k,v in metrics.items():
            self.writer.add_scalar(k, v, self.global_steps)
        self.writer.flush()
        if self.args.local_rank in [-1, 0]:
            pass
            # self.log(metrics)
            # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            # self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics


    def caculate_metrics(self, data_loader, is_output_file=False, prefix='eval',n_docs = 5,vanilla_qg_model=None,retrieve_key='question'):
        model = self.model
        device = self.args.device
        model.to(device)
        output_data = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader,desc="eval...."):
                # generate question through vanilla qg model
                if vanilla_qg_model:
                    features = self.tokenizer.batch_encode_plus(
                                    batch['contexts'],
                                    max_length=self.eval_data_args.max_combined_length,
                                    return_tensors='pt',
                                    padding='longest',
                                    truncation=True,
                                ) # highlighted context
                    outs = vanilla_qg_model.generate(
                        features['input_ids'].to(device),
                        attention_mask = features['attention_mask'].to(device),
                        max_length=self.eval_data_args.val_max_target_length,
                        num_beams=self.eval_data_args.eval_beams,
                    )
                    
                    # decode
                    generated_question = self.tokenizer.batch_decode(outs,skip_special_tokens=True)
                    if retrieve_key == 'question_skeleton':
                        if self.eval_data_args.task == 'question_generation':
                            question_skeleton = get_question_skeleton_for_qg(generated_question,batch['contexts'],self.args.use_dlc)
                        elif self.eval_data_args.task == 'paraphrase_generation':
                            question_skeleton = get_question_skeleton_for_paraphrase(generated_question,batch['contexts'],self.args.use_dlc)
                        generated_question = question_skeleton
                    # encode in dprquestionencoder
                    question_ids = self.model.retriever.question_encoder_tokenizer.batch_encode_plus(
                        generated_question,
                        max_length=128,
                        return_tensors='pt',
                        padding='longest',
                        truncation=True,
                    )
                    outs, skeleton_candidates,question_candidates,retrieved_context,doc_scores,sequences_scores,greedy_search_sequences = model.generate(
                        question_ids['input_ids'].to(device),
                        attention_mask = question_ids['attention_mask'].to(device),
                        contexts = batch['contexts'],
                        max_length=self.eval_data_args.val_max_target_length,
                        num_beams=self.eval_data_args.eval_beams,
                        do_sample=self.eval_data_args.do_sample,
                        n_docs = n_docs,
                        is_eval=True,
                        add_rl_base_reward = self.args.add_rl_base_reward,
                        retrieve_num = self.args.eval_retrieve_num,
                        filter_type = self.args.eval_filter_type,
                        use_bm25=self.args.use_bm25,
                        data_args=self.eval_data_args,
                        # num_return_sequences=5,
                    )
                else:
                    outs, skeleton_candidates,question_candidates,retrieved_context,doc_scores,sequences_scores,greedy_search_sequences = model.generate(
                        batch['input_ids'].to(device),
                        attention_mask = batch['attention_mask'].to(device),
                        contexts = batch['contexts'],
                        max_length=self.eval_data_args.val_max_target_length,
                        num_beams=self.eval_data_args.eval_beams,
                        do_sample=self.eval_data_args.do_sample,
                        n_docs = n_docs,
                        is_eval=True,
                        add_rl_base_reward = self.args.add_rl_base_reward,
                        retrieve_num = self.args.eval_retrieve_num,
                        filter_type = self.args.eval_filter_type,
                        use_bm25=self.args.use_bm25,
                        data_args=self.eval_data_args,
                        # num_return_sequences=5,
                    )
                #  
                batch_size = len(batch['input_ids'])
                k = len(outs) // len(batch['input_ids'])
                predicted_ids = outs.view((batch_size,k,-1))
                doc_scores = torch.tensor(doc_scores).to(device)
                doc_scores = doc_scores.view(-1,k)
                sequences_scores = sequences_scores.view(-1,k)

                # choose 5 out of n_docs
                # for outputs,score in zip(predicted_ids,scores):
                #     key_value = {k:v for k,v in zip(outputs,score)}
                #     key_value= sorted(key_value.items())
                #     key
                # best_idx = [0] * len(doc_scores) # 默认第一个就是概率最高的
                # best_idx = torch.argmax(torch.tensor(doc_scores),dim=1).tolist()
                best_idx = torch.argmax(sequences_scores+doc_scores,dim=1).tolist()

                labels = batch['labels']
                # decode
                ground_skeleton = [self.model.retriever.question_encoder_tokenizer.decode([id for id in ids if id != 0 and id!= 101 and id != 102],skip_special_tokens=False) for ids in batch['input_ids']]
                skeleton_candidates = [skeleton_candidates[i:i+k] for i in range(0,len(skeleton_candidates),k)]
                question_candidates = [question_candidates[i:i+k] for i in range(0,len(question_candidates),k)]
                retrieved_context = [retrieved_context[i:i+k] for i in range(0,len(retrieved_context),k)]
                prediction = [self.tokenizer.batch_decode(batch_ids, skip_special_tokens=True) for batch_ids in predicted_ids]
                best_predict = [item[i] for item,i in zip(prediction,best_idx)]
                output_data.extend([{'context':c,'ground_skeleton':o,'retrieve_skeleton':r,'retrieve_question':q,'retrieved_context':re_c,'predicted':p,'ground_truth':g,'best_predicted':b,'retrieve_score':s.tolist(),'generate_score':g_s.tolist()} for c,o,r,q,re_c,p,g,b,s,g_s in zip(batch['contexts'],ground_skeleton,skeleton_candidates,question_candidates,retrieved_context, prediction,labels,best_predict,doc_scores,sequences_scores)])

        # process result
        # predicted_ids = [[id for id in ids if id != 2 and id != 0 and id != 1] for ids in output_ids.cpu().detach().numpy()]


        # caculate metrics
        if self.eval_data_args.task == "question_generation":
            top_1_bleu,pairwise_bleu,oracle_bleu,average_bleu = calculate_bleu(output_data)
            # rouge_l = calculate_rouge(output_data,self.args.use_dlc) # top-1
            # dist_1,dist_2 = calculate_dist() # top-1
            return {
                'top_1_bleu' : top_1_bleu,
                'oracle_bleu': oracle_bleu,
                'pairwise_bleu': pairwise_bleu,
                'average_bleu':average_bleu,
                'overall_bleu': (top_1_bleu * oracle_bleu) / pairwise_bleu,
                # 'rouge_l':rouge_l,
                # 'dist_1':dist_1,
                # 'dist_2':dist_2
                }, output_data
        elif self.eval_data_args.task == "paraphrase_generation":
            top_1_bleu,self_bleu,ibleu_7,ibleu_8,ibleu_9,mean_ibleu,pairwise_bleu = calculate_bleu_for_paraphrase(output_data, prefix, 'data_qqp' in self.eval_data_args.data_dir,self.args.use_dlc)
            return {'top_1_bleu' : top_1_bleu,
                'self_bleu':self_bleu,
                'ibleu_0.7':ibleu_7,
                'ibleu_0.8':ibleu_8,
                'ibleu_0.9':ibleu_9,
                'mean_ibleu':mean_ibleu,
                'pairwise_bleu': pairwise_bleu,
                },output_data
    def save_data(self,current_bleu,output_data):
        with open(os.path.join(self.args.output_dir + '/' + str(current_bleu),'output_data.json'),'w') as fp:
            json.dump(output_data,fp,indent=2)
            # for item in output_data:
            #     json.dump(item, fp,indent=2)
            #     fp.write('\n')
