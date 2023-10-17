from transformers import Seq2SeqTrainer
import torch
from torch import nn
from datasets_tasks import SQuAD_DataLoader
from collator import T2TDataCollator
from tqdm import tqdm
from utils import *
import sys
sys.path.append('/mnt/workspace/gouqi/code/RQG/reward_model')
sys.path.append('/root/data/ruzhen/RQG/reward_model')
from eval_squad import get_score_from_data
import os
import json
class QATrainer(Seq2SeqTrainer):
    def __init__(self,eval_data_args,**kwargs):
        self.eval_data_args = eval_data_args
        test_dataset = kwargs.pop('test_dataset')
        self.test_dataset = test_dataset
        self.best_f1 = -float('inf')
        super().__init__(**kwargs)
    def caculate_metrics(self, data_loader, is_output_file=False, prefix='eval'):
        model = self.model
        device = self.args.device
        model.to(device)
        output_data = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                outs = model.generate(
                    input_ids=batch['input_ids'].to(device), 
                    attention_mask=batch['attention_mask'].to(device),
                    num_beams = self.eval_data_args.eval_beams,
                    max_length = self.eval_data_args.val_max_target_length,
                    do_sample = self.eval_data_args.do_sample,
                    top_k = self.eval_data_args.top_k,
                    top_p = self.eval_data_args.top_p,
                    num_return_sequences = self.eval_data_args.num_return_sequences,
                    output_scores = True,
                    return_dict_in_generate=True
                    # length_penalty=length_penalty,
                )
                # decode
                context = [self.tokenizer.decode([id for id in ids if id != 0 and id!= 1 and id != 2],skip_special_tokens=False) for ids in batch['input_ids'].cpu().detach().numpy()]
                prediction = self.tokenizer.batch_decode(outs.sequences, skip_special_tokens=True)
                batch['labels'][batch['labels'] == -100] = 0
                labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                output_data.extend([{'context':c,'predicted':p,'ground_truth':g} for c,p,g in zip(context,prediction,labels)])
        # caculate metrics
        em,f1 = get_score_from_data(output_data)
        
        if is_output_file:
            with open(os.path.join(self.args.output_dir,'output_data.txt'),'w') as fp:
                for item in output_data:
                    fp.writelines('context:\t' + item['context'])
                    fp.write('\n')
                    fp.writelines('ground_truth:\t' + item['ground_truth'])
                    fp.write('\n')
                    fp.writelines('generated:')
                    fp.write('\n')
                    fp.writelines(item['predicted'])
                    fp.write('\n')
        return {'em': em,'f1': f1}
    def evaluate(self,ignore_keys=-100):
        metrics = {}
        model = self._wrap_model(self.model, training=False)
        model.eval()
        dev_collator = T2TDataCollator(
            tokenizer = self.tokenizer,
            model = self.model,
            max_length=self.eval_data_args.max_source_length,
            padding=True,
        )
        eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.args.per_device_eval_batch_size, collate_fn=dev_collator)
        eval_metrics = self.caculate_metrics(data_loader=eval_dataloader)
        self.log(eval_metrics)
        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.per_device_eval_batch_size, collate_fn=dev_collator)
        test_metrics = self.caculate_metrics(data_loader=test_dataloader,is_output_file=True,prefix='test')
        for k,v in eval_metrics.items():
            metrics['eval_'+ k] = v
        for k,v in test_metrics.items():
            metrics['test_'+ k] = v
        self.log(test_metrics)
        # test
        if metrics['eval_f1'] > self.best_f1:
            self.save_model(self.args.output_dir)
            self.best_f1 = metrics['eval_f1']
            self.save_model(self.args.output_dir + '/' + str(self.best_f1))
        
        # val_loss = self.compute_loss(model, inputs)

        # if self.eval_args.datasets == 'conll03' or self.eval_args.datasets == 'ace2005_ner':
        #     if metrics['entity_f1'] > self.best_f1:
        #         self.save_model(self.args.output_dir)
        #         self.best_f1 = metrics['entity_f1']
        #         self.save_model(self.args.output_dir + '/' + str(self.best_f1))
        # else:
        #     if metrics['relation_f1'] > self.best_f1:
        #         self.save_model(self.args.output_dir)
        #         self.best_f1 = metrics['relation_f1']
        #         self.save_model(self.args.output_dir + '/' + str(self.best_f1))

        # metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        if self.args.local_rank in [-1, 0]:
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.data_collator.label_pad_token_id)
        loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
#     def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
#         if self.control.should_log:
#             logs = {}
#             tr_loss_scalar = tr_loss.item()
#             # reset tr_loss to zero
#             tr_loss -= tr_loss
#             # print(f'trainer.py, 1201：{tr_loss_scalar}')
#             logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
#             logs["learning_rate"] = self._get_learning_rate()
#             self.log(logs)

#             self._total_loss_scalar += tr_loss_scalar
#             self._globalstep_last_logged = self.state.global_step

#         metrics = None
#         if self.control.should_evaluate:
#             metrics = self.evaluate()
#             if self.args.local_rank in [-1, 0]:
#                 print("11111111111")
#                 self._report_to_hp_search(trial, epoch, metrics)

#         if self.control.should_save:
#             self._save_checkpoint(model, trial, metrics=metrics)
#             self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _report_to_hp_search(self, trial, step, metrics):
        if self.hp_search_backend is None or trial is None:
            return
        self.objective = self.compute_objective(metrics.copy())
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            import optuna

            trial.report(self.objective, step)
            if trial.should_prune():
                self.callback_handler.on_train_end(self.args, self.state, self.control)
                raise optuna.TrialPruned()
        elif self.hp_search_backend == HPSearchBackend.RAY:
            from ray import tune

            if self.control.should_save:
                self._tune_save_checkpoint()
            tune.report(objective=self.objective, **metrics)

