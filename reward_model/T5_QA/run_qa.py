#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
import torch
import transformers
from collator import T2TDataCollator
from datasets_tasks import SQuAD_QA_Dataset
from arguments import ModelArguments, DataTrainingArguments,TrainingArguments
import numpy as np
# from seq2seq_training_args import Seq2SeqTrainingArguments
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    # DefaultDataCollator,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    set_seed,
    BartForConditionalGeneration
)

from QATrainer import QATrainer

# from QG_trainer import QGTrainer
# from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode

# from utils import (
#     assert_all_frozen,
#     build_compute_metrics_fn,
#     check_output_dir,
#     freeze_embeds,
#     freeze_params,
#     lmap,
#     save_json,
#     use_task_specific_params,
#     write_txt_file,
# )


def handle_metrics(split, metrics, output_dir,logger):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    # save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))

            

def main():
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args,training_args = parser.parse_args_into_dataclasses()
    
    
    training_args.generation_max_length = 128
    training_args.generation_num_beams = 5
    # check_output_dir(training_args)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    output_dir = training_args.output_dir
    output_dir = os.path.join(
        output_dir,
        f'{model_args.model_name_or_path.split("/")[-1]}'
        f'-GPUNums{torch.cuda.device_count()}' # 
        f'-len{data_args.max_source_length}'
        f'-fp16{training_args.fp16}'
        f'--warm{training_args.warmup_ratio}'
        f'--warmSteps{training_args.warmup_steps}'
        f'--weightDecay{training_args.weight_decay}'
    )
    output_dir += f'-{data_args.max_target_length}'

    # if training_args.learning_rate != 5e-4:
    output_dir += f'-lr{training_args.learning_rate}'

    output_dir += f'-b{training_args.per_device_train_batch_size}'
    # decode method
    if data_args.eval_beams == 1 and data_args.do_sample == False:
        output_dir += f'-greedy'
    elif data_args.eval_beams > 1 and data_args.do_sample == False:
        output_dir += f'-beam_search{data_args.eval_beams}'
    elif data_args.do_sample == True:
        if data_args.top_k != 0:
            output_dir += f'-top_k{data_args.top_k}'
        if data_args.top_p != 1.0:
            output_dir += f'-top_p{data_args.top_p}'

    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir + '/' + 'tensorboard/' 


    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # Setup logging
    filename= training_args.output_dir + '/' + __name__ + '.log'
    logging.basicConfig(
        filename= filename,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
        filemode='w+'
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & voca
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
    )
    tokenizer.add_tokens(['<HL>','<sep>','[mask]'], special_tokens=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    # use task specific params
    # use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.tgt_lang)

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())


    # Get datasets
    train_dataset = (SQuAD_QA_Dataset(
            tokenizer,
            data_dir=data_args.train_data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
            extra_data = data_args.add_extra_data
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (SQuAD_QA_Dataset(
            tokenizer,
            data_dir=data_args.valid_data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            seed = training_args.seed,
            prefix=model.config.prefix or "",
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (SQuAD_QA_Dataset(
            tokenizer,
            data_dir=data_args.test_data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            seed = training_args.seed,
            prefix=model.config.prefix or "",
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    # train_loader = SQuAD_DataLoader(
    #     train_dataset,
    #     tokenizer,
    #     batch_size=2,
    #     max_target_length=data_args.val_max_target_length,
    #     max_source_length=data_args.max_source_length,
    # )
    
    # for batch in train_loader:
    #     print(batch)
    #     input_ids, attention_mask,qry_id = batch
    #     output = model(input_ids,attention_mask)
    #     print(ouput)
    # Initialize our Trainer
    # compute_metrics_fn = (
    #     build_compute_metrics_fn(data_args.task, tokenizer) if training_args.predict_with_generate else None
    # )
    def compute_metrics(eval_pred):
        logger.info(eval_pred)
        predictions,label = eval_pred
    if isinstance(model, BartForConditionalGeneration):
        model_args.model_type = 'bart'
    else:
        model_args.model_type = 't5'
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model = model,
        max_length=data_args.max_source_length,
        padding=True,
    )
    trainer =  QATrainer(
        model=model,
        args=training_args,
        eval_data_args=data_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator, 
        # compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        # trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir,logger)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            # tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        logger.info(metrics)
    #     metrics["val_n_objs"] = data_args.n_val
    #     # metrics["val_loss"] = round(metrics["val_loss"], 4)

    #     if trainer.is_world_process_zero(): 

    #         handle_metrics("val", metrics, training_args.output_dir,logger)
    #         all_metrics.update(metrics)

#     if training_args.do_predict:
#         logger.info("*** Predict ***")
#         test_output = trainer.predict(test_dataset=eval_dataset, metric_key_prefix="test")
#         metrics = test_output.metrics
#         metrics["test_n_objs"] = data_args.n_test

#         if trainer.is_world_process_zero():
#             # metrics["test_loss"] = round(metrics["test_loss"], 4)
#             handle_metrics("test", metrics, training_args.output_dir,logger)
#             all_metrics.update(metrics)

#             if training_args.predict_with_generate:
#                 test_preds = tokenizer.batch_decode(
#                     test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
#                 )
#                 # test_preds = lmap(str.strip, test_preds)
#                 # write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))

#     if trainer.is_world_process_zero():
#         # save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

#         return all_metrics

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()