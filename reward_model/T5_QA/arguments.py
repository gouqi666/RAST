import transformers
from dataclasses import dataclass, field
from typing  import Any, Dict, List, Optional, Union
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str = field(default='t5',metadata={"help": "One of 't5', 'bart'"})

    model_name_or_path: str = field(
        default= 't5-base',#'facebook/bart-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default='t5-base', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='t5-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    add_extra_data: bool = field(default=False)
    train_data_dir: str = field(
        default='RQG/data/SQuAD_1.1_split2/processed/train.jsonl',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    valid_data_dir: str = field(
        default='RQG/data/SQuAD_1.1_split2/processed/dev.jsonl',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    test_data_dir: str = field(
        default='RQG/data/SQuAD_1.1_split2/processed/test.jsonl',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    test_max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "The maximum total sequence length for test target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    n_train: Optional[float] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[float] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[float] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=5, metadata={"help": "# num_beams to use for evaluation."})
    top_p: float = field(
        default=1.0,metadata={"help":"top_p"}
    )
    top_k: int = field(
        default=50,metadata={"help":"top_p"}
    )
    do_sample:bool = field(default=False)
    num_return_sequences:int = field(default=1)
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    """
    Arguments for the Trainer.
    """
    no_cuda:bool = field(default=False)
    save_strategy: str = field(
        default="no",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs : int = field(default = 5)
    warmup_ratio: float = field(
        default=0.2, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_strategy:  str = field(
        default="epoch",
        metadata={"help": "The logging strategy to use."},
    )
    report_to: Optional[List[str]] = field(
        default='tensorboard', metadata={"help": "The list of integrations to report the results and logs to."}
    )
    logging_dir: Optional[str] = field(default='run', metadata={"help": "Tensorboard log dir."})
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )
    predict_with_generate : bool = field(
        default=True, metadata={"help":"predict_with_generate"}
    )
    optim: str = field(
        default='adamw_hf',
        metadata={"help": "The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor."}
    )
    reward_type: str = field(
        default='f1',
        metadata={"help": "reward_type"}
    )

    output_dir: str = field(
        default='RQG/output/SQuAD_1.1_split2/QA/experiments_test',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )

    zero_shot: bool = field(
        default=False,
        metadata={"help": "Zero-shot setting"}
    )

    # per_gpu_train_batch_size: Optional[int] = field(
    #     default=1,
    #     metadata={
    #         "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
    #         "Batch size per GPU/TPU core/CPU for training."
    #     },
    # )
    # per_gpu_eval_batch_size: Optional[int] = field(
    #     default=1,
    #     metadata={
    #         "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
    #         "Batch size per GPU/TPU core/CPU for evaluation."
    #     },
    # )
    # per_device_eval_batch_size: int = field(
    #     default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    # )
    #
    # per_device_train_batch_size: int = field(
    #     default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    # )

    eval_epoch_interval: int = field(
        default=1,
        metadata={"help": "evaluate while training for every interval epochs"}
    )

    eval_dataset: str = field(
        default='dev',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )
    do_train: bool = field(default=True, metadata={"help": "train"})
    do_eval: bool = field(default=True, metadata={"help": "eval"})
    do_predict: Optional[bool] = field(default=False, metadata={"help": "eval"})


