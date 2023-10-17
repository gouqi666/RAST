import transformers
from dataclasses import dataclass, field
from typing  import Any, Dict, List, Optional, Union
from typing_extensions import Literal
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str = field(default='t5',metadata={"help": "One of 't5', 'bart'"})
    question_encoder_name_or_path : str = field(default="facebook/dpr-question_encoder-single-nq-base") #     'facebook/dpr-question_encoder-single-nq-base'
    generator_name_or_path: str = field(default='output/SQuAD_1.1_split1/QG/experiments_with_skeleton/t5-base-GPUNums1-len128-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b32-top_k30-top_p0.9-temperature1.0')

    reward_model_name_or_path : str = field(default='output/SQuAD_1.1_split1/QA/experiments/t5-base-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-64-lr0.0002-b32-beam_search5') # qa reward
    sm_reward_model_name_or_path : str = field(default='output/SQuAD_1.1_split1_pre/SM/experiments/bert-base-cased-GPUNums1-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-128-lr1e-05-b128/0.9825465083122253') # style reward
    vanilla_qg_model_name_or_path: str = field(default='/home/gq/RAST/output/SQuAD_1.1_split1/QG/experiments/t5-base-GPUNums1-len128-fp16False--warm0.1--warmSteps0--weightDecay0.1-64-lr5e-05-b64-top_k50-top_p0.9-temperature1.0/20.827964423502124')
    eval_model_name_or_path: str = field(default='None')
    model_name_or_path: str = field(
        default= 'facebook/rag-sequence-base', #'facebook/bart-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default='facebook/rag-sequence-base', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='facebook/rag-sequence-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
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

    dataset_path: str = field(default='/home/gq/RAST/data/SQuAD_1.1_split1/processed/my_knowledge_dataset_question_skeleton')

    index_path: str = field(default="/home/gq/RAST/data/SQuAD_1.1_split1/processed/my_knowledge_dataset_hnsw_index_question_skeleton.faiss")

    data_dir: str = field(default='/home/gq/RAST/data/SQuAD_1.1_split1/processed')

    is_use_skeleton: bool = field(default=True)
    train_data_dir: str = field(
        default='--',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    valid_data_dir: str = field(
        default='--',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    test_data_dir: str = field(
        default='--',
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="question_generation",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum length of retrieve_key "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_combined_length: Optional[int] = field(
        default=128,
        metadata={
            "help":("max length of combined context and skeleton")
        }
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=128,
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
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for test target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    n_train: Optional[float] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[float] = field(default=0.01, metadata={"help": "# validation examples. -1 means use all."})
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
    num_return_sequences:int = field(default=5)
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    """
    Arguments for the Trainer.
    """
    use_bm25:bool = field(default=False)
    eval_steps: int = field(default=2000)
    train_filter_type: str = field(default='v3')
    eval_filter_type: str = field(
        default='v0',
        # metadata={"choices":['random','cluster','']}
    )
    set_context_encoder_for_training: bool = field(default=False)
    sm_reward_type: str = field(
        default='jaccard'
    )
    train_retrieve_num : int = field(default=100)
    eval_retrieve_num: int = field(default=500)
    train_sample : bool = field(default=True)
    reward_batch_normalization: bool = field(default=True)
    retrieve_key: str = field(default='question_skeleton')
    ctx_retrieve_key: str = field(default='question_skeleton')
    reward_type: str = field(
        default='qa',
        metadata={"help": "reward_type:"}
    )
    add_rl_base_reward: bool = field(default=True)
    add_sm_base_reward: bool = field(default=True)
    use_dlc:bool = field(default=False)
    n_questions: int = field(default=3)
    eval_n_questions: int = field(default=5)
    train_dpr:bool = field(default=True)
    no_cuda:bool = field(default=False)
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
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
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    seed: int = field(default=42)
    learning_rate: float = field(default=5e-6, metadata={"help": "The initial learning rate for AdamW."})
    dpr_learning_rate: float = field(default=1e-6)
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs : int = field(default = 10)
    warmup_ratio: float = field(
        default=0.2, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_strategy:  str = field(
        default="no",
        metadata={"help": "The logging strategy to use."},
    )
    report_to: Optional[List[str]] = field(
        default='tensorboard', metadata={"help": "The list of integrations to report the results and logs to."}
    )
    logging_dir: Optional[str] = field(default='run', metadata={"help": "Tensorboard log dir."})
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    predict_with_generate : bool = field(
        default=True, metadata={"help":"predict_with_generate"}
    )
    optim: str = field(
        default='adamw_hf',
        metadata={"help": "The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor."}
    )
    output_dir: str = field(
        default='RQG/output/SQuAD_1.1_split1/RAG/experiments_test',
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
    do_train: bool = field(default=False, metadata={"help": "train"})
    do_eval: bool = field(default=True, metadata={"help": "eval"})
    do_predict: Optional[bool] = field(default=False, metadata={"help": "eval"})



@dataclass
class ReinforcementArguments:
    lamda: float = field(default=0.6)
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2"}
    )
    target: Optional[float] = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control, default: 6.0"}
    )
    horizon: Optional[int] = field(
        default=10000,
        metadata={"help": "Horizon for adaptive KL control, default: 10000"}
    )
    gamma: Optional[float] = field(
        default=1.0,
        metadata={"help": "Gamma parameter for advantage calculation, default: 1."}
    )
    whiten:bool = field(default=False)
    lam: Optional[float] = field(
        default=0.95,
        metadata={"help": "Lambda parameter for advantage calcualation, default: 0.95"}
    )
    cliprange: Optional[float] = field(
        default=0.2,
        metadata={"help": "Range for clipping in PPO policy gradient loss, default: 0.2"}
    )
    cliprange_value: Optional[float] = field(
        default=0.2,
        metadata={"help": "Range for clipping values in loss calculation, default: 0.2"}
    )
    vf_coef: Optional[float] = field(
        default=0.1,
        metadata={"help": "Scaling factor for value loss, default: 0.1"}
    )
    adap_kl_ctrl: Optional[bool] = field(
        default=True,
        metadata={"help": "Use adaptive KL control, otherwise linear, default: True"}
    )

    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "batch size, default: 8"}
    )

    forward_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "batch size = forward batch size * world size"
                          "When not using parallel training, it's equal to batch size, default: 8"}
    )

    ppo_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "Number of optimisation epochs per batch of samples, default: 4"}
    )
    sm_coef:float = field(default=0.6)