from transformers import RagModel,RagSequenceForGeneration,BertModel,BertForSequenceClassification,DPRQuestionEncoder
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from transformers.utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.modeling_outputs import ModelOutput
import torch
from torch import nn
_CONFIG_FOR_DOC = "RagConfig"
import sys
sys.path.append('/root/data/ruzhen/RQG/rag')
print(sys.path)
from rag.ragModel import MyRagModel


RAG_FORWARD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [`RagConfig`], used to initialize the model, specifies
            which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
            obtain the indices.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*)
            Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
            *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs *
            sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
            generator's encoder.

            Used by the ([`RagModel`]) model during decoding.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for generation tasks. `None` by default, construct as per instructions for the generator model
            you're using with your RAG instance.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,  target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        past_key_values (`tuple(tuple(torch.FloatTensor))`):
            Tuple consists of two elements: `encoder_outputs` of the RAG model (see `encoder_outputs`) and
            `past_key_values` of the underlying generator. Can be used to speed up decoding. `past_key_values` are used
            in the ([`RagTokenForGeneration`]) model during decoding.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
            has to be provided to the forward pass. `doc_scores` can be computed via
            `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.

            If the model has is not initialized with a `retriever` ``context_input_ids` has to be provided to the
            forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`]. context_attention_mask
            (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*,
            returned when *output_retrieved=True*): Attention mask post-processed from the retrieved documents and the
            question encoder `input_ids` by the retriever.

            If the model has is not initialized with a `retriever` `context_attention_mask` has to be provided to the
            forward pass. `context_attention_mask` are returned by [`~RagRetriever.__call__`].
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_retrieved(`bool`, *optional*):
            Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask`. See returned tensors for more detail.
        n_docs (`int`, *optional*, defaults to `config.n_docs``)
            Number of documents to retrieve and/or number of documents for which to generate an answer.
"""


@dataclass
class RetrievAugLMOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class MyRagSequenceForGeneration(RagSequenceForGeneration):
    def __init__(self,
        config = None,
        question_encoder= None,
        generator  = None,
        retriever = None,
        **kwargs,
    ):
        super().__init__(config,question_encoder,generator,retriever)

        # instantiate model
        self.rag = MyRagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    # @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     decoder_input_ids: Optional[torch.LongTensor] = None,
    #     decoder_attention_mask: Optional[torch.BoolTensor] = None,
    #     past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     context_input_ids: Optional[torch.LongTensor] = None,
    #     context_attention_mask: Optional[torch.LongTensor] = None,
    #     doc_scores: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_retrieved: Optional[bool] = None,
    #     exclude_bos_score: Optional[bool] = None,
    #     reduce_loss: Optional[bool] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     n_docs: Optional[int] = None,
    #     **kwargs  # needs kwargs for generation
    # ) -> RetrievAugLMMarginOutput:
    #     r"""
    #     exclude_bos_score (`bool`, *optional*):
    #         Only relevant if `labels` is passed. If `True`, the score of the BOS token is disregarded when computing
    #         the loss.
    #     reduce_loss (`bool`, *optional*):
    #         Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
    #         operation.
    #     kwargs (`Dict[str, any]`, optional, defaults to *{}*):
    #          Legacy dictionary, which is required so that model can use *generate()* function.
    #     Returns:
    #     Example:
    #     ```python
    #     >>> from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    #     >>> import torch
    #     >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    #     >>> retriever = RagRetriever.from_pretrained(
    #     ...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
    #     ... )
    #     >>> # initialize with RagRetriever to do everything in one forward call
    #     >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    #     >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
    #     >>> with tokenizer.as_target_tokenizer():
    #     ...     targets = tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
    #     >>> input_ids = inputs["input_ids"]
    #     >>> labels = targets["input_ids"]
    #     >>> outputs = model(input_ids=input_ids, labels=labels)
    #     >>> # or use retriever separately
    #     >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
    #     >>> # 1. Encode
    #     >>> question_hidden_states = model.question_encoder(input_ids)[0]
    #     >>> # 2. Retrieve
    #     >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
    #     >>> doc_scores = torch.bmm(
    #     ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    #     >>> ).squeeze(1)
    #     >>> # 3. Forward to generator
    #     >>> outputs = model(
    #     ...     context_input_ids=docs_dict["context_input_ids"],
    #     ...     context_attention_mask=docs_dict["context_attention_mask"],
    #     ...     doc_scores=doc_scores,
    #     ...     decoder_input_ids=labels,
    #     ... )
    #     ```"""
    #     n_docs = n_docs if n_docs is not None else self.config.n_docs
    #     exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
    #     reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

    #     if labels is not None:
    #         if decoder_input_ids is None:
    #             decoder_input_ids = labels
    #         use_cache = False

    #     outputs = self.rag(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         encoder_outputs=encoder_outputs,
    #         decoder_input_ids=decoder_input_ids,
    #         decoder_attention_mask=decoder_attention_mask,
    #         context_input_ids=context_input_ids,
    #         context_attention_mask=context_attention_mask,
    #         doc_scores=doc_scores,
    #         past_key_values=past_key_values,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         output_retrieved=output_retrieved,
    #         n_docs=n_docs,
    #     )

    #     loss = None
    #     if labels is not None:
    #         loss = self.get_nll(
    #             outputs.logits,
    #             outputs.doc_scores,
    #             decoder_input_ids,
    #             reduce_loss=reduce_loss,
    #             epsilon=self.config.label_smoothing,
    #             exclude_bos_score=exclude_bos_score,
    #             n_docs=n_docs,
    #         )

    #     return RetrievAugLMMarginOutput(
    #         loss=loss,
    #         logits=outputs.logits,
    #         doc_scores=outputs.doc_scores,
    #         past_key_values=outputs.past_key_values,
    #         context_input_ids=outputs.context_input_ids,
    #         context_attention_mask=outputs.context_attention_mask,
    #         retrieved_doc_embeds=outputs.retrieved_doc_embeds,
    #         retrieved_doc_ids=outputs.retrieved_doc_ids,
    #         question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
    #         question_enc_hidden_states=outputs.question_enc_hidden_states,
    #         question_enc_attentions=outputs.question_enc_attentions,
    #         generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
    #         generator_enc_hidden_states=outputs.generator_enc_hidden_states,
    #         generator_enc_attentions=outputs.generator_enc_attentions,
    #         generator_dec_hidden_states=outputs.generator_dec_hidden_states,
    #         generator_dec_attentions=outputs.generator_dec_attentions,
    #         generator_cross_attentions=outputs.generator_cross_attentions,
    #     )
        
    # @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        contexts: Optional[List] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        do_deduplication: Optional[bool] = None,  # defaults to True
        num_return_sequences: Optional[int] = None,  # defaults to 1
        num_beams: Optional[int] = None,  # defaults to 1
        n_docs: Optional[int] = None,
        is_eval=False,
        add_rl_base_reward:Optional[bool] = None,
        retrieve_num = None,
        filter_type = None,
        use_bm25 = None,
        data_args = None,
        **model_kwargs
    ) -> torch.LongTensor:
        """
        Implements RAG sequence "thorough" decoding. Read the [`~generation_utils.GenerationMixin.generate`]`
        documentation for more information on how to set other generate input parameters.
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.
                If the model is not initialized with a `retriever` or `input_ids` is not given, `context_input_ids` and
                `context_attention_mask` have to be provided to the forward pass. They are returned by
                [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.
                If the model is not initialized with a `retriever` or `input_ids` is not given, `doc_scores` has to be
                provided to the forward pass. `doc_scores` are returned by [`~RagRetriever.__call__`].
            do_deduplication (`bool`, *optional*):
                Whether or not to deduplicate the generations from different context documents for a given input. Has
                to be set to `False` if used while training with distributed backend.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation_utils.GenerationMixin.generate`]`
                function, where we set `num_return_sequences` to `num_beams`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            kwargs:
                Additional kwargs will be passed to [`~generation_utils.GenerationMixin.generate`].
        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        assert (
            input_ids is not None or context_input_ids is not None
        ), " At least one of input_ids or context_input_ids must be given"

        if self.retriever is not None:
        ##
            # question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask).pooler_output

            # device = torch.device('cuda:0')
            # question_input_ids = [[id for id in ids if id != 0 and id!= 101 and id!= 102]  for ids in input_ids.cpu().detach().numpy()]
            # input_strings = self.retriever.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=False)
            # question_encoder = BertModel.from_pretrained('bert-base-cased').to(device)
            # question_encoder.resize_token_embeddings(len(self.retriever.question_encoder_tokenizer))
            # result = self.retriever.question_encoder_tokenizer(input_strings,max_length=128,padding='max_length',truncation=True)
            # test_input_ids = torch.tensor(result['input_ids']).to(device)
            # attention_mask = torch.tensor(result['attention_mask']).to(device)
            # outputs = question_encoder(input_ids, attention_mask=attention_mask).pooler_output
        ##
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask).pooler_output
            retriever_outputs,skeleton_candidates,question_candidates,retrieved_context,retrieved_scores = self.retriever(
                input_ids,
                question_hidden_states,
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                contexts = contexts,
                return_tensors="pt",
                is_eval = is_eval,
                retrieve_num = retrieve_num,
                filter_type= filter_type,
                use_bm25 = use_bm25,
                data_args=data_args,
            )
            if self.rag.context_encoder_training and not is_eval: # train context encoder
                new_input = skeleton_candidates if self.rag.retrieve_key == 'question_keleton' else question_candidates if self.rag.retrieve_key =='question' else retrieved_context
                result = self.retriever.ctx_encoder_tokenizer(new_input,max_length=128,padding='max_length',truncation=True)
                retrieve_doc_input_ids = torch.tensor(result['input_ids']).to(question_hidden_states.device)
                retrieve_doc_attention_mask = torch.tensor(result['attention_mask']).to(question_hidden_states.device)
                ctx_encoder = self.rag.ctx_encoder.to(question_hidden_states.device)
                retrieved_doc_embeds = ctx_encoder(
                    retrieve_doc_input_ids, attention_mask=retrieve_doc_attention_mask, return_dict=True
                ).pooler_output
                
                retrieved_doc_embeds = retrieved_doc_embeds.view(question_hidden_states.shape[0],-1,768)

                retrieved_scores = torch.bmm(
                    question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)
                retrieved_scores = torch.nn.functional.softmax(retrieved_scores,dim=-1).view(-1)
        context_input_ids, context_attention_mask = (
            retriever_outputs["context_input_ids"],
            retriever_outputs["context_attention_mask"],
        )
        # set to correct device
        context_input_ids = context_input_ids.to(input_ids)
        context_attention_mask = context_attention_mask.to(input_ids)
        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = 1
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["output_scores"] = True
        model_kwargs["return_dict_in_generate"] = True
        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs

        output = self.generator.generate(
                context_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
        output_sequences = output.sequences
        sequence_scores = output.sequences_scores if hasattr(output,'sequences_scores') else None
        # output_sequences = [output_sequences[i * n_docs : (i + 1) * n_docs] for i in range(batch_size)]
        # retrieve_candidates = [retrieve_candidates[i * n_docs : (i + 1) * n_docs] for i in range(batch_size)]
        greedy_search_sequences = None
        if add_rl_base_reward:
            greedy_search_sequences = self.generator.generate(
                context_input_ids,
                attention_mask = context_attention_mask,
                max_length = model_kwargs['max_length'],
                num_beams = 1,
                do_sample = False
            ) 
        return output_sequences,skeleton_candidates,question_candidates,retrieved_context,retrieved_scores,sequence_scores,greedy_search_sequences


        #     # then, run model forwards to get nll scores:
        #     if input_ids is not None:
        #         new_input_ids = input_ids[index : index + 1].repeat(num_candidates, 1)
        #         outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
        #     else:  # input_ids is None, need context_input_ids/mask and doc_scores
        #         assert (
        #             context_attention_mask is not None
        #         ), "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
        #         assert (
        #             doc_scores is not None
        #         ), "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."

        #         individual_input_ids = generator_input_ids.repeat(
        #             num_candidates, 1
        #         )  # (num_candidates*n_docs, max_len)

        #         individual_attention_mask = context_attention_mask[index * n_docs : (index + 1) * n_docs]
        #         individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)

        #         individual_doc_scores = doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
        #         individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]

        #         outputs = self(
        #             context_input_ids=individual_input_ids,
        #             context_attention_mask=individual_attention_mask,
        #             doc_scores=individual_doc_scores,
        #             labels=output_sequences,
        #             exclude_bos_score=True,
        #         )

        #     top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

        #     # add hypothesis
        #     hypos.append(output_sequences[top_cand_inds])

        # return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)