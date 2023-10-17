from transformers import RagRetriever
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from utils import caculate_jaccard_distance
from transformers.tokenization_utils_base import BatchEncoding
class MyRagRetriever(RagRetriever):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def postprocess_docs(self, filter_skeleton, contexts,input_strings, prefix, n_docs, return_tensors): # 这里contexts是原始问题对应的context而不是检索回来的context
        def cat_input_and_doc(question_skeleton, context, prefix):
            if question_skeleton.startswith('"'):
                question_skeleton = question_skeleton[1:]
            if question_skeleton.endswith('"'):
                question_skeleton = question_skeleton[:-1]
            if prefix is None:
                prefix = ""
            out = (prefix + context + ' <sep> ' + question_skeleton).replace(
                "  ", " "
            )
            return out
        k = len(filter_skeleton) // len(contexts) 
        rag_input_strings = [
            cat_input_and_doc(
                filter_skeleton[i*k + j],
                contexts[i],
                prefix,
            )
            for i in range(len(contexts))
            for j in range(k)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="longest",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states,
        prefix=None,
        n_docs=None,# 这里n_docs代表n_question
        is_eval = False,
        contexts = None,
        retrieve_num = None,
        return_tensors=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified `question_hidden_states`.
        Args:
            question_input_ids: (`List[List[int]]`) batch of input ids
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (`str`, *optional*):
                The prefix used by the generator's tokenizer.
            n_docs (`int`, *optional*):
                The number of docs retrieved per query.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        Returns: [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **context_input_ids** -- List of token ids to be fed to a model.
              [What are input IDs?](../glossary#input-ids)
            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).
              [What are attention masks?](../glossary#attention-mask)
            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """
        # if is_eval:

        prefix = prefix if prefix is not None else self.config.generator.prefix
        question_hidden_states = torch.nn.functional.normalize(question_hidden_states,p=2,dim=1)
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states.cpu().detach().to(torch.float32).numpy(),retrieve_num)
    
        # all_dataset_score_q0_top100 = []
        # all_dataset_score_q1_top100 = []
        # all_embeddings = [self.index.dataset[i]['embeddings']  for i in range(len(self.index.dataset))]
        # all_embeddings = torch.tensor(all_embeddings).to(question_hidden_states)
        # all_embeddings = torch.nn.functional.normalize(all_embeddings,p=2,dim=1)
        # scores = []
        # ids = []
        # for i in range(10):
        #     s = torch.mm(torch.nn.functional.normalize(question_hidden_states[i],p=2,dim=0).unsqueeze(0),all_embeddings.transpose(1,0))
        #     score,id_ = torch.topk(s, 100)
        #     scores.append(score)
        #     ids.append(id_)
        
        # 计算score
        retrieved_doc_embeds = torch.tensor(retrieved_doc_embeds).to(question_hidden_states)
        retrieved_scores = torch.bmm(
            question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1).view(-1,retrieve_num)

        question_input_ids = [[id for id in ids if id != 0 and id!= 101 and id!= 102]  for ids in question_input_ids.cpu().detach().numpy()]
        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=False)
        if is_eval:
            # # 使用jaccard 距离聚类
            # filter_skeleton = []
            # filter_question = []
            # filter_context = []
            # filter_score = []
            # for i in range(len(retrieved_scores)):
            #     jaccard_distance = np.ones((retrieve_num,retrieve_num))
            #     for j in range(retrieve_num):
            #         for k in range(retrieve_num):
            #             jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
            #     cluster = AgglomerativeClustering(n_clusters=n_docs,affinity='precomputed',linkage='average')
            #     clustering = cluster.fit(jaccard_distance)
            #     for j in range(n_docs):
            #         current = np.where(clustering.labels_==j)[0]
            #         idx = np.random.choice(current)
            #         filter_score.append(retrieved_scores[i][idx])
            #         filter_skeleton.append(docs[i]['question_skeleton'][idx])
            #         filter_question.append(docs[i]['question'][idx])
            #         filter_context.append(docs[i]['context'][idx])

            # 聚n_docs类
            k = retrieve_num // (n_docs - 1) # 每类k个
            filter_skeleton = []
            filter_question = []
            filter_context = []
            filter_score = []
            for i in range(len(retrieved_scores)):
                exist_skeleton = set()
                idx = 0
                exist_skeleton.add(docs[i]['question_skeleton'][idx].lower())
                filter_score.append(retrieved_scores[i][idx])
                filter_skeleton.append(docs[i]['question_skeleton'][idx])
                filter_question.append(docs[i]['question'][idx])
                filter_context.append(docs[i]['context'][idx])
                for j in range(n_docs-1):
                    order = list(range(k))
                    # order = reversed(order)
                    np.random.shuffle(order)
                    for t in order:
                        idx = j*k+t
                        if docs[i]['question_skeleton'][idx].lower() not in exist_skeleton or t == order[-1]:
                            exist_skeleton.add(docs[i]['question_skeleton'][idx])
                            filter_score.append(retrieved_scores[i][idx])
                            filter_skeleton.append(docs[i]['question_skeleton'][idx])
                            filter_question.append(docs[i]['question'][idx])
                            filter_context.append(docs[i]['context'][idx])
                            break
        else:
            filter_skeleton = []
            filter_question = []
            filter_context = []
            filter_score = []

            # 采样n_docs个，均匀间隔，
            interval = retrieve_num // (n_docs)
            for i in range(len(retrieved_scores)):
                exist_skeleton = set()
                for j in range(n_docs):
                    k = np.random.randint(j*interval,(j+1)*interval)
                    count = 0
                    while docs[i]['question_skeleton'][k].lower() in exist_skeleton and count < retrieve_num: #有耗时的风险
                        k = np.random.randint(j*k,(j+1)*k)
                        count += 1
                    exist_skeleton.add(docs[i]['question_skeleton'][k].lower())
                    filter_score.append(retrieved_scores[i][k])
                    filter_skeleton.append(docs[i]['question_skeleton'][k])
                    filter_question.append(docs[i]['question'][k])
                    filter_context.append(docs[i]['context'][k])
            
            # 任意采样
            # for i in range(len(retrieved_scores)):
            #     exist_skeleton = set()
            #     for j in range(n_docs):
            #         k = np.random.randint(0,retrieve_num)
            #         count = 0
            #         while docs[i]['question_skeleton'][k] in exist_skeleton and count < retrieve_num: #有耗时的风险
            #             k = np.random.randint(0,retrieve_num)
            #             count += 1
            #         exist_skeleton.add(docs[i]['question_skeleton'][k])
            #         filter_score.append(retrieved_scores[i][k])
            #         filter_skeleton.append(docs[i]['question_skeleton'][k])
            #         filter_question.append(docs[i]['question'][k])
            #         filter_context.append(docs[i]['context'][k])



        context_input_ids, context_attention_mask = self.postprocess_docs(
            filter_skeleton, contexts, input_strings, prefix, n_docs, return_tensors=return_tensors)
        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
            },
            tensor_type=return_tensors,
        ),filter_skeleton,filter_question,filter_context,filter_score