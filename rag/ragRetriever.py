from transformers import RagRetriever
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import heapq
from sklearn.cluster import AgglomerativeClustering
from utils import caculate_jaccard_distance
from transformers.tokenization_utils_base import BatchEncoding
class MyRagRetriever(RagRetriever):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.bm25 = None
    def postprocess_docs(self, filter_skeleton, contexts,prefix, n_docs, return_tensors,data_args): # 这里contexts是原始问题对应的context而不是检索回来的context
        def cat_input_and_doc(question_skeleton, context, prefix,data_args):
            if question_skeleton:
                if question_skeleton.startswith('"'):
                    question_skeleton = question_skeleton[1:]
                if question_skeleton.endswith('"'):
                    question_skeleton = question_skeleton[:-1]
                if prefix is None:
                    prefix = ""
                if 'newsqa' in data_args.data_dir:
                    out = (prefix + question_skeleton + ' <sep> ' + context ).replace(
                        "  ", " "
                    )
                else:
                    out = (prefix + context + ' <sep> ' + question_skeleton).replace(
                        "  ", " "
                    )
            else:
                out = context
            return out
        k = len(filter_skeleton) // len(contexts) 
        rag_input_strings = [
            cat_input_and_doc(
                filter_skeleton[i*k + j],
                contexts[i],
                prefix,
                data_args
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
        filter_type=None,
        use_bm25 = None,
        data_args=None,
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
        if use_bm25:
            question_input_ids = [[id for id in ids if id != 0 and id!= 101 and id!= 102]  for ids in question_input_ids.cpu().detach().numpy()]
            input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=False)
            get_query_score = [self.bm25.get_scores(item.replace('[MASK]','').strip().lower().split()) for item in input_strings]
            topk_indexs = [heapq.nlargest(retrieve_num,range(len(each_data)),each_data.__getitem__) for each_data in get_query_score]
            docs = [self.index.dataset[idx] for idx in topk_indexs] 
            retrieved_scores = [[score[i] for i in idx] for idx,score in zip(topk_indexs,get_query_score)]
            retrieved_scores = torch.tensor(retrieved_scores) # 与dpr retriever保持一致
        else:
            retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states.cpu().detach().to(torch.float32).numpy(),retrieve_num)
        
            # 计算score
            retrieved_doc_embeds = torch.tensor(retrieved_doc_embeds).to(question_hidden_states)
            retrieved_scores = torch.bmm(
                question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
            ).squeeze(1).view(-1,retrieve_num)
        # retrieved_scores = torch.nn.functional.softmax(retrieved_scores,dim=-1)
        filter_skeleton = []
        filter_question = []
        filter_context = []
        filter_score = []
        if is_eval:
            if filter_type == 'v0': # with null skeleton
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    for j in range(retrieve_num):
                        if docs[i]['question_skeleton'][j] not in exist_skeleton:
                            exist_skeleton.add(docs[i]['question_skeleton'][j])
                            filter_score.append(retrieved_scores[i][j])
                            filter_skeleton.append(docs[i]['question_skeleton'][j])
                            filter_question.append(docs[i]['question'][j])
                            filter_context.append(docs[i]['context'][j])
                        if len(exist_skeleton) >= n_docs:
                            break
            if filter_type == 'v0_1': # with null skeleton
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    filter_score.append(torch.tensor(1))
                    filter_skeleton.append(None)
                    filter_question.append(None)
                    filter_context.append(None)
                    for j in range(retrieve_num):
                        if docs[i]['question_skeleton'][j] not in exist_skeleton:
                            exist_skeleton.add(docs[i]['question_skeleton'][j])
                            filter_score.append(retrieved_scores[i][j])
                            filter_skeleton.append(docs[i]['question_skeleton'][j])
                            filter_question.append(docs[i]['question'][j])
                            filter_context.append(docs[i]['context'][j])
                        if len(exist_skeleton) >= n_docs-1:
                            break
            if filter_type == 'v0_2': #
                question_input_ids = [[id for id in ids if id != 0 and id != 101 and id != 102] for ids in
                                      question_input_ids.cpu().detach().numpy()]
                input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids,
                                                                             skip_special_tokens=False)
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    filter_score.append(torch.tensor(1))
                    filter_skeleton.append(input_strings[i])
                    filter_question.append(None)
                    filter_context.append(None)
                    for j in range(retrieve_num):
                        if docs[i]['question_skeleton'][j] not in exist_skeleton:
                            exist_skeleton.add(docs[i]['question_skeleton'][j])
                            filter_score.append(retrieved_scores[i][j])
                            filter_skeleton.append(docs[i]['question_skeleton'][j])
                            filter_question.append(docs[i]['question'][j])
                            filter_context.append(docs[i]['context'][j])
                        if len(exist_skeleton) >= n_docs-1:
                            break
            if filter_type == 'v1': # with null skeleton
            #  按分数聚n_docs类
                k = retrieve_num // (n_docs) # 每类k个
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    # exist_skeleton.add(None)
                    # filter_score.append(torch.tensor(1))
                    # filter_skeleton.append(None)
                    # filter_question.append(None)
                    # filter_context.append(None)
                    for j in range(n_docs):
                        order = list(range(k))
                        # order = reversed(order)
                        np.random.shuffle(order)
                        for t in order:
                            idx = j*k+t
                            if docs[i]['question_skeleton'][idx] not in exist_skeleton or t == order[-1]:
                                exist_skeleton.add(docs[i]['question_skeleton'][idx])
                                filter_score.append(retrieved_scores[i][idx])
                                filter_skeleton.append(docs[i]['question_skeleton'][idx])
                                filter_question.append(docs[i]['question'][idx])
                                filter_context.append(docs[i]['context'][idx])
                            break
            elif filter_type == 'v2':
                k = retrieve_num // (n_docs - 1) # 先取得分最高的，再均匀选取
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
            elif filter_type == 'v3':
                # 使用jaccard 距离聚类，每一类随机选一个
                for i in range(len(retrieved_scores)):
                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    for j in range(n_docs):
                        current = np.where(clustering.labels_==j)[0]
                        idx = np.random.choice(current)
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
            elif filter_type == 'v4':
                # 使用jaccard 距离聚类，每一类中选取得分最高的那一个
                for i in range(len(retrieved_scores)):
                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    for j in range(n_docs):
                        current = np.where(clustering.labels_==j)[0]
                        current_scores = [retrieved_scores[i][k].item() for k in current]
                        idx = current[np.argmax(current_scores)]
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
            elif filter_type == 'v5': # 探索利用
                # 使用jaccard 距离聚类,按类平均分数给类排序,对分数进行归一化，然后采样
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    cluster_scores = {}
                    for j in range(n_docs):
                        current = np.where(clustering.labels_==j)[0]
                        current_scores = [retrieved_scores[i][k].item() for k in current]
                        avg_score = sum(current_scores) / len(current_scores)
                        cluster_scores[avg_score] = tuple(current)
                    ranked_cluster_scores = sorted(cluster_scores.items(), reverse=True)
                    temperature = 1.0
                    normalized_score = torch.nn.functional.softmax(torch.tensor([item[0] for item in ranked_cluster_scores])/ temperature,dim=0)
                    init_sample = 0
                    sample_threshold = 1 - 1/n_docs
                    first_sample_clusters_num = 0
                    for score in normalized_score:
                        if init_sample < sample_threshold:
                            init_sample += score
                            first_sample_clusters_num += 1
                        else:
                            break
                    for j in range(first_sample_clusters_num): # 先采first_sample_clusters_num 个
                        idx = np.random.choice(ranked_cluster_scores[j][1])
                        exist_skeleton.add(docs[i]['question_skeleton'][idx])
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
                    for _ in range(0,n_docs - first_sample_clusters_num): # 再根据概率继续采剩下的n_docs-first_sample_clusters_num个
                        # second_score = torch.nn.functional.softmax(normalized_score[:first_sample_clusters_num],dim=0)
                        # cluster = np.random.choice(range(first_sample_clusters_num),replace=True, p=normalized_score.numpy())
                        cluster = np.random.randint(0,first_sample_clusters_num)
                        idx = np.random.choice(ranked_cluster_scores[cluster][1])
                        count = 0
                        while docs[i]['question_skeleton'][idx] in exist_skeleton and count < 500: # 防止死锁   
                            if count < 20: 
                                cluster = np.random.randint(0,first_sample_clusters_num)
                                idx = np.random.choice(ranked_cluster_scores[cluster][1])
                            else:
                                cluster = np.random.choice(range(n_docs),replace=True, p=normalized_score.numpy())
                                idx = np.random.choice(ranked_cluster_scores[cluster][1])
                            count += 1
                        exist_skeleton.add(docs[i]['question_skeleton'][idx])
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
            elif filter_type == 'v6': #先给一个空的skeleton，每一类再取最高得分的
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    exist_skeleton.add(None)
                    filter_score.append(torch.tensor(1))
                    filter_skeleton.append(None)
                    filter_question.append(None)
                    filter_context.append(None)

                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs-1,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    for j in range(n_docs-1):
                        current = np.where(clustering.labels_==j)[0]
                        current_scores = [retrieved_scores[i][k].item() for k in current]
                        idx = current[np.argmax(current_scores)]
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
            elif filter_type == 'v7': #先给一个空的skeleton,每一类中在高分中采样
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    exist_skeleton.add(None)
                    filter_score.append(torch.tensor(1))
                    filter_skeleton.append(None)
                    filter_question.append(None)
                    filter_context.append(None)

                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs-1,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    for j in range(n_docs-1):
                        lamda = 0.5
                        current = np.where(clustering.labels_==j)[0]
                        score_dict = {retrieved_scores[i][k].item():pos for pos,k in enumerate(current)}
                        sorted_score_list = sorted(score_dict.items(),key=lambda x : x[0],reverse=True)
                        num = int(lamda * len(sorted_score_list))
                        if  num <= 1:
                            num = len(sorted_score_list)
                        high_score_list = sorted_score_list[: num]
                        high_score_idx = [ x[1] for x in high_score_list]
                        idx = np.random.choice(high_score_idx)
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])

        else: # train
            if filter_type == 'v0': # 直接取前5个
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    for j in range(n_docs):
                        exist_skeleton.add(docs[i]['question_skeleton'][j])
                        filter_score.append(retrieved_scores[i][j])
                        filter_skeleton.append(docs[i]['question_skeleton'][j])
                        filter_question.append(docs[i]['question'][j])
                        filter_context.append(docs[i]['context'][j])
            elif filter_type == 'v1':
                # 采样n_docs个，20% 间隔，
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    for j in range(n_docs):
                        k = np.random.randint(int(0.2*j*retrieve_num),int(0.2*(j+1)*retrieve_num))
                        count = 0
                        while docs[i]['question_skeleton'][k] in exist_skeleton and count < retrieve_num: #有耗时的风险
                            k = np.random.randint(int(0.2*j*retrieve_num),int(0.2*(j+1)*retrieve_num))
                            count += 1
                        exist_skeleton.add(docs[i]['question_skeleton'][k])
                        filter_score.append(retrieved_scores[i][k])
                        filter_skeleton.append(docs[i]['question_skeleton'][k])
                        filter_question.append(docs[i]['question'][k])
                        filter_context.append(docs[i]['context'][k])

            elif filter_type == 'v2':
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

            elif filter_type == 'v3':
                for i in range(len(retrieved_scores)):
                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    for j in range(n_docs):
                        current = np.where(clustering.labels_==j)[0]
                        idx = np.random.choice(current)
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
            elif filter_type == 'v4':
            # 使用jaccard 距离聚类,按类平均分数给类排序,对分数进行归一化，然后采样
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    cluster_scores = {}
                    for j in range(n_docs):
                        current = np.where(clustering.labels_==j)[0]
                        current_scores = [retrieved_scores[i][k].item() for k in current]
                        avg_score = sum(current_scores) / len(current_scores)
                        cluster_scores[avg_score] = tuple(current)
                    ranked_cluster_scores = sorted(cluster_scores.items(), reverse=True)
                    temperature = 1.0
                    normalized_score = torch.nn.functional.softmax(torch.tensor([item[0] for item in ranked_cluster_scores]) / temperature,dim=0)
                    init_sample = 0
                    sample_threshold = 1 - 1/n_docs
                    first_sample_clusters_num = 0
                    for score in normalized_score:
                        if init_sample < sample_threshold:
                            init_sample += score
                            first_sample_clusters_num += 1
                        else:
                            break
                    for j in range(first_sample_clusters_num): # 先采first_sample_clusters_num 个
                        idx = np.random.choice(ranked_cluster_scores[j][1])
                        exist_skeleton.add(docs[i]['question_skeleton'][idx])
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
                    for _ in range(n_docs - first_sample_clusters_num): # 再根据概率继续采剩下的n_docs-first_sample_clusters_num个
                        # second_score = torch.nn.functional.softmax(normalized_score[:first_sample_clusters_num],dim=0)
                        if normalized_score.size()[0] != n_docs:
                            print(f'normalized_score:{normalized_score},ranked_cluster_scores:{ranked_cluster_scores},n_docs:{n_docs}')
                            n_docs = normalized_score.size()[0]

                        cluster = np.random.choice(range(n_docs),replace=True, p=normalized_score.numpy())
                        idx = np.random.choice(ranked_cluster_scores[cluster][1])
                        while docs[i]['question_skeleton'][idx] in exist_skeleton:
                            cluster = np.random.choice(range(n_docs),replace=True, p=normalized_score.numpy())
                            idx = np.random.choice(ranked_cluster_scores[cluster][1])  
                        exist_skeleton.add(docs[i]['question_skeleton'][idx])
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])

            elif filter_type == 'v5': #针对newsqa，先给一个空的skeleton
                for i in range(len(retrieved_scores)):
                    exist_skeleton = set()
                    exist_skeleton.add(None)
                    filter_score.append(torch.tensor(1))
                    filter_skeleton.append(None)
                    filter_question.append(None)
                    filter_context.append(None)

                    jaccard_distance = np.ones((retrieve_num,retrieve_num))
                    for j in range(retrieve_num):
                        for k in range(retrieve_num):
                            jaccard_distance[j,k] = caculate_jaccard_distance(docs[i]['question_skeleton'][j],docs[i]['question_skeleton'][k])
                    cluster = AgglomerativeClustering(n_clusters=n_docs-1,affinity='precomputed',linkage='average')
                    clustering = cluster.fit(jaccard_distance)
                    for j in range(n_docs-1):
                        current = np.where(clustering.labels_==j)[0]
                        current_scores = [retrieved_scores[i][k].item() for k in current]
                        idx = current[np.argmax(current_scores)]
                        filter_score.append(retrieved_scores[i][idx])
                        filter_skeleton.append(docs[i]['question_skeleton'][idx])
                        filter_question.append(docs[i]['question'][idx])
                        filter_context.append(docs[i]['context'][idx])
        context_input_ids, context_attention_mask = self.postprocess_docs(
            filter_skeleton, contexts,prefix, n_docs, return_tensors=return_tensors,data_args=data_args)
        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
            },
            tensor_type=return_tensors,
        ),filter_skeleton,filter_question,filter_context,filter_score