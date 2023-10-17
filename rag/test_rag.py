import sys
sys.path.insert(0,'/mnt/workspace/gouqi/code/RQG')
# print(sys.path)
from transformers import RagRetriever,RagModel,RagTokenizer,AutoModelForSeq2SeqLM,T5ForConditionalGeneration,RagConfig,AutoConfig,BertForSequenceClassification,DPRQuestionEncoder,T5Tokenizer,DPRQuestionEncoderTokenizer,AutoTokenizer
from datasets import load_from_disk,load_dataset
from ragRetriever import MyRagRetriever
from ragModel import MyRagModel
from ragSequenceForGeneration import MyRagSequenceForGeneration


if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("quora")
    dataset_path = '/mnt/workspace/gouqi/code/RQG/data/SQuAD_V2/processed/my_knowledge_dataset'
    index_path = "/mnt/workspace/gouqi/code/RQG/data/SQuAD_V2/processed/my_knowledge_dataset_hnsw_index.faiss"
    dataset = load_from_disk(dataset_path)
    dataset.load_faiss_index("embeddings", index_path)
    test_dataset = load_dataset("/mnt/workspace/gouqi/code/RQG/data/SQuAD_V2/processed", split="test")
    
    config = RagConfig.from_pretrained('facebook/rag-sequence-base')
    config.index_name = 'custom'
    config.passages_path = dataset_path
    config.index_path = index_path
    config.use_dummy_dataset = False

    # tokenizer
    question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder_tokenizer.add_tokens(['<HL>','[mask]','<sep>'], special_tokens=True)
    generator_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    generator_tokenizer.add_tokens(['<HL>','[mask]','<sep>'], special_tokens=True) 
    ctx_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    ctx_encoder_tokenizer.add_tokens(['<HL>','[mask]','<sep>'], special_tokens=True)



    # model
    generator =  T5ForConditionalGeneration.from_pretrained(
        "/mnt/workspace/gouqi/code/RQG/experiments_skeleton/t5-base-GPUNums4-len128-fp16False--warm0.2--warmSteps0--weightDecay0.1-128-lr5e-05-b32-top_p0.9",
    )
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-multiset-base')
    question_encoder.resize_token_embeddings(len(question_encoder_tokenizer))
    # generator.resize_token_embeddings(len(generator_tokenizer))

    config.generator = generator.config
    config.question_encoder = question_encoder.config
    #question_encoder_config = AutoConfig.from_pretrained('bert-base-uncased')
    #generator_config = AutoConfig.from_pretrained('t5-base')
    
    #rag_config = RagConfig.from_question_encoder_generator_configs(question_encoder_config,generator_config)
    retriever = MyRagRetriever(config = config,
                                question_encoder_tokenizer= question_encoder_tokenizer,
                                generator_tokenizer=generator_tokenizer,
                            )

    retriever.set_ctx_encoder_tokenizer(ctx_encoder_tokenizer)
    # retriever = RagRetriever.from_pretrained("facebook/rag-sequence-base", index_name="legacy")



    model = MyRagModel(rag_config = config,question_encoder=question_encoder,generator=generator,retriever=retriever)
    if False: # 此处设置的是在生成模型的encode阶段是否对 contex的encode模型进行训练，之所以能训练，是因为 context encoder 用来计算 doc score
        # ctx_encoder = DPRContextEncoder.from_pretrained(hparams.context_encoder_name)
        ctx_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-multiset-base')
        ctx_encoder.resize_token_embeddings(len(ctx_encoder_tokenizer))
        model.set_context_encoder_for_training(ctx_encoder)
    # model = RagModel.from_pretrained("facebook/rag-sequence-base", retriever=retriever,generator = generator)
    inputs = question_encoder_tokenizer.batch_encode_plus([test_dataset[0]['question'],test_dataset[1]['question']],max_length=32,padding='longest',return_tensors="pt")
    batch_size = 16
    def highlight_context(context,answer):
        res = []
        for c,a in zip(context,answer):
            answer_start = c.find(a)
            assert answer_start != -1
            HL_context = c[:answer_start-1] + ' <HL> ' + a + \
                    ' <HL> ' + c[answer_start + len(a):]
            res.append(HL_context)
        return res
    for i in range(0,len(test_dataset),batch_size):
        data = test_dataset[i:i+batch_size]
        hl_context = highlight_context(data['context'],data['answer'])
        outputs = model(input_ids=inputs["input_ids"],contexts = hl_context)
        print(outputs)
