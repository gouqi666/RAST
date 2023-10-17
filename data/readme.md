
  
# process data
1. download data in data/{dataset_name}   
2. python process_data.py --dataset={dataset_name}  
   2.1 process_squad_split1(path)   ---  preprocess  
   2.2 process_v3(path)             ---  extract skeleton  
   2.3 corrupt(path)                ---  corrupt skeleton  
   2.4 process_for_with_null_skeleton(path)   ---  use for  qg model initial training  
3. python root_dir/rag/prepare_dataset.py ()  --- process for faiss 
# data structure

```
data
│  process_data.py
│  readme.md
│  stopwords.txt
│              
├─newsqa
│  │  process_data.py
│  │  
│  ├─corrupt
│  │      train.jsonl
│  │      
│  ├─processed
│  │  │  dev.jsonl
│  │  │  index_dataset.json
│  │  │  my_knowledge_dataset_hnsw_index_question_skeleton.faiss
│  │  │  test.jsonl
│  │  │  train.jsonl
│  │  │  
│  │  └─my_knowledge_dataset_question_skeleton
│  │          data-00000-of-00001.arrow
│  │          dataset_info.json
│  │          state.json
│  │          
│  └─processed_with_null_skeleton
│          dev.jsonl
│          test.jsonl
│          train.jsonl
│          
├─SQuAD_1.1_split1
│  │  dev.jsonl
│  │  test.jsonl
│  │  train.jsonl
│  │  
│  ├─corrupt
│  │      train.jsonl
│  │      
│  ├─processed
│  │      dev.jsonl
│  │      index_dataset.json
│  │      squad.py
│  │      stopwords.txt
│  │      test.jsonl
│  │      train.jsonl
│  │      
│  └─processed_with_null_skeleton
│          dev.jsonl
│          test.jsonl
│          train.jsonl
└─SQuAD_1.1_split2
    │  long_text_2023-05-03-14-38-18.txt
    │  
    ├─corrupt
    │      dev.jsonl
    │      test.jsonl
    │      train.jsonl
    │      
    └─processed
    |        dev.jsonl
    |        index_dataset.json
    |        test.jsonl
    |        train.jsonl
    └─processed_with_null_skeleton
             dev.jsonl
             test.jsonl
             train.jsonl          
     
```

[comment]: <> (.)

[comment]: <> (+-- all_datasets  )

[comment]: <> (  |    +-- processed  )

[comment]: <> (      -- train.jsonl&#40;原始的训练数据 + 已经处理好的question_skeleton&#41;  )

[comment]: <> (      -- all_datasets.jsonl&#40;错的，不用管&#41;  )

[comment]: <> (      -- index_dataset.json（filter之后的数据,只用作retrieve时候的all_corpus）  )

[comment]: <> (  -- SQuAD_1.1_split1  )

[comment]: <> (    -- processed  )

[comment]: <> (      -- train.jsonl（split1原生数据+ skeleton）  )

[comment]: <> (      -- dev.jsonl（split1原生数据+ skeleton）  )

[comment]: <> (      -- test.jsonl（split1原生数据+ skeleton）  )

[comment]: <> (    -- processed_ctx&#40;训练过程保存doc embedding的中间产物&#41;  )

[comment]: <> (    -- corrupt  )

[comment]: <> (      -- train.jsonl &#40;processed/train.jsonl 中的skeleton经过corrput处理&#41;  )

[comment]: <> (    -- processed_with_null_skeleton（processed中的数据经过加入了null_skeleton,只用作训练base_qg_model_v2）  )

[comment]: <> (      -- train.jsonl  )

[comment]: <> (      -- dev.jsonl  )

[comment]: <> (      -- test.jsonl  )

[comment]: <> (    -- train.jsonl &#40;split1原生数据，process_squad_split1&#41;)

[comment]: <> (    -- dev.jsonl &#40;split1原生数据，process_squad_split1&#41;)

[comment]: <> (    -- test.jsonl &#40;split1原生数据，process_squad_split1&#41;)

[comment]: <> (  -- SQuAD_1.1_split2)

[comment]: <> (    -- processed)

[comment]: <> (      -- train.jsonl（split2原生数据+ skeleton）)

[comment]: <> (      -- dev.jsonl（split1原生数据+ skeleton）)

[comment]: <> (      -- test.jsonl（split1原生数据+ skeleton）)

[comment]: <> (    -- processed_ctx&#40;训练过程保存doc embedding的中间产物&#41;)

[comment]: <> (    -- corrupt)

[comment]: <> (      -- train.jsonl &#40;processed/train.jsonl 中的skeleton经过corrput处理&#41;)

[comment]: <> (      -- dev.jsonl&#40;与processed中的dev一样&#41;)

[comment]: <> (      -- test.jsonl（与processed中的test一样）)

[comment]: <> (    -- processed_with_null_skeleton（processed中的数据经过加入了null_skeleton）)

[comment]: <> (      -- train.jsonl)

[comment]: <> (      -- dev.jsonl)

[comment]: <> (      -- test.jsonl)

[comment]: <> (    -- train.json &#40;split2原生数据&#41;)

[comment]: <> (    -- dev.json &#40;split2原生数据&#41;)

[comment]: <> (    -- test.json &#40;split2原生数据&#41;)

[comment]: <> (  -- newsqa)

[comment]: <> (    -- processed)

[comment]: <> (      -- train.jsonl（newsqa原生数据+ skeleton）)

[comment]: <> (      -- dev.jsonl（newsqa原生数据+ skeleton）)

[comment]: <> (      -- test.jsonl（newsqa原生数据+ skeleton）)

[comment]: <> (    -- processed_ctx&#40;训练过程保存doc embedding的中间产物&#41;)

[comment]: <> (    -- corrupt)

[comment]: <> (      -- train.jsonl &#40;processed/train.jsonl 中的skeleton经过corrput处理&#41;)

[comment]: <> (      -- dev.jsonl&#40;与processed中的dev一样&#41;)

[comment]: <> (      -- test.jsonl（与processed中的test一样）)

[comment]: <> (    -- processed_with_null_skeleton（processed中的数据经过加入了null_skeleton）)

[comment]: <> (      -- train.jsonl)

[comment]: <> (      -- dev.jsonl)

[comment]: <> (      -- test.jsonl)

[comment]: <> (    -- split_data)

[comment]: <> (      -- train.json &#40;newsqa原生数据&#41;)

[comment]: <> (      -- dev.json &#40;newsqa原生数据&#41;)

[comment]: <> (      -- test.json &#40;newsqa原生数据&#41;)
