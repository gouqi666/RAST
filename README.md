# RAST
Our paper link: [Diversify Question Generation with Retrieval-Augmented Style Transfer]()  
- we provide our processed_data in [data_link]().
- we also provide our model checkpoint in [checkpoint_link]().
- if you use our repository, please cite [paper]().
# To reproduce
## 1. download data
- squad1.1, zhou split  
    - This split of squad refers to [Neural Question Generation from Text- A Preliminary Study](/https://arxiv.org/pdf/1704.01792.pdf)  
    - data num of train/dev/test is 86,635/8,965/8,964 respectively.
- squad1.1, du split
    - This split of squad refers to [Learning to Ask: Neural Question Generation for Reading Comprehension](https://arxiv.org/pdf/1705.00106.pdf)  
    - data num of train/dev/test is 70484/10570/11877 respectively.
- newsqa 
    - This dataset refers to [NewsQA: A Machine Comprehension Dataset](https://arxiv.org/pdf/1611.09830.pdf)
    - data num of train/dev/test is 92549/5166/5126 respectively.
## 2. process data
- process original data
```
python data/process_data.py  
refer to  data/readme.md  
```
- convert and store corpus data into faiss vector 
```
python rast/rag/prepare_dataset.py
refer to rast/rag/prepare_dataset.py
```
## 3. train generator with skeleton
```
refer to rast/qg/readme.md 
```
## 4. train vanilla generator
```
refer to rast/qg/readme.md 
```
## 5. train QA model
```
refer to rast/reward_mdoel/T5_QA/readme.md
```
## 6. train rag
```
refer to rast/rag/readme_v100.md
```



