# RAST
This repository contains code for the paper [Diversify Question Generation with Retrieval-Augmented Style Transfer](https://arxiv.org/pdf/2310.14503.pdf)  
- we provide our processed_data in [data_link](https://drive.google.com/drive/folders/1eZ8_01Sg_xgz4z947nANGH38QCXWKm7c?usp=drive_link).
- we also provide our model checkpoint in [checkpoint_link](https://drive.google.com/drive/folders/1EYwBbkSr-6oCI4JDX35tXb_V-abwRPgU?usp=drive_link).
- if you use our repository, please cite [paper]().
If you find this code useful in your research, please consider citing:
```
@misc{gou2023diversify,
      title={Diversify Question Generation with Retrieval-Augmented Style Transfer}, 
      author={Qi Gou and Zehua Xia and Bowen Yu and Haiyang Yu and Fei Huang and Yongbin Li and Nguyen Cam-Tu},
      year={2023},
      eprint={2310.14503},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
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



