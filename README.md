# QG

# 1. download data
    ```
    squad1.1 两种split方式：   
    - zhou split,zhao split，是将dev中拆分出一般作为test，而且是sentence级别。86,635/8,965/8,964, Neural Question Generation from Text- A Preliminary Study
    - du split,  dev用作test，从训练集中抽取一部分作用dev，是paragraph层面的。 70484/10570/11877, Learning to Ask: Neural Question Generation for Reading Comprehension
    ```
# 2. process data
    ```
    python data/process_data.py   
    refer to  data/readme.md  
    prepare rag datasets(faiss), refer to rast/rag/prepare_dataset.py
    ```
# 2. train generator with skeleton
   ```
   refer to rast/qg/readme.md \
   ```
# 3. train vanilla generator
   ```
   refer to rast/qg/readme.md \
   ```
# 4. train QA model
    ```
    refer to rast/reward_mdoel/T5_QA/readme.md \
    ```
# 5. train rag
    ```
    refer to rast/rag/readme_v100.md \
    ```