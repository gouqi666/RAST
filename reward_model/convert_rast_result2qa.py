import json
path = '/home/student2021/gouqi/RAST/output/SQuAD_1.1_split1/RAG/experiments_v24/20.7399927157597/output_data.json'
with open(path) as f:
    data = json.load(f)
qa_data = []
for item in data:
    context = item['context'].replace('<HL>', '')
    answer = item['context'].split('<HL>')[1]
    for q in item['predicted']:
        item = {}
        item['context'] = context
        item['answer'] = answer
        item['question'] = q
        qa_data.append(item)
with open(path.replace('output_data.json','qa_data.jsonl'),'w') as fp:
    for item in qa_data:
        json.dump(item,fp)
        fp.write('\n')
