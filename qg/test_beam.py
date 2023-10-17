from transformers import T5ForConditionalGeneration,T5Tokenizer
import torch
model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
tokenizer = T5Tokenizer.from_pretrained('t5-base')
inputs = ['hello, how','I am']
inputs = tokenizer(inputs,max_length=28,padding='max_length')
kwags = {}
kwags['num_return_sequences'] = 5
kwags['num_beams'] = 10
kwags['num_beam_groups'] = 5
kwags['diversity_penalty'] = 10.0
out = model.generate(input_ids=torch.tensor(inputs['input_ids']).cuda(),max_length=128,**kwags)
outputs = tokenizer.batch_decode(out)
print(outputs)