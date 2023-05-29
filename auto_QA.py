import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)



tokenizer.add_special_tokens({'pad_token': '[PAD]'})

with open('input.json', 'r') as file:
    data = json.load(file)


output_data = []
for item in data['data']:
    title = item['title']
    context = item['context']
    qa_pairs = []

    
    for _ in range(5):  
        question_input = title + " " + context
        question_input_ids = tokenizer.encode(question_input, return_tensors='pt', truncation=True, padding=True)
        question_output = model.generate(question_input_ids, max_length=100, num_return_sequences=1)
        generated_question = tokenizer.decode(question_output[0], skip_special_tokens=True)

    
        answer_input = generated_question + " " + context
        answer_input_ids = tokenizer.encode(answer_input, return_tensors='pt', truncation=True, padding=True)
        answer_output = model.generate(answer_input_ids, max_length=100, num_return_sequences=1)
        generated_answer = tokenizer.decode(answer_output[0], skip_special_tokens=True)

        qa_pair = {"question": generated_question, "answer": generated_answer}
        qa_pairs.append(qa_pair)

    output_item = {"title": title, "context": context, "qa_pairs": qa_pairs}
    output_data.append(output_item)

# Write output to JSON file
with open('output.json', 'w') as file:
    json.dump({"data": output_data}, file, indent=2)
