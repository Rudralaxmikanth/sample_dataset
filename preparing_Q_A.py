import json
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

with open('input.json', 'r') as file:
    data = json.load(file)

def generate_unique_question(title, context):
    max_attempts = 10  
    generated_questions = set()  
    for _ in range(max_attempts):
        question_input = title + " " + context
        question_input_ids = tokenizer.encode(question_input, return_tensors='pt', truncation=True, padding=True)
        question_output = model.generate(question_input_ids, max_length=50, num_return_sequences=1)
        generated_question = tokenizer.decode(question_output[0], skip_special_tokens=True)

        if generated_question not in generated_questions:
            return generated_question

        generated_questions.add(generated_question)

    return None

output_data = []
for item in data['data']:
    title = item['title']
    context = item['context']
    qa_pairs = []

    for _ in range(20):  
        context_variations = [
            context,
            context + " Please provide some insights on it.",
            "What are the uses of " + title + "? " + context,
            "How does " + title + " work? " + context
        ]
        context_variation = random.choice(context_variations)

        generated_question = generate_unique_question(title, context_variation)

        if generated_question is not None:
            qa_pair = {"question": generated_question}
            qa_pairs.append(qa_pair)

    output_item = {"title": title, "context": context, "qa_pairs": qa_pairs}
    output_data.append(output_item)

with open('output145.json', 'w') as file:
    json.dump({"data": output_data}, file, indent=2)

print("Output JSON file created successfully.")
