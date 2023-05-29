import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

nltk.download('stopwords')

def generate_questions(sentence):
    # Generate questions based on a given sentence
    questions = []

    # Generate "What" and "Who" questions
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)

    # Generate "What" questions
    what_questions = []
    for i, (word, tag) in enumerate(tagged_tokens):
        if tag.startswith('N'):
            question = f"What is {word.lower()}?"
            what_questions.append(question)
    questions.extend(what_questions)

    # Generate "Who" questions
    who_questions = []
    for i, (word, tag) in enumerate(tagged_tokens):
        if tag == 'NNP' or (i > 0 and tagged_tokens[i-1][1] == 'NNP'):
            question = f"Who is {word}?"
            who_questions.append(question)
    questions.extend(who_questions)

    return questions

def generate_answers(generated_question, context):
    # Generate answers using a pre-trained model and tokenization
    answer_input = generated_question + " " + context
    answer_input_ids = tokenizer.encode(answer_input, return_tensors='pt', truncation=True, padding=True)
    answer_output = model.generate(answer_input_ids, max_length=100, num_return_sequences=1)
    generated_answer = tokenizer.decode(answer_output[0], skip_special_tokens=True)

    return generated_answer

def generate_questions_from_document(document):
    data = document["data"]
    all_question_answer_pairs = []

    for entry in data:
        context = entry["context"]
        sentences = sent_tokenize(context)

        for sentence in sentences:
            # Generate questions for each sentence
            sentence_questions = generate_questions(sentence)

            for question in sentence_questions:
                answer = generate_answers(question, sentence)
                question_answer_pair = {"question": question, "answer": answer}
                all_question_answer_pairs.append(question_answer_pair)

    return all_question_answer_pairs

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load the document from input.json
with open("input.json") as file:
    document = json.load(file)

# Generate question-answer pairs
question_answer_pairs = generate_questions_from_document(document)

# Save question-answer pairs to a new JSON file
output_file = "Rudra.json"
with open(output_file, "w") as file:
    json.dump(question_answer_pairs, file, indent=4)

print(f"Question-answer pairs saved to {output_file}.")
