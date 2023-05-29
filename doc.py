import nltk
from nltk import sent_tokenize, pos_tag
from nltk.corpus import stopwords
from transformers import GPT2LMHeadModel, GPT2Tokenizer
nltk.download('stopwords')


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_questions(sentence):
    questions = []
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    stop_words = set(stopwords.words('english'))

    
    wh_words = ['who', 'what', 'when', 'where', 'why', 'how']
    for word, pos in tagged_tokens:
        if word.lower() not in stop_words and pos.startswith('NN'):
            for wh_word in wh_words:
                question = f"{wh_word.capitalize()} {word}?"
                questions.append(question)

    
    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    for i in range(len(tagged_tokens) - 1):
        if tagged_tokens[i][1] in verb_tags and tagged_tokens[i+1][1].startswith('NN'):
            question = f"Does {tagged_tokens[i+1][0]} {tagged_tokens[i][0]}?"
            questions.append(question)

    return questions

# Function to generate questions from a document
def generate_questions_from_document(document):
    questions = []
    sentences = sent_tokenize(document)
    for sentence in sentences:
        sentence_questions = generate_questions(sentence)
        questions.extend(sentence_questions)
    return questions

# Get user input as a document
document = input("Enter the document: ")

# Generate questions from the document
document_questions = generate_questions_from_document(document)

# Generate model responses for each question
for question in document_questions:
    input_ids = tokenizer.encode(question, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Question:", question)
    print("Model Response:", response)
    print()
