from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', type=str, default="gpt2",
                        help='the path to load the fine-tuned model')
    parser.add_argument('--max_length', type=int, default=32,
                        help='maximum length for code generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature for sampling-based code generation')
    parser.add_argument("--use_cuda", default=False, action="store_true", help="inference with GPU?")
    args, unknown = parser.parse_known_args()

    print(f"Use CUDA: {args.use_cuda}")

    # Load fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    model.eval()

    context = input("Enter the context: ")

    # Generate questions
    input_ids = tokenizer.encode(context, return_tensors='pt')
    input_ids = input_ids.to("cuda") if args.use_cuda else input_ids

    generated_questions = set()
    unique_questions = []
    max_iterations = 1# Maximum number of iterations to prevent infinite loop

    while len(unique_questions) < 20 and len(generated_questions) < max_iterations:
        outputs = model.generate(input_ids=input_ids,
                                 max_length=args.max_length,
                                 temperature=args.temperature,
                                 num_return_sequences=1)

        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)

            if decoded not in generated_questions:
                generated_questions.add(decoded)
                unique_questions.append(decoded)

    print("Generated Questions:")
    for question in unique_questions:
        print(question)
