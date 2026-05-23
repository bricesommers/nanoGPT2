"""
chat.py — Chat with your fine-tuned model.

Usage:
    python chat.py --model out/
    python chat.py --model out/ --temperature 0.7

Type your message and press Enter.
Type 'quit' to exit.
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chat(model_dir, temperature, max_tokens):
    print(f"Loading model from {model_dir}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()

    # use MPS on Apple Silicon if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    print(f"Running on: {device}")
    print("\nnanoGPT2 ready! Type your message (or 'quit' to exit)\n")
    print("-" * 50)

    while True:
        user_input = input("Human: ").strip()
        if not user_input or user_input.lower() == 'quit':
            print("Goodbye!")
            break

        prompt = f"Human: {user_input}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                inputs['input_ids'],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        full = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = full[len(prompt):].split('Human:')[0].strip()
        print(f"Assistant: {reply}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with your fine-tuned GPT-2')
    parser.add_argument('--model',       type=str,   default='out/')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens',  type=int,   default=150)
    args = parser.parse_args()

    chat(args.model, args.temperature, args.max_tokens)
