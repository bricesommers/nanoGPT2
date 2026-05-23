"""
prepare.py — Tokenize your conversation dataset using GPT-2's tokenizer.

Usage:
    python data/prepare.py --input data/input.txt

Output:
    data/train.bin  — 90% of the data for training
    data/val.bin    — 10% of the data for validation

Format of input.txt:
    Human: your question here
    Assistant: your answer here

    Human: next question
    Assistant: next answer
"""

import os
import argparse
import numpy as np
import tiktoken

def prepare(input_path):
    with open(input_path, 'r') as f:
        data = f.read()

    print(f"Dataset: {len(data):,} characters")

    # use GPT-2's tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(data)
    tokens = np.array(tokens, dtype=np.uint16)
    print(f"Tokens: {len(tokens):,}")

    # 90/10 split
    n = len(tokens)
    train = tokens[:int(n * 0.9)]
    val   = tokens[int(n * 0.9):]
    print(f"Train: {len(train):,} | Val: {len(val):,}")

    out_dir = os.path.dirname(input_path)
    train.tofile(os.path.join(out_dir, 'train.bin'))
    val.tofile(os.path.join(out_dir, 'val.bin'))
    print("Saved train.bin and val.bin")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/input.txt')
    args = parser.parse_args()
    prepare(args.input)
