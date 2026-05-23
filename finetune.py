"""
finetune.py — Fine-tune GPT-2 on your conversation dataset.

This is the core of nanoGPT2. It takes the pretrained GPT-2 model
(which already knows English) and teaches it your conversation format
by fine-tuning on Human/Assistant pairs.

This is the same process used to turn GPT-3 into ChatGPT,
just at a smaller scale and without RLHF.

Usage:
    python finetune.py --data data/input.txt --epochs 5 --output out/
"""

import os
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

class ConversationDataset(Dataset):
    """
    Loads a text file of Human/Assistant conversations and
    chunks it into fixed-size blocks for training.
    """
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, 'r') as f:
            text = f.read()

        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=False
        )['input_ids'][0]

        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(tokens[i : i + block_size])

        print(f"Dataset: {len(self.examples)} blocks of {block_size} tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {
            'input_ids': self.examples[i],
            'labels':    self.examples[i],
        }


def finetune(data_path, output_dir, epochs, lr, batch_size):
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    dataset = ConversationDataset(data_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=50,
        logging_steps=20,
        save_steps=200,
        use_cpu=False,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )

    print("Fine-tuning GPT-2 on your conversations...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done! Model saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on conversations')
    parser.add_argument('--data',       type=str,   default='data/input.txt')
    parser.add_argument('--output',     type=str,   default='out/')
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--lr',         type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int,   default=2)
    args = parser.parse_args()

    finetune(args.data, args.output, args.epochs, args.lr, args.batch_size)
