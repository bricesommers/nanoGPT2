# nanoGPT2

The simplest, clearest repository for fine-tuning GPT-2 into a chatbot.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, nanoGPT2 takes the next step: it shows you how to turn a base language model into an instruct model — the same process used to build ChatGPT, at a scale you can run on a laptop.

    Human: What is gravity?
    Assistant: Gravity is a fundamental force that attracts objects with
    mass toward one another. On Earth, it pulls everything toward the
    center of the planet at about 9.8 meters per second squared.

## What this teaches you

nanoGPT taught you how to train a language model from scratch.
nanoGPT2 teaches you the next step: **fine-tuning**.

| Stage | What it is | Repo |
|---|---|---|
| Pretraining | Learn language from raw text | nanoGPT |
| Fine-tuning (SFT) | Learn to follow instructions | **nanoGPT2** |
| RLHF | Learn human preferences | ChatGPT |

The core idea: GPT-2 already knows English. We just show it 100+ examples of Human/Assistant conversations and it learns the format in under 2 minutes.

## Requirements

- Python 3.10+
- Apple Silicon Mac, NVIDIA GPU, or CPU
- ~2GB disk space for GPT-2 weights

## Install

    git clone https://github.com/bricesommers/nanoGPT2
    cd nanoGPT2
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

### 1. Conversation dataset

A 50-conversation dataset was created covering topics including science, philosophy, AI, programming, and general knowledge. Use it to fine tune GPT-2.
Each exchange uses a consistent format:

    Human: What is the capital of France?
    Assistant: The capital of France is Paris.

    Human: What is gravity?
    Assistant: Gravity is a force that attracts objects toward each other.

### 2. Fine-tune

    python finetune.py --data data/input.txt --epochs 5 --output out/

Options:

    --data        path to your input.txt  (default: data/input.txt)
    --output      where to save the model (default: out/)
    --epochs      number of training epochs (default: 5)
    --lr          learning rate (default: 5e-5)
    --batch_size  batch size (default: 2)

### 3. Chat

    python chat.py --model out/

Options:

    --model        path to fine-tuned model (default: out/)
    --temperature  randomness of responses, 0.1-1.0 (default: 0.7)
    --max_tokens   max length of each response (default: 150)

## How it works

**Base model vs Instruct model**

A base language model like GPT-2 was trained on 40GB of internet text to predict the next token. It knows English, facts, and structure — but it has no idea it is supposed to answer questions.

Fine-tuning shows it examples of the format we want:

    Human: [question]
    Assistant: [answer]

After seeing enough examples, it learns the pattern and starts responding as an assistant. This is the same process OpenAI used to turn GPT-3 into ChatGPT.

**Why GPT-2 and not from scratch?**

Training from scratch requires millions of examples and days of compute. GPT-2 already has that knowledge baked in. Fine-tuning takes minutes and needs only a few hundred conversations.

**What nanoGPT2 does not cover**

RLHF (Reinforcement Learning from Human Feedback) — the step that makes ChatGPT polished, safe, and reliable. A possible future extension is RLAIF (Reinforcement Learning from AI Feedback), where a capable AI like Claude rates responses instead of humans.

## Results

Trained on 103 conversations, 5 epochs, Apple M1:

    Training time:  87 seconds
    Final loss:     2.808
    Model size:     124M parameters (GPT-2 Small)

The model learns the Human/Assistant format reliably. Factual accuracy reflects GPT-2's underlying knowledge from 2019.

## File structure

    nanoGPT2/
    ├── README.md           this file
    ├── finetune.py         fine-tuning script
    ├── chat.py             interactive chat interface
    ├── requirements.txt    dependencies
    └── data/
        └── prepare.py      tokenize your dataset

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy — the inspiration for this project
- [GPT-2](https://openai.com/research/gpt-2) by OpenAI — the pretrained model we fine-tune
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — the fine-tuning infrastructure

## License

MIT
