"""
This file creates the `finetune` uv tool (https://docs.astral.sh/uv/concepts/tools/),
which is used to fine-tune a particular embedding model on a provided text file.

Run `uv run finetune` on the command line for usage instructions.
"""
import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

import lib.config as config

from tools.constants import (
    FILENAME_ARG, OUTPATH_ARG, BASE_MODEL_ARG, LOG_DIR_ARG, TUNED_MODEL_ARG
)

parser = argparse.ArgumentParser(
    prog="uv run finetune",
    description="Fine-tunes a particular HuggingFace model on a body of text.",
)

parser.add_argument(
    FILENAME_ARG,
    help="Text file to use for model tuning.",
)
parser.add_argument(
    OUTPATH_ARG,
    help="Destination path for the fine-tuned model.",
    default=config.huggingface_model_cache_folder,
)
parser.add_argument(
    BASE_MODEL_ARG,
    help="Base model to fine-tune from.",
    default=config.huggingface_default_embedding_model,
)
parser.add_argument(
    TUNED_MODEL_ARG,
    help="Name for the fine-tuned model.",
    default=os.path.join(
        config.huggingface_finetuned_embedding_model,
    ),
)
parser.add_argument(
    LOG_DIR_ARG,
    help="Directory in which to store training log files.",
)


def main():
    args = parser.parse_args()

    # load base model
    model_name = args.base_model

    print(f'Base model: {model_name} -> New model: {args.tuned_model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # load training data
    data_files = {"train": args.filename}
    dataset = load_dataset("text", data_files=data_files)

    # tokenize and mask
    def tokenize_and_mask(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        labels = tokenized_inputs["input_ids"].clone()

        batch_size, seq_length = labels.shape

        probability_matrix = torch.full(labels.shape, 0.15)  # 15% probability per token

        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(
                seq.tolist(), already_has_special_tokens=True
            ) for seq in labels
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # Replace masked tokens with [MASK]
        tokenized_inputs["input_ids"][masked_indices] = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_mask, batched=True)
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True)

    # fine tune the model
    training_args = TrainingArguments(
        output_dir=args.outpath,
        evaluation_strategy="no",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=args.log_dir,
        logging_steps=500,
    )
    print(f"Using {training_args.n_gpu} GPU{"" if training_args.n_gpu == 1 else "s"}.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    trainer.train()

    # save the fine-tuned model
    new_model_path = os.path.join(args.outpath, args.tuned_model_name)
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    print(f"Trained model can be found at {new_model_path}")


if __name__ == "__main__":
    main()
