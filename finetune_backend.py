# finetune_backend.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from safetensors.torch import load_file
import pandas as pd
from torch.utils.data import Dataset
import os


class ExcelTextDataset(Dataset):
    def __init__(self, tokenizer, dataframe, input_col='input', output_col='output', max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.input_col = input_col
        self.output_col = output_col
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data.iloc[idx][self.input_col])
        output_text = str(self.data.iloc[idx][self.output_col])
        encoding = self.tokenizer(
            input_text + self.tokenizer.eos_token + output_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


def load_model_from_config_and_weights(config_path, weights_path):
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config)
    weights = load_file(weights_path)
    model.load_state_dict(weights, strict=False)
    return model


def fine_tune_model(config_path, model_path, tokenizer_path, excel_path, output_dir="finetuned_model", device='cpu'):
    df = pd.read_excel(excel_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    model = load_model_from_config_and_weights(config_path, model_path)
    model.to(device)

    dataset = ExcelTextDataset(tokenizer, df)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_total_limit=1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
