import os
from functools import partial
from pathlib import Path

import torch
import torchao
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TorchAoConfig,
    Trainer,
    TrainingArguments,
)

torchao.quantization.utils.recommended_inductor_config_setter()

os.environ["WANDB_PROJECT"] = "LOTR_Gemma2B_LORA"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = "cpu"

# global params
OUTPUT_DIR = os.path.join(Path.home(), "Models", "lotr_gemma2b_adapters")
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
NUM_EPOCHS = 5
MAX_LENGTH = 512
LR_SCHEDULER = "cosine"
DATA_PATH = os.path.join(Path.home(), "Data", f"lotr_grouped_{MAX_LENGTH}.csv")
DATA_COL = "Text_Chunk_LOTR"


def encode_dataset(tokenizer, example):
    outputs = tokenizer(
        example[DATA_COL],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_overflowing_tokens=False,
    )

    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


def get_dataset(tokenizer):
    dataset = load_dataset("csv", data_files=[DATA_PATH])
    dataset = dataset.shuffle(seed=42)
    train_split = 0.8
    train_size = int(train_split * len(dataset))

    print(dataset)
    dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # encode text data to tokens
    encode = partial(encode_dataset, tokenizer)
    tokenized_train_dataset = train_dataset.map(
        encode, num_proc=4, batched=True, desc="Tokenizing train dataset..."
    )

    tokenized_test_dataset = test_dataset.map(
        encode, num_proc=4, batched=True, desc="Tokenizing test dataset..."
    )

    return tokenized_train_dataset, tokenized_test_dataset


def main():
    # load model
    print("Downloading base model and tokenizer...")
    model_id = "google/gemma-2-2b"

    # config to load model in 8 bit
    quant_config = TorchAoConfig(quant_type="int8_weight_only", group_size=128)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    print("Quantized model loaded...")

    # lora config
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = prepare_model_for_kbit_training(model)
    print("Model ajsuted for 8bit training...")

    # get model with adapters
    model = get_peft_model(model, peft_config)
    print("Peft model created...")

    print("Trainable Model Parameters: ", model.print_trainable_parameters())

    # put model on compute device
    model.to(DEVICE)
    print("Model put in compute device...")

    # optimize model with compile => peft expects non-compiled model
    # model = torch.compile(model, mode="max-autotune")
    # print("Model optimized with compile...")

    # load data
    train_dataset, test_dataset = get_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # training config
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        lr_scheduler_type=LR_SCHEDULER,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        dataloader_num_workers=4,
        optim="adamw_torch_fused",
        # report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # train the model
    # trainer.train()

    # save the model
    # trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()


# TODO:
# - [x] implement text grouping
# - [x] text encoding with correct args
# - [x] wandb experiment tracking
