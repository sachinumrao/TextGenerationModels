from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from pathlib import Path

# load model
print("Downloading base model and tokenizer...")
model_id = ""
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id)

# lora config
peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

# get model with adapters
model = get_peft_model(model, peft_config)

print("Trainable Model Parameters: ", model.print_trainable_parameters())

# load data
train_dataset = None
eval_dataset = None
data_collator = None
compute_metrics = None

# training config
training_args = TrainingArguments(
    output_dir="",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train the model
trainer.train()

# save the model
model_output_dir = ""
model.save_pretrained(model_output_dir)
