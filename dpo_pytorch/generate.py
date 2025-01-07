from peft import AutoPeftModelForCausalLM
from transformers import LlamaTokenizer
import torch
from pathlib import Path
from time import perf_counter

# load model
model_path = None
model_id = None
model = AutoPeftModelForCausalLM.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_id)

# generation config
generation_config = None


# generate text from prompt
while True:
    input_prompt = input("> ")

    t_init = perf_counter()
    output = model.generate(input_prompt, generation_config)
    decoded_output = None
    t_stop = perf_counter()
    print(decoded_output)
    print("Time Taken: ", t_stop - t_init)
    print("-"*120)
