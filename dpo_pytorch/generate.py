import os
from pathlib import Path
from time import perf_counter

import torch
import torchao
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

torchao.quantization.utils.recommended_inductor_config_setter()

# load model
ADAPTER_PATH = os.path.join(Path.home(), "Models", "lotr_gemma2b_adapters2")
MODEL_ID = "google/gemma-2-2b"

GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = "cpu"


def load_model(with_adapters=True):
    # load model
    print("Loading model...")
    quant_config = TorchAoConfig(quant_type="int8_weight_only", group_size=128)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    # add adapters
    if with_adapters:
        print("Adding adapters...")
        model.load_adapter(ADAPTER_PATH)

    print("Model loaded successfully...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    return model, tokenizer


def generate(prompt, model, tokenizer):
    print("Generating lore...")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    t_init = perf_counter()
    outputs = model.generate(inputs.to(DEVICE), **GENERATION_CONFIG)
    t_stop = perf_counter()

    time_taken = t_stop - t_init
    num_tokens = len(outputs[0])
    tps = num_tokens / time_taken

    print("Generation Time: ", time_taken)
    print("TPS: ", tps)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response


if __name__ == "__main__":
    prompt = "Who is Aragorn?"

    print("Running base model...")
    model, tokenizer = load_model(with_adapters=False)

    model.to(DEVICE)
    _ = model.eval()

    response = generate(prompt, model, tokenizer)
    print("prompt: ", prompt)
    print("Response: ", response)

    print("Running model with adapter...")
    model, tokenizer = load_model(with_adapters=True)

    model.to(DEVICE)
    _ = model.eval()

    response = generate(prompt, model, tokenizer)
    print("prompt: ", prompt)
    print("Response: ", response)


# TODO
# - fix very slow generation
