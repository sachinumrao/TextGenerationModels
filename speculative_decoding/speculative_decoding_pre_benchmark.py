import time
import traceback

from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
import torch
import torchao
torchao.quantization.utils.recommended_inductor_config_setter()

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

SAMPLE_QUERY = "What is the capital of Germany."

GREEDY_GENERATION_CONFIG = {
    "max_new_tokens": 32,
    "do_sample": False
}

def benchmark_generated_text(model, tokenizer):
    """Test script to benchmark generation from model"""
    input_ids = tokenizer.encode(SAMPLE_QUERY, return_tensors="pt").to(DEVICE)

    t_start = time.perf_counter()
    output = model.generate(input_ids, **GREEDY_GENERATION_CONFIG)
    t_stop = time.perf_counter()

    output_text = tokenizer.decode(output[0].to("cpu"), skip_special_tokens=True)
    num_tokens = len(output[0])
    print(f"Time taken: {t_stop - t_start:.2f} seconds")
    print(f"Generation Speed: {num_tokens / (t_stop - t_start):.2f} tokens/second")
    print(f"Generated text: {output_text}")
    print("-"*80)
    return output_text


def validate_tokenizers_vocab(draft_tokenizer, target_tokenizer):
    """Validate if both tokenizers have exact same vocabulary"""
    if set(draft_tokenizer.get_vocab().keys()) == set(target_tokenizer.get_vocab().keys()):
        print("Tokenizers have same vocabularies")
    else:
        print("Tokenizers have different vocabularies")

    print("-"*80)

def validate_input_ids(draft_tokenizer, target_tokenizer):
    """Validate if both tokenizers have exact same input_ids"""
    input_ids = draft_tokenizer.encode(SAMPLE_QUERY, return_tensors="pt")
    target_input_ids = target_tokenizer.encode(SAMPLE_QUERY, return_tensors="pt")
    if torch.equal(input_ids, target_input_ids):
        print("Input_ids are same")
    else:
        print("Input_ids are different")

    print("-"*80)

def main():
    draft_model_name = "Qwen/Qwen2.5-0.5B"
    target_model_name = "Qwen/Qwen2.5-1.5B"

    quant_config = TorchAoConfig(quant_type="int8_weight_only", group_size=128)

    print("Loading draft model...")
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto"
    )
    draft_model.generation_config.pad_token_id = draft_tokenizer.pad_token_id
    draft_model.to(DEVICE)
    draft_model.eval()

    print("Loading target model...")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto"
    )
    target_model.generation_config.pad_token_id = target_tokenizer.pad_token_id
    target_model.to(DEVICE)
    target_model.eval()

    # step-1 validate tokenizers
    try:
        validate_tokenizers_vocab(draft_tokenizer, target_tokenizer)
    except Exception as e:
        traceback.print_exc()
        return

    # step-2 validate input_ids
    try:
        validate_input_ids(draft_tokenizer, target_tokenizer)
    except Exception as e:
        traceback.print_exc()
        return

    # step-3 understand structure of generated output
    try:
        print("Inside the input output test block...")
        input_ids = draft_tokenizer.encode(SAMPLE_QUERY, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        print("Input Ids: " + "-"*80)
        print(input_ids)

        output_ids = draft_model.generate(input_ids, **GREEDY_GENERATION_CONFIG)
        print("Output Ids: " + "-"*80)
        print(output_ids)
        print("-"*80)
    except:
        traceback.print_exc()
        return

    # step-4 benchmark target model
    try:
        benchmark_generated_text(target_model, target_tokenizer)
    except Exception as e:
        traceback.print_exc()
        return

    # step-5 benchmark draft model
    try:
        benchmark_generated_text(draft_model, draft_tokenizer)
    except Exception as e:
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()


# TODO:
# - [ ] fix template for model input
# - [ ] fix warning about attention mask
