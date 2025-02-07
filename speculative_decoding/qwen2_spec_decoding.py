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

def benchmark_generated_text(model, tokenizer, max_length=100, temperature=0.5):
    """Test script to benchmark generation from model"""
    input_ids = tokenizer.encode(SAMPLE_QUERY, return_tensors="pt").to(DEVICE)

    t_start = time.perf_counter()
    output = model.generate(input_ids, max_length=max_length, temperature=temperature)
    t_stop = time.perf_counter()

    output_text = tokenizer.decode(output[0].to("cpu"), skip_special_tokens=True)
    num_tokens = 0
    print(f"Time taken: {t_stop - t_start:.2f} seconds")
    print(f"Generation Speed: {num_tokens / (t_stop - t_start):.2f} tokens/second")
    print(f"Generated text: {output_text}")
    return output_text


def validate_tokenizers_vocab(draft_tokenizer, target_tokenizer):
    """Validate if both tokenizers have exact same vocabulary"""
    assert set(draft_tokenizer.get_vocab().keys()) == set(target_tokenizer.get_vocab().keys()), "Tokenizers have different vocabularies"

def validate_input_ids(draft_tokenizer, target_tokenizer):
    """Validate if both tokenizers have exact same input_ids"""
    input_ids = draft_tokenizer.encode(SAMPLE_QUERY, return_tensors="pt")
    target_input_ids = target_tokenizer.encode(SAMPLE_QUERY, return_tensors="pt")
    assert torch.equal(input_ids, target_input_ids), "Input_ids are different"

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
    target_model.to(DEVICE)
    target_model.eval()

    # step-1 validate tokenizers
    try:
        validate_tokenizers_vocab(draft_tokenizer, target_tokenizer)
    except AssertionError as e:
        traceback.print_exc()
        return

    # step-2 validate input_ids
    try:
        validate_input_ids(draft_tokenizer, target_tokenizer)
    except AssertionError as e:
        traceback.print_exc()
        return

    # step-3 understand structure of generated output
    try:
        input_ids = draft_tokenizer.encode(SAMPLE_QUERY, return_tensors="pt").to(DEVICE)
        output_ids = draft_model.generate(input_ids, max_length=32, temperature=1.0)
        print(output_ids)
    except Exception as e:
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
