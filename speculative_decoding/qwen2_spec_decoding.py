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




if __name__ == "__main__":
    main()
