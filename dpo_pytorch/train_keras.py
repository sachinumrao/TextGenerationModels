import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import keras
import keras_nlp
import pandas as pd

keras.config.set_floatx("bfloat16")

TEMPLATE = "Instruction:\n{instruction}\n\nResponse:\n{response}"
INPUT_DATA_FILE = None
OUTPUT_MODEL_FILE = os.path.join(Path.home(), "Models", "lotr_gemma2b_adapters2")
LR = 5e-5
BATCH_SIZE = 2
NUM_EPOCHS = 5
MAX_LEN = 512
MODEL_ID = "gemma2_2b_en"


def get_data():
    df = pd.read_csv(INPUT_DATA_FILE)
    data = []
    for i, row in df.iterrows():
        instruction = row["instruction"]
        response = row["response"]
        data.append(TEMPLATE.format(instruction=instruction, response=response))

    return data


def load_model():
    model = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
    model.quantize("int8")
    return model


def main():
    # get data
    data = get_data()

    # load model
    model = load_model()

    # configure model
    model.quantize("int8")

    # add adapters
    model.backbone.enable_lora(rank=8)
    model.preprocessor.sequence_length = MAX_LEN
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print(model.summary())

    # train
    model.fit(data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.1)

    # save model
    model.save(OUTPUT_MODEL_FILE)

    pass


if __name__ == "__main__":
    main()


## TODO
# - [ ] add data in instruction format
# - [ ] add wandb logging
