# Imports
import transformer_lens
import pandas as pd
import os
import pickle
from typing import Any

# Constants
PROMPT_COLUMN = "Prompt"
ETHICAL_AREA_COLUMN = "Ethical_Area"
POS_COLUMN = "Positive"
DATA_PATH = os.path.join("..", "data")
PROMPT_FILE = "prompts.csv"
OUTPUT_PICKLE = os.path.join(DATA_PATH, "activations_cache.pkl")
MODEL_NAME = "gpt2-small"
PROMPT_FILE_PATH = os.path.join(DATA_PATH, PROMPT_FILE)

# Load ethical prompts
def csv_to_dictionary(filename: str) -> dict[str, Any]:
    # with open(filename, 'r') as file:
    #     reader = csv.DictReader(file)
    df = pd.read_csv(filename)
    df = df.dropna()
    data = df.to_dict(orient="list")

    return data


# Compute forward pass and cache data
def compute_activations(model: Any, prompts: list[str]) -> Any:
    activations_cache = []
    for prompt in prompts:
        # Run the model and get logits and activations
        logits, activations = model.run_with_cache(prompt)
        activations_cache.append(activations)
        # print(activations)
    return activations_cache


# Write cache data
def write_cache(filename: str, cache_data: Any) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(cache_data, f)


# Load cached data from file
def load_cache(filename: str) -> Any:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":

    prompts_dict = csv_to_dictionary(PROMPT_FILE_PATH)
    # print(prompts_dict)

    # Load a model (eg GPT-2 Small)
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME)

    # import pudb; pu.db
    activations_cache = compute_activations(model, prompts_dict[PROMPT_COLUMN])
    
    write_cache(OUTPUT_PICKLE, activations_cache)

