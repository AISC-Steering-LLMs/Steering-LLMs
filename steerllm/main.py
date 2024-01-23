# Imports
import transformer_lens
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pickle
import torch
import sys
import sklearn
from sklearn.manifold import TSNE
from typing import Any
from dataclasses import dataclass
import logging
import datetime
import json

# Constants
PROMPT_COLUMN = "Prompt"
ETHICAL_AREA_COLUMN = "Ethical_Area"
POS_COLUMN = "Positive"
DATA_PATH = os.path.join("..", "data")
# PROMPT_FILE = "prompts.csv"
# PROMPT_FILE = "prompts.csv"
OUTPUT_PICKLE = os.path.join(DATA_PATH, "outputs", "activations_cache.pkl")
MODEL_NAME = "gpt2-small"
# PROMPT_FILE_PATH = os.path.join(DATA_PATH, "inputs", PROMPT_FILE)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# any environ changes are local to this script
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# May need to install tkinter if not included
matplotlib.use('TkAgg')

@dataclass
class Activation:
    prompt: str
    ethical_area: str
    positive: bool
    raw_activations: Any
    hidden_states: list[np.ndarray]


# Load ethical prompts
# Will work with csv or excel
def csv_to_dictionary(filename: str) -> dict[str, Any]:
    # with open(filename, 'r') as file:
    #     reader = csv.DictReader(file)
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filename)
    else:
        raise Exception("Expected file ending with .csv or .xlsx")
    # df = df.dropna()
    data = df.to_dict(orient="list")

    return data


# Compute forward pass and cache data
def compute_activations(model: Any, activations_cache: list[Activation]) -> None:
    for act in activations_cache:
        # Run the model and get logits and activations
        logits, raw_activations = model.run_with_cache(act.prompt)
        act.raw_activations = raw_activations


# Load cached data from file
def load_cache(filename: str) -> Any:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def add_numpy_hidden_states(activations_cache: list[Activation]) -> None:
    # hidden_states = {prompt: [] for prompt in prompts_dict[PROMPT_COLUMN]}
    for act in activations_cache:
        raw_act = act.raw_activations
        for hook_name in raw_act:
            # Only get the hidden states, the last activations in the block
            if "hook_resid_post" in hook_name:
                # Convert to numpy array
                numpy_activation = raw_act[hook_name].cpu().numpy()
                # Get only the last token activations using '-1'
                act.hidden_states.append(numpy_activation[0][-1])


def populate_data(prompts_dict: dict[str, Any]) -> list[Activation]:
    activations_cache = []
    for prompt, ethical_area, positive in zip(prompts_dict[PROMPT_COLUMN],
                                              prompts_dict[ETHICAL_AREA_COLUMN],
                                              prompts_dict[POS_COLUMN]):
        act = Activation(prompt, ethical_area, bool(positive), None, [])
        activations_cache.append(act)
    return activations_cache




def tsne_plot(activations_cache: list[Activation], images_dir: str) -> None:
    data = np.stack([act.hidden_states[-1] for act in activations_cache])
    labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]

    SEED = 42
    tsne = TSNE(n_components=2, random_state=SEED)
    embedded_data = tsne.fit_transform(data)
    
    df = pd.DataFrame(embedded_data, columns=["X", "Y"])
    df["Ethical Area"] = labels
    ax = sns.scatterplot(x='X', y='Y',hue='Ethical Area',data=df)
    
    plot_path = os.path.join(images_dir, "tsne_plot.png")
    plt.savefig(plot_path)


def create_directories() -> str:
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    experiment_base_dir = os.path.join(DATA_PATH, "outputs", current_datetime)
    images_dir = os.path.join(experiment_base_dir, 'images')

    os.makedirs(experiment_base_dir)
    os.makedirs(images_dir)

    return experiment_base_dir, images_dir


# Write cache data
def write_activations_cache(activations_cache: Any, experiment_base_dir: str) -> None:
    filename = os.path.join(experiment_base_dir, "activations_cache.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(activations_cache, f)

# Write experiment parameters
def write_experiement_parameters(experiment_params: dict, experiment_base_dir: str) -> None:
    with open(os.path.join(experiment_base_dir, "experiment_parameters.json"), 'w') as f:
        json.dump(experiment_params, f)

# Logging commented out for now
# Alternative to just saving a json file of parameters
# Overkill

# def setup_logger(path: str) -> logging.Logger:
#     full_path = os.path.join(experiment_base_dir, "experiment_parameters.log")
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     handler = logging.FileHandler(full_path)
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     return logger

# def log_parameters(params, logger):
#     for key, value in params.items():
#         logger.info(f"{key}: {value}")

    

if __name__ == "__main__":
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME)

    # Run with:
    # >>> python3 main.py ../data/prompts_good_evil_justice.xlsx
    # Check if a filename is provided as an argument
    if len(sys.argv) < 2:
        print("Please provide a filename as an argument.")
        sys.exit(1)

    experiment_params = {
        'model_name': MODEL_NAME,
        'experiment_notes': "Testing out the transformer lens library."
    }

    experiment_base_dir, images_dir = create_directories()

    input_path = sys.argv[1]
    # file_path = PROMPT_FILE_PATH
    prompts_dict = csv_to_dictionary(input_path)

    # Copy the specific prompts being used from inputs to outputs.
    # Why? While we are still figuring out which prompt datasets work.
    # Prompt datsets as inputs which don't work will probably be deleted over time.
    # Saving them for a specific output lets us track which prompts were used for which output.
    # We can stop doing this once we have prompts that work.
    input_file_stem  = os.path.splitext(os.path.basename(input_path))[0]
    prompts_output_path = os.path.join(experiment_base_dir, input_file_stem+".csv")
    prompts_df = pd.DataFrame(prompts_dict)
    prompts_df.to_csv(prompts_output_path)

    print(prompts_dict.keys())
    print(len(prompts_dict.keys()))
    # Log the number of prompts

    activations_cache  = populate_data(prompts_dict)

    # Load a model (eg GPT-2 Small)
    # GPT-2 Small uses embedding 768 dimension word/token embedding
    # http://jalammar.github.io/illustrated-gpt2/
    # 1024 medium, 1280 large, 1600 extra large
    # >>> print(activations_cache[0]['hook_embed'].shape)                                     
    # torch.Size([1, 11, 768])   
    # This represents the input, 11 tokens each with 768 dimensions
    # Each layer has k, q, v vectors. Attention scores. Normalize. Feed forward/MLP. Residual Connection
    # Can get all of the steps from transformer lens
    # transformer_lens supported models https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py

    # import pudb; pu.db
    # n_layers: 12. The number of transformer blocks in the model (a block contains an attention layer and an MLP layer)
    # n_heads: 12. The number of attention heads per attention layer
    # d_model: 768. The residual stream width.
    # d_head: 64. The internal dimension of an attention head activation.
    # d_mlp: 3072. The internal dimension of the MLP layers (ie the number of neurons).
    # d_vocab: 50267. The number of tokens in the vocabulary.
    # n_ctx: 1024. The maximum number of tokens in an input prompt.
    
    # Hooks can be used to change model
    # have access to all the vectors, as well as scaling, normalization, how the residual layer is added, etc.
    # blocks.0.hook_resid_post is the final output
    # transformer_lens demo - https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=O15dgROWq3zy
    
    compute_activations(model, activations_cache)
    
    add_numpy_hidden_states(activations_cache)

    # print(activations_cache)
        
    # Print last hidden state and prompt
    # for activation in activations_cache:
        # print(activation.prompt)
        # print(activation.hidden_states[-1])
        # data.append(activation.hidden_states[-1])
    
    # TODO: Give a better name
    tsne_plot(activations_cache, images_dir)

    # plt.show()
        
    # Each transformer component has a HookPoint for every activation, which
    # wraps around that activation.
    
    # The HookPoint acts as an identity function, but has a variety of helper
    # functions that allows us to put PyTorch hooks in to edit and access the
    # relevant activation
    
    # print("\n".join(activations_cache[0]))
    # print(activations_cache[0]["blocks.0.hook_resid_post"])
    # calling .cpu() while using a cpu is a no-op. Needed if run with gpu
    # print(activations_cache[0]["blocks.0.hook_resid_post"].cpu().numpy())
    # >>> print(np.max(activations_cache[0]["blocks.0.hook_resid_post"].cpu().numpy()))       
    # 120.465195                                                                              
    # >>> print(np.mean(activations_cache[0]["blocks.0.hook_resid_post"].cpu().numpy()))      
    # 6.321705e-09      
    # >>> print(np.std(activations_cache[0]["blocks.0.hook_resid_post"].cpu().numpy()))       
    # 2.7253985 
    # After each, we scale and normalize. But we can use the unscaled version or the scaled version.
    # Scaled version of output of block 0 is
    # blocks.1.ln1.hook_scale
    # print(np.std(activations_cache[0]["blocks.1.ln1.hook_scale"].cpu().numpy()))
    # and normalized is
    # blocks.1.ln1.hook_normalized
    # print(np.std(activations_cache[0]["blocks.1.ln1.hook_normalized"].cpu().numpy()))
    # 0.99999887
    # I don't think the normalization actually matters much for our purposes, I think it is fine.
    # TODO: I am not sure if the "hidden layers" that pytorch gives for RepE are normalized or unnormalized


    # TODO: Create the raster plot
    # TODO: Cluster with PCS in addition to TSNE, figure out which are most simlar
    # For now, using the last layer, last token as representative embedding for input
    # np.digitize useful for raster plot - https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/classify-plot-raster-data-in-python/

    # Logging commented out for now
    # logger = setup_logger(experiment_base_dir)
    # log_parameters(experiment_params, logger)

    write_experiement_parameters(experiment_params, experiment_base_dir)
    write_activations_cache(activations_cache, experiment_base_dir)

