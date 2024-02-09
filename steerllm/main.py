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
from sklearn.decomposition import PCA
from typing import Any
from dataclasses import dataclass
import datetime
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import shutil
from transformers import AutoTokenizer,AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.io as pio



# Constants
# If going forward with Hydra, we put everything here
# That really is constant betweeen runs.
# Everything that is configurable goes in config.yaml
# The model name is now there.
PROMPT_COLUMN = "Prompt"
ETHICAL_AREA_COLUMN = "Ethical_Area"
POS_COLUMN = "Positive"
SRC_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(SRC_PATH, "..", "data")
OUTPUT_PICKLE = os.path.join(DATA_PATH, "outputs", "activations_cache.pkl")
SEED = 42

# May need to install tkinter if not included
matplotlib.use('TkAgg')


# Store all relevant data for a given prompt within this class
# Can construct progressively, assign the data as it is processed
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


# Add the hidden states (the weights of the model between attention/MLP blocks)
# Use the last token
# TODO: I am not sure if the "hidden layers" that pytorch gives for RepE are normalized or unnormalized
def add_numpy_hidden_states(activations_cache: list[Activation]) -> None:
    for act in activations_cache:
        raw_act = act.raw_activations
        for hook_name in raw_act:
            # Only get the hidden states, the last activations in the block
            if "hook_resid_post" in hook_name:
                # Convert to numpy array
                # calling .cpu() while using a cpu is a no-op. Needed if run with gpu
                numpy_activation = raw_act[hook_name].cpu().numpy()
                # Get only the last token activations using '-1'
                # print("numpy_activation.shape", numpy_activation.shape)
                act.hidden_states.append(numpy_activation[0][-1])


# Extract the data we are actually using from the dictionary
def populate_data(prompts_dict: dict[str, Any]) -> list[Activation]:
    activations_cache = []
    for prompt, ethical_area, positive in zip(prompts_dict[PROMPT_COLUMN],
                                              prompts_dict[ETHICAL_AREA_COLUMN],
                                              prompts_dict[POS_COLUMN]):
        act = Activation(prompt, ethical_area, bool(positive), None, [])
        activations_cache.append(act)
    return activations_cache


# Make tsne plot for hidden state of every layer
def tsne_plot(activations_cache: list[Activation], images_dir: str) -> None:

    # Dictionary to store the embedded data for each layer
    # Key: layer number, Value: embedded data
    # Will be used for classification in the function classifier_battery
    embedded_data_dict = {}

    # Using activations_cache[0] is arbitrary as they all have the same number of layers
    # (12 with GPT-2-small) with representations
    for layer in range(len(activations_cache[0].hidden_states)):
        
        # print(f"layer {layer}")

        data = np.stack([act.hidden_states[layer] for act in activations_cache])
        labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]
        prompts = [act.prompt for act in activations_cache]

        # print("data.shape", data.shape)

        tsne = TSNE(n_components=2, random_state=SEED)
        embedded_data = tsne.fit_transform(data)

        embedded_data_dict[layer] = embedded_data
        
        df = pd.DataFrame(embedded_data, columns=["X", "Y"])
        df["Ethical Area"] = labels
        ax = sns.scatterplot(x='X', y='Y', hue='Ethical Area', data=df)
        
        plot_path = os.path.join(images_dir, f"tsne_plot_layer_{layer}.png")
        plt.savefig(plot_path)

        plt.close()

    return embedded_data_dict, labels, prompts

  



# Make PCA plot for hidden state of every layer
def pca_plot(activations_cache: list[Activation], images_dir: str) -> None:

    # Using activations_cache[0] is arbitrary as they all have the same number of layers
    for layer in range(len(activations_cache[0].hidden_states)):
        
        data = np.stack([act.hidden_states[layer] for act in activations_cache])
        labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]

        pca = PCA(n_components=2, random_state=SEED)
        embedded_data = pca.fit_transform(data)
        
        df = pd.DataFrame(embedded_data, columns=["X", "Y"])
        df["Ethical Area"] = labels
        ax = sns.scatterplot(x='X', y='Y', hue='Ethical Area', data=df)
        
        plot_path = os.path.join(images_dir, f"pca_plot_layer_{layer}.png")
        plt.savefig(plot_path)

        plt.close()


# Raster Plot has columns = Neurons and rows = Prompts. One chart for each layer
def raster_plot(activations_cache: list[Activation],
                images_dir: str,
                compression: int=5) -> None:
    # Using activations_cache[0] is arbitrary as they all have the same number of layers
    for layer in range(len(activations_cache[0].hidden_states)):
        
        data = np.stack([act.hidden_states[layer] for act in activations_cache])
        
        # Set the font size for the plot
        plt.rcParams.update({'font.size': (15 / compression)})

        neurons = activations_cache[0].hidden_states[0].size
        # Create the raster plot
        plt.figure(figsize=((neurons / (16 * compression)),
                            (len(activations_cache) + 2) / (2 * compression)))
        plt.imshow(data, cmap='hot', aspect='auto', interpolation="nearest")
        plt.colorbar(label='Activation')

        plt.xlabel('Neuron')
        plt.ylabel('Prompts')
        plt.title(f'Raster Plot of Neural Activations Layer {layer}')

        # Add Y-tick labels of the prompt
        plt.yticks(range(len(activations_cache)),
                   [activation.prompt for activation in activations_cache],
                   wrap=True)

        plot_path = os.path.join(images_dir, f"raster_plot_layer_{layer}.svg")
        plt.savefig(plot_path)

        plt.close()


# Initialize output directory with timestamp
def create_output_directories() -> tuple[str, str]:
    # Note this is in UTC, so regardless of where the code is run,
    # we can track when one experiment was run relative to another
    current_datetime = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')

    experiment_base_dir = os.path.join(DATA_PATH, "outputs", current_datetime)
    images_dir = os.path.join(experiment_base_dir, 'images')
    metrics_dir = os.path.join(experiment_base_dir, 'metrics')

    os.makedirs(experiment_base_dir)
    os.makedirs(images_dir)
    os.makedirs(metrics_dir)

    return experiment_base_dir, images_dir, metrics_dir


# Write cache data
def write_activations_cache(activations_cache: Any, experiment_base_dir: str) -> None:
    filename = os.path.join(experiment_base_dir, "activations_cache.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(activations_cache, f)


# Write experiment parameters
def write_experiement_parameters(experiment_params: dict, experiment_base_dir: str) -> None:
    with open(os.path.join(experiment_base_dir, "experiment_parameters.json"), 'w') as f:
        json.dump(experiment_params, f)


def classifier_battery(embedded_data_dict, labels, prompts, metrics_dir) -> None:

    # Define classifiers
    classifiers = {
        "logistic_regression": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "svc": SVC(probability=True),
        "knn": KNeighborsClassifier(),
        "gradient_boosting": GradientBoostingClassifier()
    }

    for name, clf in classifiers.items():

        print(name)

        metrics_dict = {"Layer": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

        for layer, representations in embedded_data_dict.items():
            print(layer)
            # random_state is set to 42 for reproducibility
            # the same train-test split is used for all classifiers
            X_train, X_test, y_train, y_test = train_test_split(representations, labels, test_size=0.2, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            # confusion = confusion_matrix(y_test, y_pred)

            metrics_dict["Layer"].append(layer)
            metrics_dict["Accuracy"].append(accuracy)
            metrics_dict["Precision"].append(precision)
            metrics_dict["Recall"].append(recall)
            metrics_dict["F1 Score"].append(f1)

            # Plot decision boundary
            # Note I am going to do the decision boundary plots
            # for the entire dataset (representations), not just the test set
            # because I want to to be able to look at all the data (prompts)
            # to get a sense of which ones are being classified incorrectly
            # with this boundary.
            x_min, x_max = representations[:, 0].min() - 1, representations[:, 0].max() + 1
            y_min, y_max = representations[:, 1].min() - 1, representations[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))

            # Predict probabilities, if possible
            if hasattr(clf, "predict_proba"):
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            else:
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            color_map = {'Bad False': 'red', 'Good True': 'green'}  # Define your own color mapping
            colors = [color_map[label] for label in labels]  # Map labels to colors

            # *****************
            # MATPOLIB PLOTTING
            # *****************

            # This code plots a continuum of decision boundaries
            # plt.contourf(xx, yy, Z, alpha=0.4)
            # plt.scatter(representations[:, 0], representations[:, 1], c=colors, s=20, edgecolor='k')

            # This code plots a single decision boundary for the 0.5 threshold
            plt.contour(xx, yy, Z, levels=[0.5], colors='black')
            plt.scatter(representations[:, 0], representations[:, 1], c=colors, edgecolors='k')

            plt.title('Decision boundary for ' + name)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.savefig(metrics_dir + "/decision_boundary_" + name + "_" + str(layer) + ".png")
            plt.close()

            # ***************
            # PLOTLY PLOTTING
            # ***************

            hover_text = [f'Prompt: {prompt}<br>Label: {label}' for prompt, label in zip(prompts, labels)]

            # Create the figure
            fig = go.Figure(data=[
                go.Contour(x=xx[0], y=yy[:, 0], z=Z, contours=dict(start=0.5, end=0.5, size=1), line_width=2, showscale=False),
                go.Scatter(x=representations[:, 0], y=representations[:, 1], mode='markers', marker=dict(size=8, color=colors), text=hover_text, hoverinfo='text')
            ])

            # Set the title and axis labels
            fig.update_layout(
                title='Decision boundary for ' + name,
                xaxis_title='Feature 1',
                yaxis_title='Feature 2'
            )

            pio.write_html(fig, metrics_dir + "/decision_boundary_" + name + "_" + str(layer) + ".html")

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(metrics_dir + "/metrics_"+ name +'.csv', index=False)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:  
    if not cfg.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Load the model
    # ie gpt2-small, gpt2-XL, meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-7b-chat-hf
    # For llama2, we must load with huggingFace and pass in
    # Must run 'huggingface-cli login' first to get access to llama2 model
    if cfg.model_name.startswith("gpt"):
        model = transformer_lens.HookedTransformer.from_pretrained(cfg.model_name)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = transformer_lens.HookedTransformer.from_pretrained(cfg.model_name,
                                                                   tokenizer=tokenizer,
                                                                   hf_model=hf_model)

    # Load the inputs (prompts)
    prompts_sheet = cfg.prompts_sheet
    prompts_dict = csv_to_dictionary(prompts_sheet)

    # Create output directories
    experiment_base_dir, images_dir, metrics_dir = create_output_directories()

    # Copy the config.yaml file to the output directory
    # Why? So we can see what the configuration was for a given run.
    # config.yaml will change from run to run, so we want to save it for each run.
    with open(os.path.join(experiment_base_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    # Copy the specific prompts being used from inputs to outputs.
    # Why? While we are still figuring out which prompt datasets work.
    # Prompt datasets as inputs which don't work will probably be deleted over time.
    # Saving them for a specific output lets us track which prompts were used for which output.
    # We can stop doing this once we have prompts that work.
    input_file_stem  = os.path.splitext(os.path.basename(prompts_sheet))[0]
    prompts_output_path = os.path.join(experiment_base_dir, f"{input_file_stem}.csv")
    prompts_df = pd.DataFrame(prompts_dict)
    prompts_df.to_csv(prompts_output_path)

    activations_cache = populate_data(prompts_dict)

    # Can use pudb as interactive commandline debugger
    # import pudb; pu.db
    
    # Keeping info in memory in activations cache uses a lot of RAM with the bigger models
    # When running this script with gpt-XL and 150 prompts, run out of memory.
    # Only 6gb of model (1.5b parameters, float32)
    # 40 prompts in good/evil. Takes 4 minutes, uses 18 GB of RAM
    # TODO: Save things as we use them, don't keep everything in memory, don't
    # keep the complete set of activations in memory (don't reload them)
    compute_activations(model, activations_cache)
    
    add_numpy_hidden_states(activations_cache)
    
    # Get various representations for each layer
    tsne_embedded_data_dict, labels, prompts = tsne_plot(activations_cache, images_dir)
    pca_plot(activations_cache, images_dir)
    raster_plot(activations_cache, images_dir)

    # See if the representations can be used to classify the ethical area
    # Why are we actually doing this? Hypothesis - better seperation of ethical areas
    # Leads to better steering vectors. This actually needs to be tested.
    classifier_battery(tsne_embedded_data_dict, labels, prompts, metrics_dir)

    # Each transformer component has a HookPoint for every activation, which
    # wraps around that activation.
    # The HookPoint acts as an identity function, but has a variety of helper
    # functions that allows us to put PyTorch hooks in to edit and access the
    # relevant activation
    # Relevant when changing the model
    
    # Activations cache takes up a lot of space, only write if user sets
    # parameter
    if cfg.write_cache:
        write_activations_cache(activations_cache, experiment_base_dir)



if __name__ == "__main__":
    # Run with:
    # >>> python3 main.py prompts_sheet="../data/inputs/prompts_honesty_integrity_compassion.xlsx" model_name="meta-llama/Llama-2-7b-hf"
    main()

