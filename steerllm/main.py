# Imports

import matplotlib
import os
from omegaconf import DictConfig
import hydra

from data_handler import DataHandler
from data_analysis import AnalysisManager
from model_handling import ModelHandler


# Constants
# If going forward with Hydra, we put everything here
# That really is constant betweeen runs.
# Everything that is configurable goes in config.yaml
# The model name is now there.
SRC_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(SRC_PATH, "..", "data")
OUTPUT_PICKLE = os.path.join(DATA_PATH, "outputs", "activations_cache.pkl")
SEED = 42

# May need to install tkinter if not included
matplotlib.use('TkAgg')

    

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:  
    
    # Create a model handler
    # Instaitiate the model handler will load the model
    model_handler = ModelHandler(cfg)

    

    # Create a data handler
    data_handler = DataHandler(DATA_PATH)

    # Load the inputs (prompts)
    prompts_dict = data_handler.csv_to_dictionary(cfg.prompts_sheet)

    # Create output directories
    experiment_base_dir, images_dir, metrics_dir = data_handler.create_output_directories()

    # Copy the config.yaml file to the output directory and the prompts
    # Why? So we can see what the configuration was for a given run.
    # config.yaml will change from run to run, so we want to save it for each run.
    data_handler.write_experiment_parameters(cfg, prompts_dict, experiment_base_dir)

    activations_cache = data_handler.populate_data(prompts_dict)

    # Can use pudb as interactive commandline debugger
    # import pudb; pu.db
    
    # Keeping info in memory in activations cache uses a lot of RAM with the bigger models
    # When running this script with gpt-XL and 150 prompts, run out of memory.
    # Only 6gb of model (1.5b parameters, float32)
    # 40 prompts in good/evil. Takes 4 minutes, uses 18 GB of RAM
    # TODO: Save things as we use them, don't keep everything in memory, don't
    # keep the complete set of activations in memory (don't reload them)
    model_handler.compute_activations(activations_cache)
    
    
    analysis_manager = AnalysisManager(images_dir=images_dir, seed=SEED)

    # Get various representations for each layer
    tsne_embedded_data_dict, labels, prompts = analysis_manager.tsne_plot(activations_cache=activations_cache)
    analysis_manager.pca_plot(activations_cache)
    analysis_manager.raster_plot(activations_cache)
    analysis_manager.random_projections_plot(activations_cache)
    analysis_manager.feature_agglomeration(activations_cache)
    analysis_manager.probe_hidden_states(activations_cache)

    # See if the representations can be used to classify the ethical area
    # Why are we actually doing this? Hypothesis - better seperation of ethical areas
    # Leads to better steering vectors. This actually needs to be tested.
    analysis_manager.classifier_battery(tsne_embedded_data_dict, labels, prompts, metrics_dir)

    # Each transformer component has a HookPoint for every activation, which
    # wraps around that activation.
    # The HookPoint acts as an identity function, but has a variety of helper
    # functions that allows us to put PyTorch hooks in to edit and access the
    # relevant activation
    # Relevant when changing the model
    
    # Activations cache takes up a lot of space, only write if user sets
    # parameter
    if cfg.write_cache:
        model_handler.write_activations_cache(activations_cache, experiment_base_dir)



if __name__ == "__main__":
    # Run with:
    # >>> python3 main.py prompts_sheet="../data/inputs/prompts_honesty_integrity_compassion.xlsx" model_name="meta-llama/Llama-2-7b-hf"
    main()

