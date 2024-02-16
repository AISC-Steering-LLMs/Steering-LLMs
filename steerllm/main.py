# Imports

import matplotlib
import os
from omegaconf import DictConfig, OmegaConf
import hydra

from data_handler import DataHandler
from data_analyser import DataAnalyzer
from model_handler import ModelHandler

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

import logging



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
    logging.info("Creating model handler to load the model")
    model_handler = ModelHandler(cfg)



    # Process data
    logging.info("Processing data")

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



    # Get activations
    logging.info("Getting activations")

    # Can use pudb as interactive commandline debugger
    # import pudb; pu.db
    
    # Keeping info in memory in activations cache uses a lot of RAM with the bigger models
    # When running this script with gpt-XL and 150 prompts, run out of memory.
    # Only 6gb of model (1.5b parameters, float32)
    # 40 prompts in good/evil. Takes 4 minutes, uses 18 GB of RAM
    # TODO: Save things as we use them, don't keep everything in memory, don't
    # keep the complete set of activations in memory (don't reload them)
    model_handler.compute_activations(activations_cache)
    


    # Analyze the data
    logging.info("Running data analysis")

    data_analyzer = DataAnalyzer(images_dir, metrics_dir, SEED)

    # Dimensionality reduction analysis
    logging.info("Running dimensionality reduction analysis")

    # ToDo: 
    # Would be good if our code could just take any valid
    # dimensionality reduction method from sci-kit learn.
    dimensionality_reduction_map = {
        'pca': PCA,
        'tsne': TSNE,
        'feature_agglomeration': FeatureAgglomeration,
        # Add more mappings as needed
    }
    
    classifier_methods = OmegaConf.to_container(cfg.classifiers.methods, resolve=True)

    # See if the dimensionality representation representations can be used to classify the ethical area
    # Why are we actually doing this? Hypothesis - better seperation of ethical areas
    # Leads to better steering vectors. This actually needs to be tested.
    for method_name, method_config in cfg.dim_red.methods.items():
        if method_name in dimensionality_reduction_map:
            # Prepare kwargs by converting OmegaConf to a native Python dict
            kwargs = OmegaConf.to_container(method_config, resolve=True)
            dr_class = dimensionality_reduction_map[method_name]
            dr_instance = dr_class(**kwargs)
            embedded_data_dict, labels, prompts = data_analyzer.plot_embeddings(activations_cache, dr_instance)
            # Now X_transformed can be used for further analysis or classification
            data_analyzer.classifier_battery(classifier_methods, embedded_data_dict, labels, prompts, dr_instance, 0.2)
        else:
            logging.warning(f"Warning: {method_name} is not a valid dimension reduction method or is not configured.")

    # Other dimensionality reduction related analysis
    logging.info("Running other dimensionality reduction related analysis")

    for method_name in cfg.other_dim_red_analyses.methods:
        if hasattr(data_analyzer, method_name):
            getattr(data_analyzer, method_name)(activations_cache)
        else:
            print(f"Warning: Method {method_name} not found in DataAnalyzer.")

    # Further analysis not based on dimensionality reduction
    logging.info("Running further analysis not based on dimensionality reduction")

    for method_name in cfg.non_dimensionality_reduction.methods:
        if hasattr(data_analyzer, method_name):
            getattr(data_analyzer, method_name)(activations_cache)
        else:
            print(f"Warning: Method {method_name} not found in DataAnalyzer.")
    
    # Activations cache takes up a lot of space, only write if user sets
    # parameter
    if cfg.write_cache:
        model_handler.write_activations_cache(activations_cache, experiment_base_dir)



if __name__ == "__main__":
    # Run with:
    # >>> python3 main.py prompts_sheet="../data/inputs/prompts_honesty_integrity_compassion.xlsx" model_name="meta-llama/Llama-2-7b-hf"
    main()

