# Imports

import matplotlib
import os
from omegaconf import DictConfig, OmegaConf
import hydra

from data_handler import DataHandler
from data_analyser import DataAnalyzer
from model_handler import ModelHandler
from steering_handler import SteeringHandler

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

    

@hydra.main(version_base=None, config_path=".", config_name="config_updated.yaml")
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
    
    
    data_analyzer = DataAnalyzer(images_dir, metrics_dir, SEED)


    model_handler.compute_activations(activations_cache)



    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_embedded_data_dict, tsne_labels, tsne_prompts = data_analyzer.plot_embeddings(activations_cache, tsne_model)
    pca_model = PCA(n_components=2, random_state=42)
    pca_embedded_data_dict, pca_labels, pca_prompts = data_analyzer.plot_embeddings(activations_cache, pca_model)
    fa_model = FeatureAgglomeration(n_clusters=2)
    fa_embedded_data_dict, fa_labels, fa_prompts = data_analyzer.plot_embeddings(activations_cache, fa_model)

    # ToDo: 
    # Would be good if our code could just take any valid
    # dimensionality reduction method from sci-kit learn.
    dimensionality_reduction_map = {
        'pca': PCA,
        'tsne': TSNE,
        'feature_agglomeration': FeatureAgglomeration,
        # Add more mappings as needed
    }



    # # Mapping of method names to their corresponding classes
    # # This assumes we have these classes imported correctly
    # # at the top of our file
    # dimensionality_reduction_map = {
    #     'pca': PCA,
    #     'tsne': TSNE,
    #     'feature_agglomeration': FeatureAgglomeration
    # }

    # results = {}
    # dim_red_methods = cfg.dim_red.methods

    # # Iterate through each dim red method and its configuration
    # for method_name, method_config in dim_red_methods.items():
    #     DimRedClass = dimensionality_reduction_map.get(method_name.lower())
        
    #     if not DimRedClass:
    #         print(f"{method_name} not found.")
    #         continue
        
    #     # Instantiate the model with parameters unpacked from method_config
    #     model = DimRedClass(**method_config)
        
    #     # Call the data_analyzer.plot_embeddings method with the model
    #     embedded_data_dict, labels, prompts = data_analyzer.plot_embeddings(activations_cache, model)
        
    #     # Store results
    #     results[method_name] = {
    #         'embedded_data_dict': embedded_data_dict,
    #         'labels': labels,
    #         'prompts': prompts
    #     }
    


    classifier_methods = OmegaConf.to_container(cfg.classifiers.methods, resolve=True)

    # See if the dimensionality reduction representations can be used to classify the ethical area
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


    if cfg.enable_steering:
        # Steering
        logging.info("Running steering")
        steering_handler = SteeringHandler(cfg, model_handler, data_handler)
        hidden_layers = model_handler.get_hidden_layers()
        concept_H_tests, concept_rep_readers = steering_handler.compute_directions(prompts_dict, rep_token=-1)
        data_analyzer.repreading_accuracy_plot(hidden_layers, concept_H_tests, concept_rep_readers)
    
    # Activations cache takes up a lot of space, only write if user sets
    # parameter
    if cfg.write_cache:
        model_handler.write_activations_cache(activations_cache, experiment_base_dir)



if __name__ == "__main__":
    # Run with:
    # >>> python3 main.py prompts_sheet="../data/inputs/prompts_honesty_integrity_compassion.xlsx" model_name="meta-llama/Llama-2-7b-hf"
    main()

