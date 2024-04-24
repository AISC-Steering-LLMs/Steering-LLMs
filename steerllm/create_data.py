import os
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from openai import OpenAI
import logging
import math
from jinja2 import Environment, FileSystemLoader

from dataset_creator import DatasetCreator

# Constants - things we don't want/need to configure in config.yaml
SRC_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(SRC_PATH), 'data')
CONFIG_DIR = os.path.join(SRC_PATH, "dataset_gen_configs")
DATASET_TEMPLATE_DIR = os.path.join(DATA_PATH, "inputs/templates/dataset_prompt_templates")
HEADER_LABELING_TEMPLATE_DIR = os.path.join(DATA_PATH, "inputs/templates/header_labelling_prompt_templates")

def get_user_response():
    print("\nAre you happy with this dataset prototype? (y/n)")
    while True:
        response = input("Enter your response: ").lower()
        if response in ['y', 'n']:
            return response
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

@hydra.main(version_base=None,
            config_path=CONFIG_DIR,
            config_name="concept_contrastive_explicit_basic.yaml")
def main(cfg: DictConfig) -> None: 

    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    # Load the LLM configuration for dataset creation
    model = cfg.llm_b_model_name
    temperature = cfg.temperature
    total_examples = cfg.total_examples
    examples_per_request = cfg.examples_per_request
    prototype_examples = cfg.prototype_examples

    # Load the dataset configuration info
    env_dataset = Environment(loader=FileSystemLoader(DATASET_TEMPLATE_DIR))
    env_header_labelling = Environment(loader=FileSystemLoader(HEADER_LABELING_TEMPLATE_DIR))
    dataset_templates = {}
    rendered_prompts = {}
    rendered_header_labelling_pairs = {}
    

    # Loop through requested datasets to build prompts
    for dataset_name, dataset_config in cfg.datasets.items():
        dataset_templates[dataset_name] = os.path.splitext(dataset_config.dataset_prompt_template)[0]
        dataset_template = env_dataset.get_template(dataset_config.dataset_prompt_template)
        dataset_prompt = dataset_template.render(**dataset_config.dataset_prompt_template_variables)
        dataset_prompt = dataset_prompt.replace("\n", " ")
        rendered_prompts[dataset_name] = dataset_prompt
        dataset_header_labelling_pairs = {}

        # Loop through additional labels for datasets
        # to be able to label the prompts
        for pair_name, pair_config in dataset_config.header_labelling_pairs.items():
            pair_template = env_header_labelling.get_template(pair_config.hl_prompt_template)
            pair_prompt = pair_template.render(**pair_config.hl_prompt_template_variables)
            pair_prompt = pair_prompt.replace("\n", " ")
            dataset_header_labelling_pairs[pair_name] = pair_prompt

        rendered_header_labelling_pairs[dataset_name] = dataset_header_labelling_pairs

    print("\nDataset Templates:")
    print(dataset_templates)

    print("\nRendered Prompts:")
    print(rendered_prompts)

    print("\nRendered Header Labelling Pairs:")
    print(rendered_header_labelling_pairs)   
    
    dataset_creator = DatasetCreator(DATA_PATH)

    for dataset_name, prompt in rendered_prompts.items():

        # ToDo: Need to send template name to create_output_directories

        dataset_dir = dataset_creator.create_output_directories(dataset_name)
        print(f"\ndataset_dir: {dataset_dir}")

        dataset_name_datetime_stamped = os.path.basename(dataset_dir)
        print(f"\dataset_name_datetime_stamped: {dataset_name_datetime_stamped}")

        test_prompt = dataset_creator.prompt_scaffolding(prompt, examples_per_request)
        print(f"\nTest Prompt: {test_prompt}")

        # Generate the sample dataset

        # First save the prompt as a text file
        with open(dataset_dir+"/test_prompt.txt", 'w', newline='', encoding='utf-8') as file:
            file.write(test_prompt)

        # Define the file path for the generated dataset
        generated_dataset_file_path = os.path.join(dataset_dir, f"{dataset_name_datetime_stamped}")

        # Define the log file path
        log_file_path = os.path.join(dataset_dir, "log")

        # Generate sample dataset
        start_time = time.time()
        dataset_creator.generate_dataset_from_prompt(test_prompt,
                                                     generated_dataset_file_path,
                                                     client,
                                                     model,
                                                     temperature,
                                                     log_file_path,
                                                     0)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"The code took {elapsed_time} seconds to run.")

        # See if the generated dataset is usable?
        user_response = get_user_response()

        if user_response == "y":
        
            # Generate the full dataset if the user is happy with the prototype
            num_iterations = math.ceil(total_examples/examples_per_request)
            start_time = time.time()
            # Generate the dataset
            for i in range(num_iterations):
                logging.info(f"Iteration {i}.")
                dataset_creator.generate_dataset_from_prompt(test_prompt,
                                                            generated_dataset_file_path,
                                                            client,
                                                            model,
                                                            temperature,
                                                            log_file_path,
                                                            i+1)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"The code took {elapsed_time} seconds to run.")

            # Compress the generated datasets into one csv file
            dataset_creator.create_csv_unlabelled(dataset_dir, dataset_name_datetime_stamped)

            dataset_creator.create_csv_labelled(dataset_dir,
                                                dataset_name_datetime_stamped,
                                                rendered_header_labelling_pairs[dataset_name],
                                                client,
                                                model,
                                                temperature)


if __name__ == "__main__":
    main()




