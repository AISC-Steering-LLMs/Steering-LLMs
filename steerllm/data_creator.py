# Imports
import pandas as pd
import main
from omegaconf import DictConfig, OmegaConf
import yaml
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose
import ipywidgets as widgets
from IPython.display import display

# For refactored code
# Need to tidy this up and remove duplicates

from data_handler import DataHandler
from data_analyser import DataAnalyzer
from model_handler import ModelHandler

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

# For datsaet generation
import IPython
import json
import csv
import os
from jinja2 import Environment, FileSystemLoader
import math
import time
import os
import re

import yaml
from ipywidgets import widgets, VBox, Button, Checkbox, Text, IntText, FloatText, SelectMultiple, Label

from openai import OpenAI
client = OpenAI()

class DataCreator:
    """
    Create datasets for steering LLM A using another LLM B.

    Suppose you want to "steer" LLM A (say Llama-2-7b) for which
    you have access to the weights to generate text that has
    more of a certain property/concept (say, politeness). This
    class can be used to create a dataset of prompts about the
    property/concept using LLM B (say GPT4) which can then be
    passed though LLM A to build a reprsentation of that
    property/concept. 
    """

    def __init__(self) -> None:
        """
        Initializes the instance with directories for the new dataset.


        Parameters
        ----------

        Returns
        -------
        None
        """

    # Forming the prompt from the template and template material
    def render_template_with_data(self,
                                template_file_path,
                                prompt_context_file_path,
                                dataset_generator_prompt_file_path,
                                num_examples_per_prompt):

        # Set up the environment with the path to the directory containing the template
        env = Environment(loader=FileSystemLoader(os.path.dirname(template_file_path)))

        # Now, get_template should be called with the filename only, not the path
        template = env.get_template(os.path.basename(template_file_path))
        
        # Load the prompt context
        with open(prompt_context_file_path, 'r') as file:
            prompt_construction_options = json.load(file)

        # Update the prompt context example to replace the list with the combined string
        example_text = "\n".join(prompt_construction_options["example"])
        prompt_construction_options["example"] = example_text
        prompt_construction_options["num_examples"] = num_examples_per_prompt

        # Render the template with the prompt_construction_options
        prompt_to_generate_dataset = template.render(prompt_construction_options)

        # Save the prompt to a file
        with open(dataset_generator_prompt_file_path, 'w') as file:
            file.write(prompt_to_generate_dataset)

        # Remove newlines from the prompt and replace with spaces
        prompt_to_generate_dataset = prompt_to_generate_dataset.replace('\n', ' ')

        # Save the prompt to a file
        with open(dataset_generator_prompt_file_path, 'w') as file:
            file.write(prompt_to_generate_dataset)

        return prompt_to_generate_dataset



    # Generate the dataset by calling the OpenAI API
    def generate_dataset_from_prompt(self,
                                    prompt,
                                    generated_dataset_file_path,
                                    model,
                                    log_file_path,
                                    i):
        completion = client.chat.completions.create(
                **{
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
        
        completion_words = completion.choices[0].message.content.strip()

        # cleaned_completion = completion.choices[0].message.content.strip()[3:-3]
        print(" ")
        print(completion_words)
        print(" ")

        # Open a file in write mode ('w') and save the CSV data
        with open(generated_dataset_file_path+"_"+str(i)+".txt", 'w', newline='', encoding='utf-8') as file:
            file.write(completion_words)

        num_words_in_prompt = count_words_in_string(prompt)
        num_words_in_completion = count_words_in_string(completion_words)
        total_words = num_words_in_prompt + num_words_in_completion

        num_tokens_in_prompt = completion.usage.prompt_tokens
        num_tokens_in_completion = completion.usage.completion_tokens
        total_tokens = num_tokens_in_prompt + num_tokens_in_completion

        prompt_cost = num_tokens_in_prompt*0.01/1000
        completion_cost = num_tokens_in_completion*0.03/1000
        total_cost = prompt_cost + completion_cost
        
        tokens_per_prompt_word = num_words_in_prompt/num_tokens_in_prompt
        tokens_per_completion_word = num_words_in_completion/num_tokens_in_completion

        log = {
                "num_words_in_prompt": num_words_in_prompt,
                "num_words_in_completion": num_words_in_completion,
                "total_words": total_words,
                "num_tokens_in_prompt": num_tokens_in_prompt,
                "num_tokens_in_completion": num_tokens_in_completion,
                "total_tokens": total_tokens,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": total_cost,
                "tokens_per_prompt_word": tokens_per_prompt_word,
                "tokens_per_completion_word": tokens_per_completion_word

        }

    for k, v in log.items():
        print(k, v)
    print(" ")

    with open(log_file_path+"_"+str(i)+".txt", 'w') as file:
        file.write(json.dumps(log, indent=4))

def count_words_in_string(input_string):
    words = input_string.split()
    return len(words)

    