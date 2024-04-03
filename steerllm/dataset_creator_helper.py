import os
import json
import math
import time

from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

import csv
import json
import math
import os
import re
import time
from IPython import get_ipython
from IPython.display import display, HTML, clear_output
from ipywidgets import (
    Button,
    Checkbox,
    Dropdown,
    FloatText,
    HBox,
    IntText,
    Label,
    Layout,
    Output,
    SelectMultiple,
    Text,
    Textarea,
    VBox,
    interactive,
    interact,
    interact_manual,
    fixed,
    widgets,
)
from jinja2 import Environment, FileSystemLoader, meta
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
import pandas as pd
import yaml

        


class NotebookHelper:
    def __init__(self, api_key=None, template_dir='../data/inputs/templates', output_dir='../data/inputs/prompts'):
        
        print(api_key)
        if api_key == "your_openai_api_key_here":
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            self.api_key = api_key

        print(f"Using api_key {self.api_key[:3]}... Update below if incorrect.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_options = ['gpt-4-0125-preview', 'gpt-4', 'gpt-3.5-turbo']
        self.model = 'gpt-4'
        self.temperature = 1.0
        self.filename = 'justice'
        self.total_examples = 100
        self.examples_per_request = 100
        
        # Jinja environment setup
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
    
    def save_api_key(self, api_key):
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        print(f"API key \"{self.api_key[:3]}...\" saved to environment variables.")
    
    def set_model(self, model):
        if model in self.model_options:
            self.model = model
        else:
            raise ValueError(f"Model {model} is not in the available options.")
    
    def set_temperature(self, temperature):
        if 0.0 <= temperature <= 2.0:
            self.temperature = temperature
        else:
            raise ValueError("Temperature must be between 0.0 and 2.0.")
    
    def set_filename(self, filename):
        self.filename = filename
    
    def set_total_examples(self, total):
        self.total_examples = total
    
    def set_examples_per_request(self, examples):
        self.examples_per_request = examples
        
    def reset_values(self):

        default_values = {
            'model': 'gpt-4',
            'temperature': 1.0,
            'filename': 'new_template',
            'total_examples': 100,
            'examples_per_request': 100
        }
        self.update_settings(**default_values)
        return default_values

    def update_settings(self, **kwargs):
        """Update internal values based on user inputs."""
        for key, value in kwargs.items():
            if key == 'model' and value in self.model_options:
                self.model = value
            elif key == 'temperature':
                self.set_temperature(value)
            elif key == 'filename':
                self.filename = value
            elif key == 'total_examples':
                self.total_examples = value
            elif key == 'examples_per_request':
                self.examples_per_request = value

    def print_settings(self):
        print("Entered information:")
        print(f"API Key: {self.api_key[:3]}...")
        print(f"OpenAI Model: {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Filename: {self.filename}.csv")
        print(f"Total Number of Examples: {self.total_examples}")
        print(f"Examples per Request: {self.examples_per_request}")
        print("----------")



    #######################
    ## Template Creation ##
    #######################
        
    def load_templates(self):
        """Load template files from the template directory."""
        import os
        return [os.path.splitext(f)[0] for f in os.listdir(self.template_dir) if f.endswith('.j2')]
    
    def render_template(self, template_name, user_input):
        """Render the template with the provided user input."""
        template = self.env.get_template(template_name)
        return template.render(**user_input)
    
    def save_template_content(self, file_path, content, overwrite=False):
        """Save the template content to the specified file path."""
        import os
        if not overwrite and os.path.exists(file_path):
            print("Doesn't have permission to overwrite existing file.")
            return False
        with open(file_path, 'w') as f:
            f.write(content)
            # print(f"Text saved as {file_path}")
        return True

    def save_rendered_content(self, file_path, rendered_text, overwrite=False):
        """Save the rendered text to the specified file path."""
        return self.save_template_content(file_path, rendered_text, overwrite)
    
    def load_template_content(self, change):
        """Load the content of the selected template into the template content input."""
        template_name = change['new'] + '.j2'
        template_path = os.path.join(self.template_dir, template_name)
        with open(template_path, 'r') as f:
            template_content = f.read()
        return template_content

    def save_template(self, content, filename):
        """Save the template content to a file."""
        # Replace any file extension from the text, if existent, with .j2
        new_filename = filename.split('.')[0] + '.j2'
        new_template_path = os.path.join(self.template_dir, new_filename)

        if new_filename == "blank_template.j2":
                return f'Cannot overwrite the "blank_template.j2" file.'
        elif os.path.exists(new_template_path):
            return f'File "{new_filename}" already exists. Do you want to overwrite it?'
        else:
            self.save_template_content(new_template_path, content)
            return None
           
    def use_template(self, template_name):
        """Use the selected template and prompt the user to fill in the template variables."""
        template_source = self.env.loader.get_source(self.env, template_name)[0]
        parsed_content = self.env.parse(template_source)
        variables = meta.find_undeclared_variables(parsed_content)
        return variables

    def on_render_and_save(self, button, placeholders, output_filename_input, template_name):
        """Render the template with the provided user input and globally save the rendered text."""

        placeholder_values = {var: placeholder.value for var, placeholder in placeholders.items()}        
        output_filename = os.path.splitext(output_filename_input)[0] + '.txt'

        rendered_text = self.render_and_save(template_name, placeholder_values, output_filename)

        if rendered_text is not None:
            # Assign the rendered_text to a variable in the global scope using globals()
            print(f"Rendered Prompt: {rendered_text}")
            globals()['rendered_prompt'] = rendered_text

    def render_and_save(self, template_name, placeholder_values, output_filename):
        """Render the template with the provided user input and save the rendered text to a file."""

        rendered_text = self.render_template(template_name, placeholder_values)
        output_path = os.path.join(self.output_dir, output_filename)

        if os.path.exists(output_path):
            print(f'File "{output_filename}.txt" already exists.') # TODO: Add overwrite option functionality
            return None
        else:
            self.save_rendered_content(output_path, rendered_text)
        return rendered_text
    

    

class DatasetGenerator:
    def __init__(self, output_dir='../data/inputs/labels', dataset_dir='../data/inputs/datasets'):
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir

    def load_hl_files(self):
        """Load heading-labelling scheme files."""
        return [''] + [os.path.splitext(f)[0] for f in os.listdir(self.output_dir) if f.endswith('.json')]

    def save_dictionary(self, hl_pairs, filename):
        """Save heading-labelling pairs to a file."""
        if not filename.endswith('.json'):
            filename += '.json'
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w') as file:
            json.dump(hl_pairs, file)

    def load_all_hl_pairs(self):
        """Load all heading-labelling pairs from existing schemes."""
        all_hl_pairs = []
        for file in os.listdir(self.output_dir):
            if file.endswith('.json'):
                file_path = os.path.join(self.output_dir, file)
                with open(file_path, 'r') as f:
                    hl_pairs = json.load(f)
                    for heading, labelling in hl_pairs.items():
                        all_hl_pairs.append((f"{heading} - {labelling}", (heading, labelling)))
        return all_hl_pairs


class DataGenerator:
    def __init__(self, client, model, temperature):
        self.client = client
        self.model = model
        self.temperature = temperature
    
    def generate_dataset_from_prompt(self, prompt, filestem, output_dir, num_examples, examples_per_request):
        dataset_dir = os.path.join(output_dir, filestem)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"Directory created: {dataset_dir}")
        else:
            print(f"Directory already exists: {dataset_dir}")
        
        generated_dataset_file_path = os.path.join(dataset_dir, f"{filestem}")
        log_file_path = os.path.join(dataset_dir, "log")
        num_iterations = math.ceil(num_examples / examples_per_request)

        for i in range(num_iterations):
            self._generate_batch(prompt, generated_dataset_file_path, log_file_path, i)
    
    def _generate_batch(self, prompt, generated_dataset_file_path, log_file_path, i):
        completion = self.client.chat.completions.create(
                **{
                    "model": self.model,
                    "temperature": self.temperature,
                    "seed": i,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
        # TODO: 
        # # Extract and process completion text, then save to file and log
        # This is a simplified representation; adjust according to your specific logic