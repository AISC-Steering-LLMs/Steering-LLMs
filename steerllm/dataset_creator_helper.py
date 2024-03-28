import os
import json
import math
import time

from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
        


class NotebookHelper:
    def __init__(self, api_key=None, template_dir='../data/inputs/templates', output_dir='../data/inputs/prompts'):
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
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
        print("API key saved to environment variables.")
    
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
        self.model = 'gpt-4'
        self.temperature = 1.0
        self.filename = 'justice'
        self.total_examples = 100
        self.examples_per_request = self.total_examples
        print("Values reset to default.")

    def update_settings(self, model=None, temperature=None, filename=None, total_examples=None, examples_per_request=None):
        """Update NotebookHelper settings based on user inputs."""
        if model is not None and model in self.model_options:
            self.model = model
        if temperature is not None:
            self.set_temperature(temperature)  # Assuming set_temperature validates the input
        if filename is not None:
            self.filename = filename
        if total_examples is not None:
            self.total_examples = total_examples
        if examples_per_request is not None:
            self.examples_per_request = examples_per_request

    def print_settings(self):
        print("Entered information:")
        print(f"Client: {self.client}")
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
            print(f"Text saved as {file_path}")
        return True

    def save_rendered_content(self, file_path, rendered_text, overwrite=False):
        """Save the rendered text to the specified file path."""
        return self.save_template_content(file_path, rendered_text, overwrite)
    

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