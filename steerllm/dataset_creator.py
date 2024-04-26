# Imports

import json
import os
import os
import datetime
import re
import csv

from openai import OpenAI
client = OpenAI()

class DatasetCreator:
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

    def __init__(self, data_path: str, current_datetime: str) -> None:
        """
        Initializes the instance with directories for the new dataset.


        Parameters
        ----------

        Returns
        -------
        None
        """
        self.data_path = data_path
        self.current_datetime = current_datetime



    def create_output_directories(self, template_name, dataset_name) -> str:
        """
        Creates directories for storing outputs of a dataset creation run.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        str
            A str containing the path to the dataset.
        """
        dataset_dir = os.path.join(self.data_path,
                                   "inputs/datasets/",
                                   template_name,
                                   self.current_datetime,
                                   dataset_name)
        os.makedirs(dataset_dir, exist_ok=True) 
        return dataset_dir


    def prompt_scaffolding(self, prompt, examples_per_request):
        """
        Scaffold the prompt for the dataset.

        Parameters
        ----------
        prompt : str
            The prompt to be scaffolded.
        examples_per_request : int
            The number of examples to be generated per request.

        Returns
        -------
        str
            The scaffolded prompt.
        """

        scaffolded_prompt = (f"Repeat the following instruction {examples_per_request} times, "
                            "always generating a unique answer to the instruction. "
                            f"Begin instruction: {prompt} End instruction. "
                            "Put the result of each instruction within a pair quote marks on a new line "
                            "as if each was the row of a single column csv and include no other text.")
        
        return scaffolded_prompt

 
    def generate_dataset_from_prompt(self,
                                    prompt,
                                    generated_dataset_file_path,
                                    client,
                                    model,
                                    temperature,
                                    log_file_path,
                                    i):
        
        completion = client.chat.completions.create(
                **{
                    "model": model,
                    "temperature": temperature,
                    "seed": i,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
        
        completion_words = completion.choices[0].message.content.strip()

        print(" ")
        print(completion_words)
        print(" ")

        # Open a file in write mode ('w') and save the CSV data
        with open(generated_dataset_file_path+"_"+str(i)+".txt", 'w', newline='', encoding='utf-8') as file:
            file.write(completion_words)

        num_words_in_prompt = self.count_words_in_string(prompt)
        num_words_in_completion = self.count_words_in_string(completion_words)
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



    def count_words_in_string(self, input_string):
        words = input_string.split()
        return len(words)



    def create_csv_unlabelled(self, dataset_dir, dataset_name):
        # Get a list of all the files you want to process
        # Makes sure they all have the same prompt context
        # eg won;t mix up honest with justice etc
        print(f"\ndataset_dir: {dataset_dir}")
        print(f"\ndataset_name: {dataset_name}")
        files = [f for f in os.listdir(dataset_dir) 
                            if f.endswith('.txt') and dataset_name in f.lower()]

        print(files)

        # Define the regular expression pattern
        # Get lines that start with a quote,
        # then have any number of characters,
        # then end with a quote and possibly comma
        # We're trying to find all valid CSV lines
        pattern = r'^\".*\",?[\r\n]*'

        # Open the master CSV file
        with open(os.path.join(dataset_dir, dataset_name+"_unlabelled.csv"), "a") as master:
            # Loop over the files
            for file in files:
                # Open the current file and read its contents
                with open(os.path.join(dataset_dir, file), 'r') as f:
                    content = f.read()

                # Use the re.findall function to find all matches in the content
                matches = re.findall(pattern, content, re.MULTILINE)        

                # Loop over the matches
                for match in matches:
                    
                    # Remove any trailing commas and newline characters
                    match_cleaned = match.rstrip(',\r\n')
                    
                    # Append the match to the master CSV file
                    master.write(match_cleaned + '\n')



    def ask_openai(self, client, model, temperature, prompt):
        completion = client.chat.completions.create(
                    **{
                        "model": model,
                        "temperature": temperature, # For most deterministic results
                        "seed": 0, # For reproducibility
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )
        return completion.choices[0].message.content.strip()



    def create_csv_labelled(self,
                            dataset_dir,
                            dataset_name_datetime_stamped,
                            hl_pairs,
                            client,
                            model,
                            temperature):

        input_file_path = os.path.join(dataset_dir, f"{dataset_name_datetime_stamped}_unlabelled.csv")
        output_file_path = os.path.join(dataset_dir, f"{dataset_name_datetime_stamped}_labelled.csv")

        with open(input_file_path, mode='r', newline='', encoding='utf-8') as infile, \
            open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Begin header row. 1st col always "prompt"
            header = ["prompt"]

            # Add a header
            for column_name in hl_pairs.keys():
                header.append(column_name)
            
            writer.writerow(header)
            
            for i, row in enumerate(reader):
                print(f"\nRow {i+1}: ")
                for labelling in hl_pairs.values():
                    # Assuming each row contains a single column with your text
                    statement = row[0]  # Adjust this if your structure is different
                    # Here you define the question you want to ask about each row
                    labelling_prompt = f"{labelling} {statement}"
                    print(labelling_prompt)
                    response = self.ask_openai(client,
                                                model,
                                                temperature,
                                                labelling_prompt)
                    # Add the OpenAI response to the row
                    row.append(response)
                writer.writerow(row)

    


