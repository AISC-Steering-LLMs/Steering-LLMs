import json
import os
import random
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
import datetime
from omegaconf import DictConfig, OmegaConf
import numpy as np

@dataclass
class Activation:
    # prompt: str
    # labels: list[str]
    # raw_activations: Any = None
    # hidden_states: List[np.ndarray] = None
    prompt: str
    ethical_area: str
    positive: bool
    raw_activations: Any = None
    hidden_states: List[np.ndarray] = None

class DataHandler:
    """
    Handles loading and preprocessing of dataset, management of experiment output directories,
    and serialization of experiment configurations and results.

    Provides utilities to load prompts and related data from CSV/Excel files, create output
    directories for storing experiment results, and save configuration and prompt data for
    reproducibility and record-keeping.
    """

    # Contants for the columns in the prompts CSV
    PROMPT_COLUMN = "prompt"
    ETHICAL_AREA_COLUMN = "ethical_area"
    POS_COLUMN = "ethical_valence"



    def __init__(self, data_path: str) -> None:
        """
        Initializes the DataHandler with a base directory path for data files and outputs.

        Parameters:
        ----------
        data_path : str
            Base directory path where data files are located and output directories will be created.

        Returns:
        -------
        None
        """
        self.data_path = data_path



    def csv_to_dictionary(self, filename: str) -> Dict[str, Any]:
        """
        Loads data from a specified CSV or Excel file into a dictionary with lists per column.

        Parameters
        ----------
        filename : str
            The name of the CSV or Excel file to load, relative to the data_path.

        Returns
        -------
        Dict[str, Any]
            A dictionary where keys are column names and values are lists containing column data.

        Raises
        ------
        ValueError
            If the specified file does not end with '.csv' or '.xlsx'.
        """
        full_path = os.path.join(self.data_path, filename)
        if filename.endswith(".csv"):
            # index_col false is required to get the same behavior as read_excel
            df = pd.read_csv(full_path, delimiter=",", index_col=False)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(full_path)
        else:
            raise ValueError("Expected file ending with .csv or .xlsx")
        return df.to_dict(orient="list")



    def create_output_directories(self) -> Tuple[str, str, str]:
        """
        Creates directories for storing outputs of an experiment run, including subdirectories
        for images and metrics, based on the current UTC timestamp.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        Tuple[str, str, str]
            A tuple containing the paths to the base experiment directory, images directory,
            and metrics directory, respectively.
        """
        current_datetime = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_base_dir = os.path.join(self.data_path, "outputs", current_datetime)
        images_dir = os.path.join(experiment_base_dir, 'images')
        metrics_dir = os.path.join(experiment_base_dir, 'metrics')
        os.makedirs(experiment_base_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(metrics_dir)
        return experiment_base_dir, images_dir, metrics_dir



    def copy_prompts_to_output(self, prompts_dict: Dict[str, Any], prompts_sheet: str, experiment_base_dir: str) -> None:
        """
        Copies the prompts data to the output directory for record-keeping, saving it as a CSV file.

        Parameters
        ----------
        prompts_dict : Dict[str, Any]
            The dictionary containing prompts data to be saved.
        prompts_sheet : str
            The name of the original prompts file, used to derive the output filename.
        experiment_base_dir : str
            The base directory path for the current experiment's outputs.

        Returns
        -------
        None
        """
        input_file_stem = os.path.splitext(os.path.basename(prompts_sheet))[0]
        prompts_output_path = os.path.join(experiment_base_dir, f"{input_file_stem}.csv")
        prompts_df = pd.DataFrame(prompts_dict)
        prompts_df.to_csv(prompts_output_path)



    def write_experiment_parameters(self, cfg: DictConfig, prompts_dict: Dict[str, Any], experiment_base_dir: str) -> None:
        """
        Writes the experiment's configuration parameters and prompts data to the output directory.
        This acts as a record of the configuration used for a given run, and the prompts data used.

        Parameters
        ----------
        cfg : DictConfig
            The configuration object containing experiment settings.
        prompts_dict : Dict[str, Any]
            The dictionary containing prompts data.
        experiment_base_dir : str
            The base directory path for the current experiment's outputs.

        Returns
        -------
        None
        """
        with open(os.path.join(experiment_base_dir, "config.yaml"), 'w') as f:
            OmegaConf.save(cfg, f)
        self.copy_prompts_to_output(prompts_dict, cfg.prompts_sheet, experiment_base_dir)



    def populate_data(self, prompts_dict: Dict[str, Any]) -> List[Activation]:
        """
        Populates a list of Activation objects from a dictionary containing prompts and associated data.

        This method iterates over the prompts dictionary, extracting prompt texts, ethical areas,
        and ethical valence indicators to create Activation objects. Each object represents a distinct
        data point with its associated metadata and is initially populated without raw activations
        or hidden states, which can be added later during model processing.

        Parameters
        ----------
        prompts_dict : Dict[str, Any]
            A dictionary where keys correspond to column names (prompts ethical area, ethical valance)
            and values are lists of entries for each column. This structure is typically obtained
            from loading a CSV or Excel file with prompts and their associated metadata.

        Returns
        -------
        List[Activation]
            A list of Activation objects populated with the data from prompts_dict. Each Activation
            object contains the prompt text, ethical area classification, ethical valance indicator, and
            placeholders for raw activations and hidden states.
        """
        activations_cache = []
        for prompt, ethical_area, positive in zip(prompts_dict[DataHandler.PROMPT_COLUMN],
                                                prompts_dict[DataHandler.ETHICAL_AREA_COLUMN],
                                                prompts_dict[DataHandler.POS_COLUMN]):
            act = Activation(prompt, ethical_area, bool(positive), None, [])
            activations_cache.append(act)
        return activations_cache



    def transform_to_steering_dataset(self, prompts_dict: Dict[str, Any], user_tag='', assistant_tag='', seed=0):
        random.seed(seed)

        template_str = '{user_tag} Consider the {concept} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
        concepts = set(prompts_dict[DataHandler.ETHICAL_AREA_COLUMN])

        raw_data = list(zip(prompts_dict[DataHandler.PROMPT_COLUMN], prompts_dict[DataHandler.ETHICAL_AREA_COLUMN], prompts_dict[DataHandler.POS_COLUMN]))

        formatted_data = {}
        for concept in concepts:
            format_str = lambda x: template_str.format(concept=concept, scenario=x, user_tag=user_tag, assistant_tag=assistant_tag)
            current = [Activation(format_str(prompt), concept, True, raw_activations=[], hidden_states=[]) for (prompt, c, _) in raw_data if c == concept]
            other = [Activation(format_str(prompt), concept, False, [], []) for (prompt, c, _) in raw_data if c != concept]
            random.shuffle(other)

            data = [[c,o] for c,o in zip(current, other)]


            train_labels = []
            for d in data:
                true_s = d[0]
                random.shuffle(d)
                train_labels.append([s == true_s for s in d])
            
            concept_train_data = np.concatenate(data).tolist()
            concept_test_data = np.concatenate([[c,o] for c,o in zip(current, other)]).tolist()


            

            formatted_data[concept] = {
                'train': {'data': concept_train_data, 'labels': train_labels},
                'test': {'data': concept_test_data, 'labels': [[1,0]* len(concept_test_data)]}
            }


        return formatted_data