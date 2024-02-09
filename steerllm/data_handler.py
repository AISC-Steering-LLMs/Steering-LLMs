import os
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
import datetime
from omegaconf import DictConfig, OmegaConf
import numpy as np

@dataclass
class Activation:
    prompt: str
    ethical_area: str
    positive: bool
    raw_activations: Any = None
    hidden_states: List[np.ndarray] = None

class DataHandler:

    # Contants for the columns in the prompts CSV
    PROMPT_COLUMN = "Prompt"
    ETHICAL_AREA_COLUMN = "Ethical_Area"
    POS_COLUMN = "Positive"

    def __init__(self, data_path: str):
        self.data_path = data_path

    def csv_to_dictionary(self, filename: str) -> Dict[str, Any]:
        """Load data from a CSV or Excel file into a dictionary."""
        full_path = os.path.join(self.data_path, filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(full_path)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(full_path)
        else:
            raise ValueError("Expected file ending with .csv or .xlsx")
        return df.to_dict(orient="list")

    def create_output_directories(self) -> Tuple[str, str]:
        """Create and return paths for experiment output directories."""
        current_datetime = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_base_dir = os.path.join(self.data_path, "outputs", current_datetime)
        images_dir = os.path.join(experiment_base_dir, 'images')
        metrics_dir = os.path.join(experiment_base_dir, 'metrics')
        os.makedirs(experiment_base_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(metrics_dir)
        return experiment_base_dir, images_dir, metrics_dir

    def copy_prompts_to_output(self, prompts_dict: Dict[str, Any], prompts_sheet: str, experiment_base_dir: str):
        """Copy prompts from input to output directory for record-keeping."""
        input_file_stem = os.path.splitext(os.path.basename(prompts_sheet))[0]
        prompts_output_path = os.path.join(experiment_base_dir, f"{input_file_stem}.csv")
        prompts_df = pd.DataFrame(prompts_dict)
        prompts_df.to_csv(prompts_output_path)

    def write_experiment_parameters(self, cfg: DictConfig, prompts_dict: Dict[str, Any], experiment_base_dir: str):
        """Write configuration parameters and prompts to the output directory."""
        with open(os.path.join(experiment_base_dir, "config.yaml"), 'w') as f:
            OmegaConf.save(cfg, f)
        self.copy_prompts_to_output(prompts_dict, cfg.prompts_sheet, experiment_base_dir)

    def populate_data(self, prompts_dict: Dict[str, Any]) -> List[Activation]:
        activations_cache = []
        for prompt, ethical_area, positive in zip(prompts_dict[DataHandler.PROMPT_COLUMN],
                                                prompts_dict[DataHandler.ETHICAL_AREA_COLUMN],
                                                prompts_dict[DataHandler.POS_COLUMN]):
            act = Activation(prompt, ethical_area, bool(positive), None, [])
            activations_cache.append(act)
        return activations_cache
