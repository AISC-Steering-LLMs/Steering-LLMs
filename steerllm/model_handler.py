import os
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformer_lens

from omegaconf import DictConfig
from typing import Any

from data_handler import Activation

class ModelHandler:
    """
    Initializes the ModelHandler with the specified configuration and loads the model.
    Includes methods to compute activations and hidden states, and to write the activations/

    Parameters:
    ----------
    config : DictConfig
        A configuration object containing model and experiment settings, including
        whether to use GPU and the name of the model to be loaded.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = self.load_model()



    def load_model(self) -> Any:
        """
        Loads the transformer model specified in the configuration.

        Depending on the configuration, this method either loads a model directly with
        transformer_lens for GPT models or uses Hugging Face's AutoModelForCausalLM and
        AutoTokenizer for other transformer models, wrapping it in a HookedTransformer.

        Returns
        -------
        Any
            The loaded transformer model, wrapped in a HookedTransformer object if necessary.
        """
        if not self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if self.config.model_name.startswith("gpt"):
            model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name,
                                                                       tokenizer=tokenizer,
                                                                       hf_model=hf_model)
        return model



    def compute_activations(self, activations_cache: list[Activation]) -> None:
        """
        Performs a forward pass using the model for each activation object in the list,
        caching the raw activations data.

        Parameters
        ----------
        activations_cache : List[Activation]
            A list of Activation objects for which to compute and cache the model's
            raw activations data.

        Returns
        -------
        None
        """
        for act in activations_cache:
            logits, raw_activations = self.model.run_with_cache(act.prompt)
            act.raw_activations = raw_activations



    def add_numpy_hidden_states(self, activations_cache: list[Activation]) -> None:
        """
        Extracts and stores the hidden states from the raw activations for each Activation
        object in the list.

        Parameters
        ----------
        activations_cache : List[Activation]
            A list of Activation objects with raw activations data. This method extracts
            the hidden states and appends them to each object's hidden_states attribute.
        
        Returns
        -------
        None
        """
        for act in activations_cache:
            raw_act = act.raw_activations
            for hook_name in raw_act:
                if "hook_resid_post" in hook_name:
                    numpy_activation = raw_act[hook_name].cpu().numpy()
                    act.hidden_states.append(numpy_activation[0][-1])



    @staticmethod
    def write_activations_cache(activations_cache: Any, experiment_base_dir: str) -> None:
        """
        Serializes and saves the activations cache to a file.

        Parameters
        ----------
        activations_cache : List[Activation]
            A list of Activation objects to be serialized and saved.
        experiment_base_dir : str
            The base directory where the activations cache file will be saved.

        Outputs
        -------
        Creates a file named "activations_cache.pkl" in the specified directory,
        containing the serialized list of Activation objects.
        """
        filename = os.path.join(experiment_base_dir, "activations_cache.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(activations_cache, f)