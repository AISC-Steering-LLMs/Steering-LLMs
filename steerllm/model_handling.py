import os
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformer_lens

from omegaconf import DictConfig
from typing import Any

from data_handler import Activation

class ModelHandler:
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = self.load_model()

    def load_model(self):
        """Load the model specified in the configuration."""
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

    def compute_activations(self, activations_cache: list[Activation]):
        """Compute forward pass and cache data for each activation in the list."""
        for act in activations_cache:
            logits, raw_activations = self.model.run_with_cache(act.prompt)
            act.raw_activations = raw_activations

    def add_numpy_hidden_states(self, activations_cache: list[Activation]):
        """Add the hidden states (the weights of the model between attention/MLP blocks)."""
        for act in activations_cache:
            raw_act = act.raw_activations
            for hook_name in raw_act:
                if "hook_resid_post" in hook_name:
                    numpy_activation = raw_act[hook_name].cpu().numpy()
                    act.hidden_states.append(numpy_activation[0][-1])

    @staticmethod
    def write_activations_cache(activations_cache: Any, experiment_base_dir: str):
        """Write cache data to a file."""
        filename = os.path.join(experiment_base_dir, "activations_cache.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(activations_cache, f)