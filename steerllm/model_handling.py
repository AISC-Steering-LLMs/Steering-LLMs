import os
import gc
from functools import partial
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformer_lens.utils as utils
import transformer_lens
from tqdm import tqdm
from omegaconf import DictConfig
from typing import Any


from data_handler import Activation


class ModelHandler:
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = self.load_model()


    def save_residual_hook(self, act, output, hook):
        """
        This hook function saves the output (residual stream) of each layer to disk.
        It generates a unique filename for each layer based on its order.
        """
        act.hidden_states.append(output.cpu().numpy()[0][-1])



    def load_model(self):
        """Load the model specified in the configuration."""

        torch.set_grad_enabled(False)

        device = "cpu" if not self.config.use_gpu else utils.get_device()
        # if not self.config.use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if self.config.model_name.startswith("gpt"):
            model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name, device=device)
        elif self.config.model_name.startswith("meta-llama") and self.config.use_gpu == True:
            hf_model = AutoModelForCausalLM.from_pretrained(self.config.model_name, load_in_4bit=True, torch_dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name,
                                                         hf_model=hf_model,
                                                         torch_dtype=torch.float32,
                                                         fold_ln=False,
                                                         fold_value_biases=False,
                                                         center_writing_weights=False,
                                                         center_unembed=False,
                                                         tokenizer=tokenizer)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name,
                                                         hf_model=hf_model,
                                                         tokenizer=tokenizer, device=device)
        gc.collect()
        gc.collect() # Running twice because of weird timing reasons.
        return model


    def compute_activations(self, activations_cache: list[Activation]) -> None:
        for act in tqdm(activations_cache, desc="Computing activations"):
            pattern_hook_names_filter = lambda name: name.startswith("blocks") and name.endswith("hook_resid_post")
            save_act = partial(self.save_residual_hook, act)
            tokens = self.model.to_tokens(act.prompt)
            self.model.run_with_hooks(
                tokens,
                return_type=None,
                fwd_hooks=[(
                    pattern_hook_names_filter,
                    save_act
                )]
            )

    @staticmethod
    def write_activations_cache(activations_cache: Any, experiment_base_dir: str):
        """Write cache data to a file."""
        filename = os.path.join(experiment_base_dir, "activations_cache.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(activations_cache, f)
