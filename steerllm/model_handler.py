import os
import re
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
    """
    Handles loading and processing of a transformer model specified in the configuration. 
    It facilitates the computation of activations for given inputs, saves these activations, 
    and supports various model configurations, including models optimized for GPU usage.
    """

    def __init__(self, config: DictConfig) -> None:
        """
        Initializes the ModelHandler with a given configuration.

        Parameters
        ----------
        config : DictConfig
            A Hydra DictConfig object containing the model's configuration settings.
        """
        self.config = config
        self.model = self.load_model()


    def save_residual_hook(self, act, output, hook) -> None:
        """
        This hook function saves the output (residual stream) of each layer to disk.
        It generates a unique filename for each layer based on its order.
        
        Parameters
        ----------
        act : Activation
            The Activation object associated with the current input being processed.
        output : torch.Tensor
            The output tensor from the layer to which this hook is attached.
        hook : Any
            The hook reference, not used in this function but required by the hook signature.

        Returns
        -------
        None
        """
        if act.hidden_states == None:
            act.hidden_states = []
        act.hidden_states.append(output.cpu().numpy()[0][-1])



    def load_model(self) -> Any:
        """
        Loads the transformer model based on the configuration provided at initialization.
        Adjusts settings for GPU usage and model-specific configurations.

        Parameters
        ----------
        No parameters.
        
        Returns
        -------
        Any
            The loaded transformer model, ready for generating activations.
        
        """
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
        """
        Computes and caches the model's activations for each input in the provided activations cache.

        Parameters
        ----------
        activations_cache : list[Activation]
            A list of Activation objects, each representing an input for which to compute activations.

        Returns
        -------
        None
        """
        self.reset_activations(activations_cache)
        for act in tqdm(activations_cache, desc="Computing activations", disable=True):
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


    def compute_continuation(self, max_new_tokens=256, input=""):
        """
        Computes the continuation of a given input text up to a maximum number of new tokens.

        This method generates text by iteratively predicting the next token based on the input text and
        previously generated tokens. The generation process stops when either the maximum number of new tokens
        is reached or an end-of-sequence token is generated.

        Parameters
        ----------
        max_new_tokens : int, optional
            The maximum number of new tokens to generate, by default 256.
        input : str, optional
            The initial input text to continue from.

        Returns
        -------
        str
            The generated text continuation.
        """

        print("input", input)

        output = ""

        for _ in tqdm(range(max_new_tokens), desc="Computing Continuation"):

            logits = self.model(input+output, return_type="logits")
            next_token = self.model.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
            output += next_token[-1]

            if next_token[-1] in [self.model.tokenizer.bos_token, self.model.tokenizer.eos_token]:
                break

        return output

    def compute_altered_continuation(self, max_new_tokens=256, input="", activations=None, pattern_hook_names_filter=None, act_patching_hook=None):
        print("input", input)

        output = ""
        for _ in tqdm(range(max_new_tokens), desc="Computing Continuation"):

            logits = self.model.run_with_hooks(
                            input+output,
                            return_type="logits",
                            fwd_hooks=[(
                                pattern_hook_names_filter,
                                act_patching_hook
                            )]
                        )

            next_token = self.model.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
            output += next_token[-1]

            if next_token[-1] in [self.model.tokenizer.bos_token, self.model.tokenizer.eos_token]:
                break

        return output



    def reset_activations(self, activations_cache: list[Activation]) -> None:
        """
        Resets the hidden states of each activation in the activations cache.
        
        Parameters
        ----------
        activations_cache : list[Activation]
            A list of Activation objects, each representing an input for which to compute activations.

        Returns
        -------
        None
        """
        for act in activations_cache:
            act.hidden_states = []


    def get_hidden_layers(self) -> list[int]:
        return list(range(-1, -self.model.cfg.n_layers, -1))
        # return list(range(0, self.model.cfg.n_layers, -1))

    @staticmethod
    def write_activations_cache(activations_cache: Any, experiment_base_dir: str) -> None:
        """
        Writes the cached activations data to a file for persistence.

        Parameters
        ----------
        activations_cache : Any
            The cached activations data to be persisted to disk.
        experiment_base_dir : str
            The base directory for the experiment, where the activations cache file will be saved.

        Returns
        -------
        None
        """
        filename = os.path.join(experiment_base_dir, "activations_cache.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(activations_cache, f)
