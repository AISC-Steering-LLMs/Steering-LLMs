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
from nnsight import LanguageModel
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
        self.model_keys = {
            "openai-community/gpt2": {
                'model': 'transformer',
                'layers': 'h',
            },
            "meta-llama/Llama-2-7b-chat-hf": {
                'model': 'model',
                'layers': 'layers',
            },
            "meta-llama/Llama-2-7b-hf": {
                'model': 'model',
                'layers': 'layers',
            }
        }

    def get_property_name(self, key: str):
        return self.model_keys[self.model.config._name_or_path][key]


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

        # if self.config.model_name.startswith("gpt"):
        #     model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name, device=device)
        # elif self.config.model_name.startswith("meta-llama") and self.config.use_gpu == True:
        #     hf_model = AutoModelForCausalLM.from_pretrained(self.config.model_name, load_in_4bit=True, torch_dtype=torch.float32)
        #     tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        #     model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name,
        #                                                  hf_model=hf_model,
        #                                                  torch_dtype=torch.float32,
        #                                                  fold_ln=False,
        #                                                  fold_value_biases=False,
        #                                                  center_writing_weights=False,
        #                                                  center_unembed=False,
        #                                                  tokenizer=tokenizer)
        # else:
        #     hf_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        #     tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        #     model = transformer_lens.HookedTransformer.from_pretrained(self.config.model_name,
        #                                                  hf_model=hf_model,
        #                                                  tokenizer=tokenizer, device=device)


        model = LanguageModel(self.config.model_name, device_map=device, torch_dtype=torch.float16)

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


        model_key = self.get_property_name('model')
        layer_key = self.get_property_name('layers')
        # with self.model.trace() as tracer:
        #     outputs = []
        #     for act in tqdm(activations_cache, desc="Computing activations", disable=False):
        #         if act.hidden_states == None:
        #             act.hidden_states = []
        #         with tracer.invoke(act.prompt):
        #             output = [layer.mlp.output[0][:].save() for layer in getattr(getattr(self.model, model_key), layer_key)]
                
        #         outputs.append(output)
                

        # for idx, act in enumerate(activations_cache):
        #     act.hidden_states = [el.value.cpu().numpy()[:][-1] for el in outputs[idx]]


        for act in tqdm(activations_cache, desc="Computing activations", disable=False):
            if act.hidden_states == None:
                act.hidden_states = []
            with self.model.trace(act.prompt):
                output = [layer.mlp.output[0][:].save() for layer in getattr(getattr(self.model, model_key), layer_key)]
            
            output = [el.value.cpu().numpy()[:][-1] for el in output]
            
            act.hidden_states = output




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

        with self.model.generate(input, max_new_tokens=max_new_tokens):
            # self.model.lm_head.output.argmax(dim=-1)
            token_ids = []
            for _ in tqdm(range(max_new_tokens - 1), desc="Computing Continuation"):

                token_ids.append(self.model.lm_head.next().output.argmax(dim=-1).save())
                if token_ids[-1][-1] in [self.model.tokenizer.bos_token, self.model.tokenizer.eos_token]:
                    break

            output = self.model.generator.output.save()

        # print(output.shape)
        # output = self.model.tokenizer.batch_decode(torch.cat(token_ids[1:]))

        return self.model.tokenizer.batch_decode(output)[0][len(input) + 4:]

    def compute_altered_continuation(self, max_new_tokens=256, input="", activations=None):
        # input = "The Eiffel Tower is in the city of"
        model_key = self.get_property_name('model')
        layer_key = self.get_property_name('layers')

        output_tokens = []
        output_str = ""

        # for _ in tqdm(range(max_new_tokens - 1), desc="Computing Altered Continuation"):
        #     _output_tokens = []
        #     with self.model.trace(input + output_str):
        #         # for idx, layer in enumerate(getattr(getattr(self.model, model_key), layer_key)):
        #         #     hidden_states_pre = layer.mlp.output[0][-1]
        #         #     noise = 0.01*torch.randn(hidden_states_pre.shape).to(activations[0].device)
        #         #     layer.mlp.output[0][-1] = hidden_states_pre + noise
                

        #         _result = self.model.lm_head.output.save()
        #         _token_ids = self.model.lm_head.output.argmax(dim=-1).save()

        #     new_token = self.model.tokenizer.decode(_token_ids[0][-1])
        #     output_str += new_token


        # return output_str




        with self.model.generate(input, max_new_tokens=max_new_tokens) as tracer:
            for _ in tqdm(range(max_new_tokens - 1), desc="Computing Altered Continuation"):

                self.model.lm_head.next()
                for idx, layer in enumerate(getattr(getattr(self.model, model_key), layer_key)):
                    hidden_states_pre = layer.mlp.output[0].clone().save()
                    # noise = 0.005*torch.randn(hidden_states_pre.shape).to(activations[0].device)
                    # layer.mlp.output[0] = hidden_states_pre + noise
                    layer.mlp.output[0] = hidden_states_pre + activations[idx]


                    # if torch.any(torch.isnan(output_tensor)): #or torch.any(torch.isinf(output_tensor)): #or torch.any(output_tensor < 0):
                    #     print("Invalid output detected: Contains NaN, Inf, or negative values.")
                

                # output_tokens.append(self.model.lm_head.next().output.argmax(dim=-1).clone().save())
                # if token_ids[0][-1] in [self.model.tokenizer.bos_token, self.model.tokenizer.eos_token]:
                #     break

            output = self.model.generator.output.save()

        # print(output_tokens)
        return self.model.tokenizer.batch_decode(output)[0][len(input) + 4:]
        # return ''.join(self.model.tokenizer.batch_decode(output_tokens))



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
        model_key = self.get_property_name('model')
        layer_key = self.get_property_name('layers')
        return list(range(-1, -len(getattr(getattr(self.model, model_key), layer_key)), -1))

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
