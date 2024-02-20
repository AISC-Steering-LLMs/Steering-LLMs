from typing import List, Union, Optional
import matplotlib
import os
from omegaconf import DictConfig
import hydra
import torch
import numpy as np
import random

from tqdm import tqdm

from data_handler import DataHandler, Activation
from data_analyser import DataAnalyzer
from model_handler import ModelHandler

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from itertools import islice


from pca_repreader import PCARepReader
from data_handler import DataHandler

class SteeringHandler:
    def __init__(self, config: DictConfig, model_handler: ModelHandler, data_handler: DataHandler):

            self.config = config
            self.user_tag = ["INST"]
            self.assistant_tag = ["/INST"]
            self.model_handler = model_handler
            self.data_handler = data_handler
            self.directions = None
            self.direction_signs = None
            self.layer_ids = list(range(-11, -30, -1))
            self.rep_token = -1
            self.direction_method = 'pca'
            self.n_components = 1


    def compute_directions(self, prompts_dict, rep_token):
        """
        Wrapper method to get a dictionary of hidden states from a list of strings using the ModelHandler's compute_activations method.
        """
        data = self.data_handler.transform_to_steering_dataset(prompts_dict=prompts_dict, user_tag=self.user_tag, assistant_tag=self.assistant_tag, seed=0)
        concepts = set(prompts_dict[DataHandler.ETHICAL_AREA_COLUMN])
        hidden_layers = self.model_handler.get_hidden_layers()

        concept_H_tests = {}
        concept_rep_readers = {}

        for concept in tqdm(concepts, desc=f"Computing Reading Directions"):
            train_data = data[concept]['train']
            test_data = data[concept]['test']

            rep_reader = self.get_directions(
                train_inputs=train_data,
                rep_token=rep_token, 
                hidden_layers=hidden_layers, 
            )


            hidden_states_test = self.batched_string_to_hiddens(
                test_data,
                rep_token, 
                hidden_layers, 
                self.model_handler, 
                )


            transformed_states = rep_reader.transform(hidden_states_test, hidden_layers, 0)
            transformed_states = self.reshape_states(transformed_states)

            concept_H_tests[concept] = transformed_states
            concept_rep_readers[concept] = rep_reader

        return concept_H_tests, concept_rep_readers


    def batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, model_handler, **tokenizer_args):
        """
        Wrapper method to get a dictionary of hidden states from a list of strings using the ModelHandler's compute_activations method.
        """

        # Compute activations for all inputs
        model_handler.compute_activations(train_inputs['data'])

        hidden_states = {layer: [] for layer in hidden_layers}
        for act in train_inputs['data']:
            for layer in hidden_layers:
                hidden_states[layer].append(act.hidden_states[layer])

        # Stack hidden states for each layer
        for layer in hidden_states:
            hidden_states[layer] = np.vstack(hidden_states[layer])

        return hidden_states

    def reshape_states(self,transformed_states):
        """
        Reorganizes the transformed_states dictionary to have an array with a dictionary for each example,
        and one value per layer.

        Parameters
        ----------
        transformed_states : dict
            A dictionary where keys are layer indices and values are lists of transformed states for each example.

        Returns
        -------
        list[dict]
            A list where each element is a dictionary with layer indices as keys and the transformed state as the value.
        """
        # Number of examples is the length of the list for any layer
        num_examples = len(next(iter(transformed_states.values())))
        reorganized = []

        for i in range(num_examples):
            example_dict = {layer: transformed_states[layer][i] for layer in transformed_states}
            reorganized.append(example_dict)

        return reorganized

    def get_directions(
            self,
            train_inputs: List[List[Activation]],
            rep_token: Union[str, int]=-1, 
            hidden_layers: Union[str, int]=-1,
            ):
        """Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """

        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
        
        # self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder
        direction_finder = PCARepReader(model_handler=self.model_handler)

        # if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        # get raw hidden states for the train inputs
        hidden_states = self.batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, self.model_handler)

        # get differences between pairs
        relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
        for layer in hidden_layers:
            relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]


        # get the directions
        train_labels = train_inputs['labels']
        direction_finder.directions = direction_finder.get_rep_directions(
            relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
            hidden_states, train_labels, hidden_layers)
        
        return direction_finder