from typing import Any, Dict, List
import numpy as np
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import islice
from model_handler import ModelHandler


class RepReader():
    """Base class for reading representations"""
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def recenter(self, x, mean=None):
        x = torch.Tensor(x)
        if mean is None:
            mean = torch.mean(x,axis=0,keepdims=True)
        else:
            mean = torch.Tensor(mean)
        return x - mean

    def project_onto_direction(self, H, direction):
        """Project matrix H (n, d_1) onto direction vector (d_2,)"""
        # Calculate the magnitude of the direction vector
        # Ensure H and direction are on the same device (CPU or GPU)
        if type(direction) != torch.Tensor:
            H = torch.Tensor(H)
        if type(direction) != torch.Tensor:
            direction = torch.Tensor(direction)
            direction = direction.to(H.device)
        mag = torch.norm(direction)
        assert not torch.isinf(mag).any()
        # Calculate the projection
        projection = H.matmul(direction) / mag
        return projection





class PCARepReader(RepReader):
    """Extract directions via PCA"""
    needs_hiddens = True 

    def __init__(self, model_handler: ModelHandler, n_components: int = 1):
        super().__init__(model_handler)
        self.direction_method = 'pca'
        self.directions = None # directions accessible via directions[layer][component_index]
        self.direction_signs = None # direction of high concept scores (mapping min/max to high/low)
        self.n_components = n_components
        self.H_train_means = {}

    def get_rep_directions(self, hidden_states: Dict[str, np.ndarray], hidden_layers: List[str], **kwargs) -> Dict[str, np.ndarray]:
        """Get PCA components for each layer"""
        directions = {}

        for layer in hidden_layers:
            H_train = hidden_states[layer]
            H_train_mean = H_train.mean(axis=0, keepdims=False)
            self.H_train_means[layer] = H_train_mean
            H_train = self.recenter(H_train, mean=H_train_mean).cpu()
            H_train = np.vstack(H_train)
            pca_model = PCA(n_components=self.n_components, whiten=False).fit(H_train)

            directions[layer] = pca_model.components_ # shape (n_components, n_features)
            self.n_components = pca_model.n_components_
        
        return directions

    def get_signs(self, hidden_states: Dict[str, np.ndarray], train_labels: List[List[bool]], hidden_layers: List[str]) -> Dict[str, np.ndarray]:
        """
        Determines the sign of PCA components for each layer based on training labels.

        This method calculates the sign of each PCA component by comparing the mean of the maximum and minimum PCA outputs
        for training labels. The sign indicates the direction of high concept scores, with a default to positive in case
        of a tie (when the difference is zero).

        Args:
            hidden_states (dict): A dictionary where keys are layer names and values are the hidden states for those layers.
            train_labels (list): A list of training labels, where each label corresponds to a hidden state.
            hidden_layers (list): A list of layer names for which to calculate the PCA component signs.

        Returns:
            dict: A dictionary where keys are layer names and values are arrays indicating the sign of each PCA component
                for that layer. Positive values indicate a direction of high concept scores, and negative values indicate
                the opposite. In case of a tie, the sign defaults to positive.
        """
        signs = {}

        for layer in hidden_layers:
            assert hidden_states[layer].shape[0] == len(np.concatenate(train_labels)), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})"
            layer_hidden_states = hidden_states[layer]

            # NOTE: since scoring is ultimately comparative, the effect of this is moot
            layer_hidden_states = self.recenter(layer_hidden_states, mean=self.H_train_means[layer])

            # get the signs for each component
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):

                transformed_hidden_states = self.project_onto_direction(layer_hidden_states, self.directions[layer][component_index]).cpu()
                
                pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))]

                # We do elements instead of argmin/max because sometimes we pad random choices in training
                pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)])
                pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])

       
                layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
                if layer_signs[component_index] == 0:
                    layer_signs[component_index] = 1 # default to positive in case of tie

            signs[layer] = layer_signs

        return signs

    def transform(self, hidden_states: Dict[str, np.ndarray], hidden_layers: List[str], component_index: int) -> Dict[str, np.ndarray]:
        """Project the hidden states onto the concept directions in self.directions

        Args:
            hidden_states: dictionary with entries of dimension (n_examples, hidden_size)
            hidden_layers: list of layers to consider
            component_index: index of the component to use from self.directions

        Returns:
            transformed_hidden_states: dictionary with entries of dimension (n_examples,)
        """

        assert component_index < self.n_components
        transformed_hidden_states = {}
        for layer in hidden_layers:
            layer_hidden_states = hidden_states[layer]

            if hasattr(self, 'H_train_means'):
                layer_hidden_states = self.recenter(layer_hidden_states, mean=self.H_train_means[layer])

            # project hidden states onto found concept directions (e.g. onto PCA comp 0) 
            H_transformed = self.project_onto_direction(layer_hidden_states, self.directions[layer][component_index])
            transformed_hidden_states[layer] = H_transformed.cpu().numpy()       
        return transformed_hidden_states