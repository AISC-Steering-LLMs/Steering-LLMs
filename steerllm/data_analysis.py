# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os

from typing import Any, List, Tuple, Dict
import logging
from dataclasses import dataclass

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn import cluster
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree

from data_handler import Activation


class AnalysisManager:
    def __init__(self, images_dir: str, seed: int = 42):
        self.images_dir = images_dir
        self.seed = seed



    def plot_embeddings(self, activations_cache: List[Activation], dim_red_model: Any) -> Tuple[Dict[int, np.ndarray], List[str], List[str]]:
        """
        Generates and saves plots from activations using the specified dimensionality
        reduction model to visualize the distribution of different ethical areas in
        the embedded space.

        Iterates over layers, applies the provided dimensionality reduction model for
        2D visualization, and saves scatter plots. Returns dictionaries of embedded data,
        labels, and prompts.

        Parameters
        ----------
        activations_cache : List[Activation]
            List of Activation objects with hidden states and metadata.
        dim_red_model : Any
            A dimensionality reduction model instance with a fit_transform method.

        Returns
        -------
        Tuple containing a dictionary of embedded data by layer, list of labels, and list of prompts.
        """
        embedded_data_dict, all_labels, all_prompts = {}, [], []
        for layer in range(len(activations_cache[0].hidden_states)):
            data = np.stack([act.hidden_states[layer] for act in activations_cache])
            labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]
            prompts = [act.prompt for act in activations_cache]

            # No need to check the method, directly use the passed dim_red_model
            embedded_data = dim_red_model.fit_transform(data)
            embedded_data_dict[layer] = embedded_data

            if not all_labels:
                all_labels.extend(labels)
                all_prompts.extend(prompts)

            # Use a generic plot title incorporating the dim_red_model's class name
            plot_title = type(dim_red_model).__name__
            self._save_plot(embedded_data, labels, layer, plot_title)

        return embedded_data_dict, all_labels, all_prompts



    def _save_plot(self, embedded_data: np.ndarray, labels: List[str], layer: int, plot_title: str) -> None:
        """
        Saves a 2D scatter plot for the specified layer using given embeddings.

        Parameters
        ----------
        embedded_data : np.ndarray
            The 2D transformed data.
        labels : List[str]
            List of labels for each data point.
        layer : int
            The layer number the data corresponds to.
        plot_title : str
            The title of the plot indicating the dimensionality reduction technique used.
        """
        df = pd.DataFrame(embedded_data, columns=["X", "Y"])
        df["Ethical Area"] = labels
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='X', y='Y', hue='Ethical Area', palette='viridis', data=df)
        plot_path = os.path.join(self.images_dir, f"{plot_title.lower()}_plot_layer_{layer}.png")
        plt.savefig(plot_path)
        plt.close()



    # Raster Plot has columns = Neurons and rows = Prompts. One chart for each layer
    def raster_plot(self, activations_cache: list[Activation],
                    compression: int=5) -> None:
        """
        Generates and saves raster plots for neural activations across different layers.

        Each raster plot visualizes the activations of neurons (columns) in response to 
        various prompts (rows) for a specific layer. The function iterates through each 
        layer in the activations cache, creating a chart that displays the neuron 
        activations for all prompts. The plots are saved as SVG files.

        Parameters
        ----------
        activations_cache : list[Activation]
            A list of `Activation` objects, each containing the hidden states for each 
            layer of a model and associated metadata, such as the prompts. It is assumed 
            that all `Activation` objects contain the same number of layers and neuron 
            activations.
        compression : int, optional
            A factor used to adjust the plot size and font scaling. Higher values compress 
            the plot and font size for a more condensed visualization. Default is 5.

        Returns
        -------
        None
        """
        # Using activations_cache[0] is arbitrary as they all have the same number of layers
        for layer in range(len(activations_cache[0].hidden_states)):
            
            data = np.stack([act.hidden_states[layer] for act in activations_cache])
            
            # Set the font size for the plot
            plt.rcParams.update({'font.size': (15 / compression)})

            neurons = activations_cache[0].hidden_states[0].size
            # Create the raster plot
            plt.figure(figsize=((neurons / (16 * compression)),
                                (len(activations_cache) + 2) / (2 * compression)))
            plt.imshow(data, cmap='hot', aspect='auto', interpolation="nearest")
            plt.colorbar(label='Activation')

            plt.xlabel('Neuron')
            plt.ylabel('Prompts')
            plt.title(f'Raster Plot of Neural Activations Layer {layer}')

            # Add Y-tick labels of the prompt
            plt.yticks(range(len(activations_cache)),
                    [activation.prompt for activation in activations_cache],
                    wrap=True)

            plot_path = os.path.join(self.images_dir, f"raster_plot_layer_{layer}.svg")
            print(plot_path)
            plt.savefig(plot_path)

            plt.close()



    def random_projections_plot(self, activations_cache: list[Activation]) -> None:
        """
        Evaluates the minimum dimension for eps-embedding using random projections and
        prints the suggested dimensionality based on the Johnson-Lindenstrauss lemma.
        
        This function iterates over each layer in the activations cache, applying the 
        concept of random projections to the hidden states data. It calculates and logs 
        the minimal embedding dimension needed to preserve the pairwise distances between 
        the data points, within a factor of (1 ± eps), where eps is a small positive number.
        
        Parameters
        ----------
        activations_cache : list[Activation]
            A list of `Activation` objects, each containing the hidden states for each layer 
            of a model, along with associated metadata like ethical area and positivity flag. 
            It is assumed that all `Activation` objects contain the same number of layers.
        
        Returns
        -------
        None
        """
        for layer in range(len(activations_cache[0].hidden_states)):
        
            data = np.stack([act.hidden_states[layer] for act in activations_cache])
            labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]

            """
            Note that the number of dimensions is independent of the original number of features 
            but instead depends on the size of the dataset: the larger the dataset, 
            the higher is the minimal dimensionality of an eps-embedding.
            """
            min_dim = johnson_lindenstrauss_min_dim(len(data), eps=0.1)
            # TODO: Would want to log this to hydra if useful
            print(f"Dataset size is {len(data)} and minimum embedding dimension suggested is {min_dim}")
            # No plot visualized here yet. Not sure if useful.



    # Hierarchical Clustering
    def feature_agglomeration(self, activations_cache: list[Activation]) -> None:
        """
        Applies hierarchical clustering to reduce the dimensionality of neural activations
        for each layer, and visualizes the resulting clusters.

        This function performs feature agglomeration, a form of hierarchical clustering, 
        on the hidden states data from each layer of the model. The method aims to cluster 
        features (i.e., neurons) into a specified number of clusters, effectively reducing 
        the dimensionality of the data. The function then generates a scatter plot for the 
        clustered data for each layer, saving the plots as PNG files.

        Parameters
        ----------
        activations_cache : list[Activation]
            A list of `Activation` objects, each containing the hidden states for each layer 
            of a model and associated metadata, such as the prompts, ethical area, and 
            positivity flag.

        Returns
        -------
        None
        """
        for layer in range(len(activations_cache[0].hidden_states)):

            data = np.stack([act.hidden_states[layer] for act in activations_cache])
            labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]

            agglo = cluster.FeatureAgglomeration(n_clusters=2)
            projected_data = agglo.fit_transform(data)

            df = pd.DataFrame(projected_data, columns=["X", "Y"])
            df["Ethical Area"] = labels
            ax = sns.scatterplot(x='X', y='Y', hue='Ethical Area', data=df)

            plot_path = os.path.join(self.images_dir, f"hier_clustering_layer_{layer}.png")
            plt.savefig(plot_path)
            plt.close()



    def probe_hidden_states(self, activations_cache: list[Activation]) -> None:
        """
        Probes the hidden states of each layer using a decision tree classifier to predict
        the ethical area and positivity based on neuron activations, and visualizes the 
        decision tree.

        For each layer, this function splits the hidden states data into training and testing 
        sets, fits a decision tree classifier to predict the ethical area and positivity flag, 
        and then generates a classification report. It also creates a visualization of the 
        trained decision tree, saving the visualization as a PNG file for each layer.

        Parameters
        ----------
        activations_cache : list[Activation]
            A list of `Activation` objects, each containing the hidden states for each layer 
            of a model and associated metadata, such as the prompts, ethical area, and 
            positivity flag.

        Returns
        -------
        None
        """
        for layer in range(len(activations_cache[0].hidden_states)):

            data = np.stack([act.hidden_states[layer] for act in activations_cache])
            labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]
            unique_labels = sorted(list(set(labels)))

            X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=self.seed)

            clf = DecisionTreeClassifier(random_state=self.seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # TODO: Would want to log to hydra if useful
            print("Classification Report:\n", classification_report(y_test, y_pred))

            # Visualize the decision tree
            plt.figure(figsize=(20, 10))
            tree.plot_tree(clf, filled=True, class_names=unique_labels)
            plt.title("Decision tree for probing task")

            plot_path = os.path.join(self.images_dir, f"decision_tree_probe_layer_{layer}.png")
            plt.savefig(plot_path)
            plt.close()




    def classifier_battery(self, embedded_data_dict, labels, prompts, test_size=0.2) -> None:
        """
        Evaluates a battery of classifiers on the given dataset, generating performance metrics and decision boundary plots.

        Parameters:
        - embedded_data_dict: Dict of layer-indexed feature data.
        - labels: List of labels corresponding to the data.
        - prompts: List of prompts associated with the data.
        - test_size: Proportion of the dataset to include in the test split.

        Outputs:
        - Saves classifier performance metrics and decision boundary plots for each layer.
        """
        classifiers = {
            "logistic_regression": LogisticRegression(),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "svc": SVC(probability=True),
            "knn": KNeighborsClassifier(),
            "gradient_boosting": GradientBoostingClassifier()
        }

        for name, clf in classifiers.items():
            logging.info(f"Evaluating {name}")
            metrics_dict = self.evaluate_classifiers(clf, embedded_data_dict, labels, prompts, test_size, name)
            self.save_metrics(metrics_dict, name)



    def evaluate_classifiers(self, clf, embedded_data_dict, labels, prompts, test_size, name):
        metrics_dict = {"Layer": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}
        for layer, representations in embedded_data_dict.items():
            X_train, X_test, y_train, y_test = train_test_split(representations, labels, test_size=test_size, random_state=self.seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metrics_dict["Layer"].append(layer)
            metrics_dict["Accuracy"].append(accuracy_score(y_test, y_pred))
            metrics_dict["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics_dict["Recall"].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics_dict["F1 Score"].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            self.plot_decision_boundary(clf, representations, labels, prompts, layer, name)
        return metrics_dict



    def save_metrics(self, metrics_dict, name):
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(f"{self.metrics_dir}/metrics_{name}.csv", index=False)



    def plot_decision_boundary(self, clf, representations, labels, prompts, layer, classifier_name):
        """
        Plots the decision boundary of a classifier along with the data points.

        Parameters:
        - clf: The classifier for which to plot the decision boundary.
        - representations: The 2D data points (after dimensionality reduction).
        - labels: The labels for the data points.
        - prompts: The prompts associated with each data point.
        - layer: The layer number being processed.
        - classifier_name: The name of the classifier.
        """
        # Plot decision boundary
        # Note I am going to do the decision boundary plots
        # for the entire dataset (representations), not just the test set
        # because I want to to be able to look at all the data (prompts)
        # to get a sense of which ones are being classified incorrectly
        # with this boundary.
        x_min, x_max = representations[:, 0].min() - 1, representations[:, 0].max() + 1
        y_min, y_max = representations[:, 1].min() - 1, representations[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        # Predict probabilities, if possible
        if hasattr(clf, "predict_proba"):
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        color_map = {'Bad False': 'red', 'Good True': 'green'}  # Define your own color mapping
        colors = [color_map[label] for label in labels]  # Map labels to colors

        # *****************
        # MATPOLIB PLOTTING
        # *****************

        # This code plots a continuum of decision boundaries
        # plt.contourf(xx, yy, Z, alpha=0.4)
        # plt.scatter(representations[:, 0], representations[:, 1], c=colors, s=20, edgecolor='k')

        # This code plots a single decision boundary for the 0.5 threshold
        plt.contour(xx, yy, Z, levels=[0.5], colors='black')
        plt.scatter(representations[:, 0], representations[:, 1], c=colors, edgecolors='k')

        plt.title('Decision boundary for ' + name)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(metrics_dir + "/decision_boundary_" + name + "_" + str(layer) + ".png")
        plt.close()

        # ***************
        # PLOTLY PLOTTING
        # ***************

        hover_text = [f'Prompt: {prompt}<br>Label: {label}' for prompt, label in zip(prompts, labels)]

        # Create the figure
        fig = go.Figure(data=[
            go.Contour(x=xx[0], y=yy[:, 0], z=Z, contours=dict(start=0.5, end=0.5, size=1), line_width=2, showscale=False),
            go.Scatter(x=representations[:, 0], y=representations[:, 1], mode='markers', marker=dict(size=8, color=colors), text=hover_text, hoverinfo='text')
        ])

        # Set the title and axis labels
        fig.update_layout(
            title='Decision boundary for ' + name,
            xaxis_title='Feature 1',
            yaxis_title='Feature 2'
        )

        pio.write_html(fig, metrics_dir + "/decision_boundary_" + name + "_" + str(layer) + ".html")

    
    
    
    
    
    
    
    
    
    
    def classifier_battery(self, embedded_data_dict, labels, prompts, metrics_dir) -> None:

        # Define classifiers
        classifiers = {
            "logistic_regression": LogisticRegression(),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "svc": SVC(probability=True),
            "knn": KNeighborsClassifier(),
            "gradient_boosting": GradientBoostingClassifier()
        }

        for name, clf in classifiers.items():

            print(name)

            metrics_dict = {"Layer": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

            for layer, representations in embedded_data_dict.items():
                print(layer)
                # random_state is set to 42 for reproducibility
                # the same train-test split is used for all classifiers
                X_train, X_test, y_train, y_test = train_test_split(representations, labels, test_size=0.2, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                # confusion = confusion_matrix(y_test, y_pred)

                metrics_dict["Layer"].append(layer)
                metrics_dict["Accuracy"].append(accuracy)
                metrics_dict["Precision"].append(precision)
                metrics_dict["Recall"].append(recall)
                metrics_dict["F1 Score"].append(f1)

                # Plot decision boundary
                # Note I am going to do the decision boundary plots
                # for the entire dataset (representations), not just the test set
                # because I want to to be able to look at all the data (prompts)
                # to get a sense of which ones are being classified incorrectly
                # with this boundary.
                x_min, x_max = representations[:, 0].min() - 1, representations[:, 0].max() + 1
                y_min, y_max = representations[:, 1].min() - 1, representations[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                    np.arange(y_min, y_max, 0.1))

                # Predict probabilities, if possible
                if hasattr(clf, "predict_proba"):
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                else:
                    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                color_map = {'Bad False': 'red', 'Good True': 'green'}  # Define your own color mapping
                colors = [color_map[label] for label in labels]  # Map labels to colors

                # *****************
                # MATPOLIB PLOTTING
                # *****************

                # This code plots a continuum of decision boundaries
                # plt.contourf(xx, yy, Z, alpha=0.4)
                # plt.scatter(representations[:, 0], representations[:, 1], c=colors, s=20, edgecolor='k')

                # This code plots a single decision boundary for the 0.5 threshold
                plt.contour(xx, yy, Z, levels=[0.5], colors='black')
                plt.scatter(representations[:, 0], representations[:, 1], c=colors, edgecolors='k')

                plt.title('Decision boundary for ' + name)
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.savefig(metrics_dir + "/decision_boundary_" + name + "_" + str(layer) + ".png")
                plt.close()

                # ***************
                # PLOTLY PLOTTING
                # ***************

                hover_text = [f'Prompt: {prompt}<br>Label: {label}' for prompt, label in zip(prompts, labels)]

                # Create the figure
                fig = go.Figure(data=[
                    go.Contour(x=xx[0], y=yy[:, 0], z=Z, contours=dict(start=0.5, end=0.5, size=1), line_width=2, showscale=False),
                    go.Scatter(x=representations[:, 0], y=representations[:, 1], mode='markers', marker=dict(size=8, color=colors), text=hover_text, hoverinfo='text')
                ])

                # Set the title and axis labels
                fig.update_layout(
                    title='Decision boundary for ' + name,
                    xaxis_title='Feature 1',
                    yaxis_title='Feature 2'
                )

                pio.write_html(fig, metrics_dir + "/decision_boundary_" + name + "_" + str(layer) + ".html")

            metrics_df = pd.DataFrame(metrics_dict)
            metrics_df.to_csv(metrics_dir + "/metrics_"+ name +'.csv', index=False)