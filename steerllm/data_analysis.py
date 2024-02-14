# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from tqdm import tqdm
import os

from typing import Any
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


    # Make tsne plot for hidden state of every layer
    def tsne_plot(self, activations_cache: list[Activation]) -> tuple[dict, list, list]:

        # Dictionary to store the embedded data for each layer
        # Key: layer number, Value: embedded data
        # Will be used for classification in the function classifier_battery
        embedded_data_dict = {}

        # Using activations_cache[0] is arbitrary as they all have the same number of layers
        # (12 with GPT-2-small) with representations
        labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]
        prompts = [act.prompt for act in activations_cache]
        for layer in tqdm(range(len(activations_cache[0].hidden_states)), desc="Computing T-SNE Plots"):
            

            data = np.stack([act.hidden_states[layer] for act in activations_cache])


            tsne = TSNE(n_components=2, random_state=self.seed)
            embedded_data = tsne.fit_transform(data)

            embedded_data_dict[layer] = embedded_data
            
            df = pd.DataFrame(embedded_data, columns=["X", "Y"])
            df["Ethical Area"] = labels
            ax = sns.scatterplot(x='X', y='Y', hue='Ethical Area', data=df)
            
            plot_path = os.path.join(self.images_dir, f"tsne_plot_layer_{layer}.png")
            plt.savefig(plot_path)

            plt.close()

        return embedded_data_dict, labels, prompts

    



    # Make PCA plot for hidden state of every layer
    def pca_plot(self, activations_cache: list[Activation]) -> None:
        pca = PCA(n_components=2, random_state=self.seed)
        labels = [f"{act.ethical_area} {act.positive}" for act in activations_cache]
        # Using activations_cache[0] is arbitrary as they all have the same number of layers
        for layer in tqdm(range(len(activations_cache[0].hidden_states)), desc="Computing PCA Plots"):
            
            data = np.stack([act.hidden_states[layer] for act in activations_cache])

            embedded_data = pca.fit_transform(data)
            
            df = pd.DataFrame(embedded_data, columns=["X", "Y"])
            df["Ethical Area"] = labels
            ax = sns.scatterplot(x='X', y='Y', hue='Ethical Area', data=df)
            
            plot_path = os.path.join(self.images_dir, f"pca_plot_layer_{layer}.png")
            plt.savefig(plot_path)

            plt.close()


    # Raster Plot has columns = Neurons and rows = Prompts. One chart for each layer
    def raster_plot(self, activations_cache: list[Activation],
                    compression: int=5) -> None:
        # Using activations_cache[0] is arbitrary as they all have the same number of layers
        for layer in tqdm(range(len(activations_cache[0].hidden_states)), desc="Computing Raster Plots"):
            
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
            plt.savefig(plot_path)

            plt.close()

    def random_projections_plot(self, activations_cache: list[Activation]) -> None:
        
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


            metrics_dict = {"Layer": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

            for layer, representations in tqdm(embedded_data_dict.items(), desc=f"Computing {name}"):
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
