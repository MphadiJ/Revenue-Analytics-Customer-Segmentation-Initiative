# src/trainer/train_pipeline.py
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import sys
sys.path.append(os.path.abspath(os.path.join('/home/selowa-mphadi/PycharmProjects/pythonProject/kmeans clustering')))

from src.features.build_features import FeatureEngineer
from src.Transformer.Preprocessing import DataPreprocessor


# src/trainer/kmeans_trainer.py
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansTrainer:
    """
    Trainer class for KMeans clustering.
    Fits only on training data and evaluates metrics on both train and test sets.
    """

    def __init__(self, train_df, test_df, k_range=range(3, 9), random_state=42):
        """
        Args:
            train_df (pd.DataFrame): Preprocessed training data.
            test_df (pd.DataFrame): Preprocessed test data.
            k_range (iterable): Range of cluster numbers to evaluate.
            random_state (int): Random seed for reproducibility.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.k_range = k_range
        self.random_state = random_state

        self.best_k = None
        self.best_model = None
        self.best_score = -1
        self.metrics = {}

    def fit(self):
        """
        Fit KMeans for each k in k_range on training data.
        Evaluate Silhouette Score on train and test sets.
        Store best model based on train score.
        """
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(self.train_df)  # Fit only on training

            # Predict clusters
            train_labels = kmeans.predict(self.train_df)
            test_labels = kmeans.predict(self.test_df)

            # Silhouette scores
            train_score = silhouette_score(self.train_df, train_labels)
            test_score = silhouette_score(self.test_df, test_labels)

            self.metrics[k] = {"train_silhouette": train_score, "test_silhouette": test_score}

            print(f"k={k}: Train Silhouette={train_score:.4f}, Test Silhouette={test_score:.4f}")

            # Keep the best model based on train silhouette
            if train_score > self.best_score:
                self.best_score = train_score
                self.best_k = k
                self.best_model = kmeans

        print(f"Best K={self.best_k} with Train Silhouette={self.best_score:.4f}")

    def assign_segments(self):
        """
        Assign clusters to train and test datasets using best model.
        """
        if self.best_model is None:
            raise ValueError("You must call fit() before assigning segments.")

        train_clusters = self.best_model.predict(self.train_df)
        test_clusters = self.best_model.predict(self.test_df)

        train_df_segmented = self.train_df.copy()
        test_df_segmented = self.test_df.copy()

        train_df_segmented['Segment'] = train_clusters
        test_df_segmented['Segment'] = test_clusters

        return train_df_segmented, test_df_segmented

    def save_model(self, path="models/kmeans_best.pkl"):
        """
        Save the best KMeans model.
        """
        if self.best_model is None:
            raise ValueError("No model trained to save.")
        joblib.dump(self.best_model, path)
        print(f"Best KMeans model saved at {path}")
