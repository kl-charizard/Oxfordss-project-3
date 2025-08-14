import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import logging
from typing import List

# Create a logger
logger = logging.getLogger(__name__)

class RecommendationService:
    """
    A service for recommending vocabulary based on a content-based filtering
    approach using a KNN model. It loads precomputed words embeddings for efficiency.
    """

    def __init__(self, embeddings_path: str, words_path: str):
        """
        Initializes the service by loading precomputed embeddings and training the KNN model.

        Args:
            df (pd.DataFrame): The preprocessed vocabulary dataset.
            embeddings_path (str): The file path to the precomputed word embeddings (.npy file).
            words_path (str): The file path to the list of words corresponding to the word embeddings (.npy file).
        """
        logger.info("Starting to load precomputed word embeddings and train KNN model...")
        
        self.embeddings = self._load_embeddings(embeddings_path)
        self.all_words = self._load_words(words_path)
        self.word_to_index = {word: i for i, word in enumerate(self.all_words)}
        
        self.knn = self._train_knn()
        
        logger.info("Recommendation service initialized successfully.")

    def _load_embeddings(self, embeddings_path: str) -> np.ndarray:
        """Loads the precomputed embeddings from a .npy file."""
        try:
            embeddings = np.load(embeddings_path)
            logger.info(f"Successfully loaded embeddings from '{embeddings_path}'. Shape: {embeddings.shape}")
            return embeddings
        except FileNotFoundError:
            logger.error(f"Embeddings file not found at '{embeddings_path}'.")
            raise RuntimeError("Failed to load precomputed embeddings.")

    def _load_words(self, words_path: str) -> List[str]:
        """Loads the list of words from a .npy file."""
        try:
            words_array = np.load(words_path, allow_pickle=True)
            words = words_array.tolist()
            logger.info(f"Successfully loaded word list from '{words_path}'. Total words: {len(words)}")
            return words
        except FileNotFoundError:
            logger.error(f"Words file not found at '{words_path}'.")
            raise RuntimeError("Failed to load word list.")

    def _train_knn(self) -> NearestNeighbors:
        """Trains the Nearest Neighbors model on the loaded embeddings."""
        knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
        knn_model.fit(self.embeddings)
        logger.info("KNN model trained successfully.")
        return knn_model

    def recommend_words_for_topic(self, topic: str, num_recommendations: int = 10) -> List[str]:
        """
        Finds the words most similar to a given topic using the trained KNN model.
        
        Args:
            topic (str): The topic word to find recommendations for.
            num_recommendations (int): The number of words to recommend.
            
        Returns:
            List[str]: A list of recommended words.
        """
        topic_norm = topic.strip().lower()

        if topic_norm not in self.word_to_index:
            logger.error(f"Topic '{topic}' not found in the vocabulary.")
            return []

        topic_index = self.word_to_index[topic_norm]
        
        topic_embedding = self.embeddings[topic_index].reshape(1, -1)

        distances, indices = self.knn.kneighbors(
            topic_embedding, n_neighbors=num_recommendations + 1
        )
        
        recommended_indices = indices[0][1:]
        recommended_words = [self.all_words[idx] for idx in recommended_indices]
        
        logger.info(f"Recommended {len(recommended_words)} words for topic '{topic}'.")
        return recommended_words