import pandas as pd
import joblib
import logging
import textstat
from wordfreq import zipf_frequency
from typing import List

# Create a logger
logger = logging.getLogger(__name__)

class DifficultyClassifierService:
    """
    A service for classifying a word's difficulty using a pre-trained
    Decision Tree model.
    """

    def __init__(self, model_path: str):
        """
        Initializes the service by loading the trained model.
        
        Args:
            model_path (str): The file path to the saved joblib model.
        """
        self.model = self._load_model(model_path)
        logger.info("Difficulty classifier service initialized successfully.")

    def _load_model(self, model_path: str):
        """Loads the pre-trained Decision Tree model."""
        try:
            model = joblib.load(model_path)
            logger.info(f"Successfully loaded model from '{model_path}'.")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found at '{model_path}'.")
            raise RuntimeError(f"Model file not found at '{model_path}'")

    def _extract_features(self, word: str) -> List[float]:
        """
        Extracts the features required by the model for a given word.
        
        Args:
            word (str): The word to extract features for.
            
        Returns:
            List[float]: A list of feature values [length, syllables, frequency].
        """
        length = len(word)
        syllables = textstat.syllable_count(word)
        freq_score = zipf_frequency(word.lower(), 'en')
        return [length, syllables, freq_score]

    def classify_word_difficulty(self, word: str) -> str:
        """
        Classifies the difficulty of a word using the loaded model.
        
        Args:
            word (str): The word to classify.
            
        Returns:
            str: The predicted difficulty level (e.g., 'Easy', 'Medium', 'Hard').
        """
        # Extract features for the given word
        features = self._extract_features(word)
        
        # The model expects a DataFrame with specific column names.
        input_df = pd.DataFrame(
            [features], 
            columns=["Word length", "Number of Syllables", "Word Frequency"]
        )
        
        # Make the prediction
        prediction = self.model.predict(input_df)[0]
        
        logger.debug(f"Classified '{word}' as '{prediction}'.")
        return prediction