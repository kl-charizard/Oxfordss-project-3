import pandas as pd
import logging
from typing import Dict, Union

# Create a logger
logger = logging.getLogger(__name__)

TOPICS = [
    "food",
    "animals",
    "places",
    "education",
    "arts",
    "technology",
    "health",
    "sports",
    "nature",
    "emotions"
    ]

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the preprocessed vocabulary dataset and handles file errors."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from '{file_path}'.")
        return df
    except FileNotFoundError:
        logger.error(f"Error: '{file_path}' not found. Please ensure the file exists.")
        exit()

def create_word_mappings(df: pd.DataFrame) -> Dict[str, Union[int, str]]:
    """Creates dictionaries for word-to-index mappings."""
    # Concatenate 'Word' columnsto and TOPICS to get all unique vocabulary items
    all_words = pd.concat([df['Word'], TOPICS]).unique().tolist()
    
    # Create a dictionary to map each word to a unique index
    word_to_index = {word: i for i, word in enumerate(all_words)}
    
    logger.info("Successfully created word-to-index mappings.")
    return word_to_index, all_words