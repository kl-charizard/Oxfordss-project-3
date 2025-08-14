import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from src.services.recommendation_service import RecommendationService
from src.services.difficulty_classifier_service import DifficultyClassifierService
from src.services.ai_agent_service import AIAgentService

# Load environment variables from .env file
load_dotenv()

# Create a logger
logger = logging.getLogger(__name__)

# Global variables to hold our loaded data and services
recommendation_service: Optional[RecommendationService] = None
difficulty_service: Optional[DifficultyClassifierService] = None
ai_agent_service: Optional[AIAgentService] = None

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    This is the modern replacement for @app.on_event.
    """
    global recommendation_service, difficulty_service, ai_agent_service

    # Setup logging on startup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Application startup initiated.")

    try:
        # Initialize the recommendation service by loading precomputed embeddings and words
        embeddings_path = os.path.join(os.getcwd(), 'models', 'word_embeddings.npy')
        words_path = os.path.join(os.getcwd(), 'models', 'all_vocab_words.npy')
        recommendation_service = RecommendationService(
            embeddings_path=embeddings_path,
            words_path=words_path
        )
        logger.info("Recommendation service loaded.")

        # Initialize the difficulty classifier service
        difficulty_model_path = os.path.join(os.getcwd(), 'models', 'dt_model.joblib')
        difficulty_service = DifficultyClassifierService(model_path=difficulty_model_path)
        logger.info("Difficulty classifier service loaded.")

        # Initialize the AI Agent service
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        ai_agent_service = AIAgentService(
            api_key=api_key,
            recommendation_service=recommendation_service,
            difficulty_service=difficulty_service
        )
        logger.info("AI Agent service loaded.")

        logger.info("All services are ready.")

    except Exception as e:
        logger.error(f"Critical error during startup: {e}", exc_info=True)
        # We can't proceed if core services fail, so we raise an error.
        raise RuntimeError("Failed to initialize the application.")

    # The 'yield' keyword indicates that the startup phase is complete.
    # The application will now handle requests.
    yield
    
    # This code runs on application shutdown
    logger.info("Application shutdown.")

# Create the FastAPI app instance with the lifespan event handler
app = FastAPI(
    title="AI Vocabulary Recommendation Agent",
    description="Backend API for a vocabulary learning recommendation system powered by content-based filtering.",
    version="1.0.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    """Pydantic model for the chat request body."""
    user_message: str

@app.get("/")
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "AI Vocabulary Recommendation API is running."}

@app.get("/recommend/{topic}", response_model=List[str])
def recommend_words_endpoint(topic: str, num_recommendations: int = 5):
    """
    API endpoint to get a list of recommended words for a given topic.

    Args:
        topic (str): The user's chosen topic of interest.
        num_recommendations (int): The number of words to recommend (default is 5).

    Returns:
        List[str]: A list of recommended words.
    """
    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not yet initialized. Please wait.")

    logger.info(f"Received recommendation request for topic '{topic}' with {num_recommendations} words.")

    recommended_words = recommendation_service.recommend_words_for_topic(topic, num_recommendations)

    if not recommended_words:
        raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found or no recommendations available.")

    return recommended_words

@app.post("/chat")
def chat_with_agent(request: ChatRequest):
    """
    API endpoint for interacting with the AI vocabulary assistant.
    """
    if ai_agent_service is None:
        raise HTTPException(status_code=503, detail="AI agent service not yet initialized.")
    
    try:
        response = ai_agent_service.invoke_agent(request.user_message)
        return {"response": response.get("output")}
    except Exception as e:
        logger.error(f"Error during chat interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing chat request.")