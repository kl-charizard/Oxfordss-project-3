import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, nest_asyncio
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import json
import re
 
 
from src.services.recommendation_service import RecommendationService
from src.services.difficulty_classifier_service import DifficultyClassifierService
from src.services.ai_agent_service import AIAgentService
 
 
# Create a logger
logger = logging.getLogger(__name__)

# Global variables to hold our loaded data and services
recommendation_service: Optional[RecommendationService] = None
difficulty_service: Optional[DifficultyClassifierService] = None
ai_agent_service: Optional[AIAgentService] = None

# simple in-memory session store placeholder (if your AIAgentService also keeps memory, /reset can be a no-op)
_session_store: Dict[str, Dict[str, Any]] = {}

# Canonical topic set and aliases for normalization
ALLOWED_TOPICS = {"daily","sport","school","travel","technology","art","business","food","general","health","nature","people","science"}
TOPIC_ALIASES = {
    "sports": "sport",
    "tech": "technology",
    "it": "technology",
    "medical": "health",
    "medicine": "health",
    "healthcare": "health",
    "doctor": "health",
    "hospital": "health",
}

def normalize_topic(topic: str) -> str:
    if not topic:
        return ""
    t = topic.strip().lower()
    t = TOPIC_ALIASES.get(t, t)
    if t not in ALLOWED_TOPICS and t.endswith("s") and t[:-1] in ALLOWED_TOPICS:
        t = t[:-1]
    return t

LEARNED_RE = re.compile(r"<learned_json>\s*(.*?)\s*</learned_json>", re.DOTALL)

def _session(ctx_id: str) -> Dict[str, Any]:
    sid = ctx_id or "default"
    if sid not in _session_store:
        _session_store[sid] = {"history": []}
    return _session_store[sid]

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
        # Resolve project root relative to this file so it works after moving the folder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

        # Initialize the recommendation service by loading precomputed embeddings and words
        embeddings_path = os.path.join(project_root, 'models', 'word_embeddings.npy')
        words_path = os.path.join(project_root, 'models', 'all_vocab_words.npy')
        recommendation_service = RecommendationService(
            embeddings_path=embeddings_path,
            words_path=words_path
        )
        logger.info("Recommendation service loaded.")

        # Initialize the difficulty classifier service
        difficulty_model_path = os.path.join(project_root, 'models', 'dt_model.joblib')
        difficulty_service = DifficultyClassifierService(model_path=difficulty_model_path)
        logger.info("Difficulty classifier service loaded.")

        # Initialize the AI Agent service (OpenRouter API key for GPT-4o)
        api_key = "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"

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # WeChat DevTools origins vary; allow all
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class ChatRequest(BaseModel):
    """Original model (kept for reference)."""
    user_message: str

class ChatRequestFlex(BaseModel):
    """Flexible body to support Mini Program payloads and the original schema."""
    session_id: Optional[str] = None
    message: Optional[str] = None
    level: Optional[str] = None
    topic: Optional[str] = None
    user_message: Optional[str] = None
    mode: Optional[str] = None  # "chat" or "daily"

@app.get("/")
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "AI Vocabulary Recommendation API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(body: ChatRequestFlex):
    sid = (body.session_id or "default").strip()
    # if your AIAgentService has a reset method, call it here; otherwise clear local placeholder
    try:
        if ai_agent_service:
            ai_agent_service.reset_session(sid)
        _session_store.pop(sid, None)
    except Exception:
        pass
    return {"status": "reset", "session_id": sid}

@app.get("/history")
def history(session_id: Optional[str] = None):
    sid = (session_id or "default").strip()
    return {"session_id": sid, "history": _session(sid)["history"]}

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

    topic_norm = normalize_topic(topic)
    logger.info(f"Received recommendation request for topic '{topic}' -> canonical '{topic_norm}' with {num_recommendations} words.")

    recommended_words = recommendation_service.recommend_words_for_topic(topic_norm, num_recommendations)

    if not recommended_words:
        raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found or no recommendations available.")

    return recommended_words

@app.post("/chat")
def chat_with_agent(request: ChatRequestFlex):
    """
    API endpoint for interacting with the AI vocabulary assistant.
    Accepts either {"user_message": "..."} or the Mini Program shape
    {"session_id", "message", "level", "topic"}.
    """
    if ai_agent_service is None:
        raise HTTPException(status_code=503, detail="AI agent service not yet initialized.")
    
    # unify fields
    text = (request.user_message or request.message or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="'user_message' or 'message' is required")
    session_id = (request.session_id or "default").strip()
    level = (request.level or "").strip()
    topic = (request.topic or "").strip()
    mode = (request.mode or "chat").strip().lower()

    # Normalize topic
    topic_norm = normalize_topic(topic)
    topic_label = topic_norm.capitalize() if topic_norm else ""

    # prepend level/topic if provided so the agent can use them
    prefix = []
    if level:
        prefix.append(f"Level: {level}")
    if topic_label:
        prefix.append(f"Topic: {topic_label}")
        prefix.append(f"Use canonical topic token: {topic_norm}")
    agent_input = ("\n".join(prefix) + "\n" + text) if prefix else text

    try:
        response = ai_agent_service.invoke_agent(agent_input, session_id=session_id, mode=mode)
        output = response.get("output") if isinstance(response, dict) else str(response)

        # Parse learned_json and store to session history
        learned_items = []
        m = LEARNED_RE.search(output or "")
        if m:
            try:
                learned_items = json.loads(m.group(1))
            except Exception:
                learned_items = []
        # Ensure we always provide one learned item in chat/daily modes so the Mini Program can save it
        need_learned = (not learned_items) and (mode in ("chat", "daily"))
        if need_learned:
            try:
                topic_used = topic_norm or "general"
                words = recommendation_service.recommend_words_for_topic(topic_used, 1) if recommendation_service else []
                picked = words[0] if words else None
                if picked is None:
                    # try a safe fallback topic
                    for candidate in ["general", "daily", "people", "nature"]:
                        words = recommendation_service.recommend_words_for_topic(candidate, 1) if recommendation_service else []
                        if words:
                            picked = words[0]
                            topic_used = candidate
                            break
                if picked:
                    level_tag = ""
                    try:
                        level_tag = difficulty_service.classify_word_difficulty(picked) if difficulty_service else ""
                    except Exception:
                        level_tag = ""
                    learned_items = [{
                        "word": picked,
                        "topic": (topic_used.capitalize() if topic_used else "General"),
                        "level": level_tag,
                        "hint": "Current learning word"
                    }]
                    # Optionally append a hidden block to the output so future parsing also works
                    try:
                        learned_json_text = json.dumps(learned_items)
                        output = f"{output}\n\n<learned_json> {learned_json_text} </learned_json>"
                    except Exception:
                        pass
            except Exception:
                pass
        if learned_items:
            sess = _session(session_id)
            sess["history"].extend(learned_items)

        return {"reply": output, "session_id": session_id, "canonical_topic": topic_norm, "learned": learned_items}
    except Exception as e:
        # Log and fallback to direct recommendation so Mini Program doesn’t break
        logger.error(f"Agent error, using fallback: {e}", exc_info=True)
        fallback_reply = ""
        learned_items = []
        try:
            topic_used = topic_norm or "general"
            words = recommendation_service.recommend_words_for_topic(topic_used, 5) if recommendation_service else []
            if words:
                # Try to tag difficulty if service supports it
                tags = {}
                try:
                    # Use the correct method name on the classifier
                    tags = {w: difficulty_service.classify_word_difficulty(w) for w in words} if difficulty_service else {}
                except Exception:
                    tags = {}

                # Prepare a human-friendly reply
                lines = [f"Here are some {topic_used.capitalize()} words:"]
                for w in words:
                    tag = tags.get(w)
                    lines.append(f"- {w}{f' ({tag})' if tag else ''}")
                fallback_reply = "\n".join(lines)

                # Also return structured 'learned' items so the Mini Program can render directly
                # Only include the first item as today's suggestion; keep others in history shape if needed
                for w in words:
                    learned_items.append({
                        "word": w,
                        "topic": topic_used.capitalize(),
                        "level": tags.get(w, ""),
                        "hint": f"Suggested {topic_used} word"
                    })
            else:
                fallback_reply = "Sorry, I couldn’t fetch recommendations right now. Please try again."
        except Exception as ie:
            logger.error(f"Fallback failed: {ie}", exc_info=True)
            fallback_reply = "Sorry, something went wrong. Please try again."
        return {"reply": fallback_reply, "session_id": session_id, "canonical_topic": topic_norm, "fallback": True, "learned": learned_items}

if __name__ == "__main__":
    nest_asyncio.apply()
    PORT = int(os.getenv("PORT", "8000"))
    print(f"Starting API server at http://0.0.0.0:{PORT} ...")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")