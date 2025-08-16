import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.services.recommendation_service import RecommendationService
from src.services.difficulty_classifier_service import DifficultyClassifierService

logger = logging.getLogger(__name__)

class AIAgentService:
    """
    A service that runs a LangChain agent for interactive vocabulary learning.
    It uses other services as tools to find words and classify their difficulty.
    """
    def __init__(self, api_key: str, recommendation_service: RecommendationService, difficulty_service: DifficultyClassifierService):
        """
        Initializes the agent with the LLM and the tools from other services.
        
        Args:
            api_key (str): Your Generative AI API key.
            recommendation_service (RecommendationService): The service for word recommendations.
            difficulty_service (DifficultyClassifierService): The service for word difficulty classification.
        """
        self.recommendation_service = recommendation_service
        self.difficulty_service = difficulty_service
        # Use OpenRouter (OpenAI-compatible) with desired model
        self.llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=2560,
        )
        self.tools = self._setup_tools()
        self.prompt = self._setup_prompt()
        self.agent_executor = AgentExecutor(agent=create_tool_calling_agent(self.llm, self.tools, self.prompt), tools=self.tools)
        # In-memory per-session chat histories
        self._histories: Dict[str, List[BaseMessage]] = {}
        logger.info("AIAgentService initialized successfully.")

    def _setup_tools(self) -> List[tool]:
        """Creates and returns the list of tools for the agent."""

        @tool
        def find_similar_vocabs(topic_of_interest: str) -> List[str]:
            """
            Finds a list of similar vocabulary words given a specific topic of interest.
            
            Args:
                topic_of_interest (str): The topic to find words for.
            
            Returns:
                List[str]: A list of similar vocabulary words.
            """
            logger.debug(f"find_similar_vocabs called with topic: {topic_of_interest}")
            topic_norm = (topic_of_interest or "").strip().lower()
            # Fallback to a known topic if not in the vocabulary index
            if topic_norm not in self.recommendation_service.word_to_index:
                for candidate in [
                    "general","daily","food","nature","people","science",
                    "technology","health","sport","school","travel","art","business"
                ]:
                    if candidate in self.recommendation_service.word_to_index:
                        topic_norm = candidate
                        break
                else:
                    topic_norm = self.recommendation_service.all_words[0] if getattr(self.recommendation_service, "all_words", []) else topic_norm
            return self.recommendation_service.recommend_words_for_topic(topic_norm)

        @tool
        def classify_difficulty(vocab_list: List[str]) -> Dict[str, str]:
            """
            Classifies difficulty for each word in vocab_list.
            
            Args:
                vocab_list (list[str]): The list of words to classify.
                
            Returns:
                dict: A dictionary mapping word to its difficulty.
            """
            logger.debug(f"classify_difficulty called with words: {vocab_list}")
            return {word: self.difficulty_service.classify_word_difficulty(word) for word in vocab_list}
        
        return [find_similar_vocabs, classify_difficulty]

    def _setup_prompt(self) -> ChatPromptTemplate:
        """Creates and returns the chat prompt template for the agent."""
        system_message = (
            """You are an advanced vocabulary learning recommendation system assistant named VocaBuddy, designed to help Chinese students aged 13–17 improve their English vocabulary.

            You must keep track of the student's English proficiency level and topic of interest throughout the entire conversation once they have been provided and confirmed. After confirmation, do not ask for this information again unless the student explicitly changes it. If the student changes their level or topic, politely confirm the change before updating your records.

            If the user provides inputs with typos or unclear information regarding their English level or topic, gently clarify by suggesting the corrected input and ask for confirmation before proceeding. For example: It seems like you meant “Sport” instead of “Sportt”. Is that correct?

            Your primary function is to identify the student's English proficiency level and topic of interest, then recommend vocabulary words matching both.

            Important details:
            - English levels map to difficulty tags: Beginner → Easy, Intermediate → Medium, Advanced → Hard.
            - Canonical topics (use these exact tokens): Food, Daily, School, Travel, Technology, Art, Business, General, Health, Nature, People, Science, Sport.
            - Always use your specialized tools to recommend topic-relevant words and tag their difficulty; do not provide vocabulary outside these topics or levels.

            Strict operating instructions:
            1) Clarify Input: If level or topic are unclear or seem mistyped, ask a brief clarification first.
            2) Tool First: After level/topic are confirmed (or updated), call tools:
               - find_similar_vocabs(topic) → candidate words for the topic
               - classify_difficulty(words) → one of Easy | Medium | Hard
            3) Select a word that matches the student’s level and topic. If no level/topic are known yet, politely ask once, and still suggest a suitable word based on a neutral “General” topic.
            4) Final Output Generation: Give a concise, friendly explanation including the definition and a simple example sentence appropriate for the student’s level.
            5) Word-Focused Discussion: You may discuss usage, part of speech, or related meanings. Do not provide etymology/history. Do not translate unless the student explicitly requests it.
            6) Learning Memory and Quizzing: Keep track of learned words during the session. Occasionally quiz the student after several words. Be encouraging.
            7) Stay on Topic: If asked off-topic, politely redirect to vocabulary learning.
            8) Tone: Friendly, encouraging, and clear for teenagers.

            Output rules for DAILY mode:
            - Provide exactly one suggestion with BOTH of the following:
              1) A single summary line: word | topic | level | one-sentence hint
              2) A machine-readable block:
                 <learned_json> [{{"word":"...","topic":"<one canonical token>","level":"<Easy|Medium|Hard>","hint":"<one-sentence learning hint>"}}] </learned_json>

            Output rules for CHAT mode:
            - Respond conversationally (no pipe-delimited summary line).
            - ALWAYS end your message with exactly one machine-readable block for the current learning word so the client can save history:
              <learned_json> [{{"word":"...","topic":"<one canonical token>","level":"<Easy|Medium|Hard>","hint":"<one-sentence learning hint>"}}] </learned_json>
            - The learned_json must contain exactly one item and use the canonical topic token. Keep keys exactly as shown: word, topic, level, hint. No extra text inside the block.

            Workflow each turn:
            1) If needed, confirm or update level/topic; otherwise reuse remembered values.
            2) Use tools (find_similar_vocabs → classify_difficulty).
            3) Choose one word that matches level/topic.
            4) Provide a brief explanation + example sentence.
            5) End with the learned_json block (exactly one item) as specified above.
            6) Wait for the next request.

            Remember: Do not answer from your own knowledge unrelated to the tools or recommended words; always rely on your tools to provide accurate, level-appropriate vocabulary.
            """
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def invoke_agent(self, user_input: str, session_id: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, Any]:
        """Invokes the agent executor with a user message.

        Args:
            user_input: The message from the user.
            session_id: Optional opaque session identifier for future per-session memory.
            mode: Optional interaction mode ("chat" or "daily"). Defaults to "chat".
        """
        if session_id:
            logger.info(f"Invoking agent (session_id={session_id}) with user input: {user_input}")
        else:
            logger.info(f"Invoking agent with user input: {user_input}")
        sid = (session_id or "default").strip()
        # Prepend mode directive
        mode_norm = (mode or "chat").strip().lower()
        if mode_norm == "daily":
            preamble = (
                "MODE: daily\nReturn exactly one suggestion with a single pipe-delimited summary line and a <learned_json> block."
            )
        else:
            preamble = (
                "MODE: chat\nRespond conversationally. Always end with exactly one <learned_json> block for a current learning word."
            )
        composed_input = f"{preamble}\n\n{user_input}".strip()

        chat_history = self._histories.get(sid, [])
        result = self.agent_executor.invoke({
            "input": composed_input,
            "chat_history": chat_history,
        })
        # Persist this turn into memory
        output_text = result.get("output") if isinstance(result, dict) else str(result)
        chat_history = chat_history + [HumanMessage(content=user_input), AIMessage(content=str(output_text))]
        self._histories[sid] = chat_history
        return result

    def reset_session(self, session_id: Optional[str]) -> None:
        sid = (session_id or "default").strip()
        self._histories.pop(sid, None)