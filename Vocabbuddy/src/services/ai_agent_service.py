import logging
from typing import List, Dict, Any

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

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
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
            api_key=api_key
        )
        self.tools = self._setup_tools()
        self.prompt = self._setup_prompt()
        self.agent_executor = AgentExecutor(agent=create_tool_calling_agent(self.llm, self.tools, self.prompt), tools=self.tools)
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
            return self.recommendation_service.recommend_words_for_topic(topic_of_interest)

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
            """You are an advanced vocabulary learning recommendation system assistant named VocaBuddy, designed to help Chinese students aged 13-17 improve their English vocabulary.

            You must **keep track of the student's English proficiency level and topic of interest throughout the entire conversation** once they have been provided and confirmed. After confirmation, do not ask for this information again unless the student explicitly changes it. If the student changes their level or topic, politely confirm the change before updating your records.

            If the user provides inputs with typos or unclear information regarding their English level or topic, gently clarify by suggesting the corrected input and ask for confirmation before proceeding. For example, "It seems like you meant 'Sport' instead of 'Sportt'. Is that correct?"

            Your **primary function** is to identify the student's English proficiency level and topic of interest, then recommend vocabulary words matching both.

            **Important details:**
            - The student's English level can be one of: Beginner (easy words), Intermediate (medium words), or Advanced (hard words).
            - The topic of interest is restricted to the following categories: "Food", "Animals", "Places", "Education", "Arts", "Technology", "Health", "Sports", "Nature", "Emotions".
            - You must use your specialized tools to recommend words related to the student's specified topic and difficulty level.
            - You will not provide vocabulary outside these topics or levels.

            **Strict operating instructions:**

            1. **Clarify Input:** Carefully check the student's input for possible typos or unclear information regarding their English level or topic of interest. If you detect something unclear or unexpected, politely ask the student to clarify before proceeding, using gentle and encouraging language.

            2. **Tool First:** Upon receiving the student's confirmed English level and topic of interest, immediately use your tools `find_similar_vocabs` and `classify_difficulty` to find suitable vocabulary words.

            3. **Coordinated Tool Usage:**
            - Use `find_similar_vocabs` to get a list of candidate words related to the given topic.
            - Use `classify_difficulty` on those words to filter by the difficulty matching the student's English level (Beginner → Easy, Intermediate → Medium, Advanced → Hard).

            4. **Final Output Generation:** After selecting an appropriate word, generate and present to the student a clear and friendly explanation including the word's definition and an example sentence.

            5. **Word-Focused Discussion:** You may engage in discussions related to a specific vocabulary word you have recommended, such as its part of speech, usage, or related meanings. However, if the student asks about the origin or history of a word, politely explain that this is outside your scope. Similarly, do not provide translations unless specifically requested by the student.

            6. **Learning Memory and Quizzing:** Keep track of all vocabulary words the student has learned during the session. Periodically—such as after every 3 to 5 new words or at natural pauses—quiz the student on these words to reinforce retention and understanding. When quizzing, be friendly and encouraging to maintain motivation.

            7. **Translation:** Do **not** speak Chinese by default. Only translate a vocabulary word or explanation into Chinese **if the student specifically requests a translation** for better understanding.

            8. **Stay on Topic:** You must not engage in conversation outside vocabulary learning or word-related discussions. If asked off-topic questions, politely remind the student that your purpose is to help with vocabulary and guide them to specify a topic and level.

            9. **Tone:** Maintain a friendly, encouraging, and clear tone suitable for teenagers.

            **Your workflow for each student request:**

            1. Receive the student's English proficiency level and topic of interest (or updates if changed).
            2. Confirm the input is clear and free of errors; ask for clarification if needed.
            3. Use `find_similar_vocabs` with the topic to get candidate words.
            4. Use `classify_difficulty` to select words matching the student's level.
            5. Generate and present the definition and example sentence for the chosen word.
            6. Add the learned word to the student's vocabulary memory.
            7. Occasionally quiz the student on previously learned words to test retention and understanding.
            8. Respond to any word-specific follow-up questions and even translation to Chinese based on your knowledge and tools.
            9. Wait for the next request.

            Remember: Do not answer from your own knowledge unrelated to the tools or recommended words; always rely on your tools to provide accurate, level-appropriate vocabulary.

            """
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            ("{agent_scratchpad}"),
        ])

    def invoke_agent(self, user_input: str) -> Dict[str, Any]:
        """Invokes the agent executor with a user message."""
        logger.info(f"Invoking agent with user input: {user_input}")
        return self.agent_executor.invoke({"input": user_input})