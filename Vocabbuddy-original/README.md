# AI Vocabulary Recommendation Backend

This is the backend for an AI-powered vocabulary learning recommendation system. It uses a combination of machine learning models and an AI agent to provide personalized word recommendations to Chinese students (ages 13-17) via a WeChat mini-program.

---

## Features

* **Content-Based Recommendation**: Uses a **K-Nearest Neighbors (KNN)** model on word embeddings to find vocabulary related to a user's chosen topic.
* **Difficulty Classification**: A **Decision Tree model** classifies the difficulty of recommended words to match the student's English proficiency level (Beginner, Intermediate, Advanced).
* **Conversational AI Agent**: A **LangChain agent** with a custom persona and tools interacts with the user in natural language, providing definitions and example sentences.
* **RESTful API**: Exposes these functionalities through a clean, FastAPI-based API.

---

## Project Setup

Follow these steps to get a local copy of the project up and running.

### 1. Clone the Repository

```bash
git clone https://github.com/whizz-tamie/vocabbuddy.git
cd vocabbuddy
````

### 2. Create the `.env` File

Create a file named `.env` in the root directory and add your Generative AI API key. This is required for the conversational agent to function.

```
GOOGLE_API_KEY="your_google_api_key_here"
```

### 3. Install Dependencies

Install all the necessary Python libraries using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Server

Start the FastAPI application using `uvicorn`. The `--reload` flag is great for development as it automatically restarts the server on code changes.

```bash
uvicorn src.api.main:app --reload
```

The server will be running at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

-----

## API Endpoints

| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | A simple health check to ensure the server is running. |
| `GET` | `/recommend/{topic}` | Gets a list of recommended words for a given topic. |
| `POST` | `/chat` | Interacts with the AI agent by sending a user message. |

-----

## Folder Structure

```
/vocabbuddy/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── services/
│   │   ├── ai_agent_service.py
│   │   ├── difficulty_classifier_service.py
│   │   └── recommendation_service.py
│   └── utils/
│       └── data_preprocessing.py
├── models/
│   ├── dt_model.joblib
|   ├── all_vocab_words.npy
|   ├── word_embeddings.npy
|
├── data/
│   └── preprocessed_vocabs.csv
├── .env
├── requirements.txt
└── README.md
```
