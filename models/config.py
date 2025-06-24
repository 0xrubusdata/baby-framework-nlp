# models/config.py
"""
Centralized configuration for NLP tasks and their associated models.
This approach allows for easy maintenance and clear extensibility.
"""

# Main configuration for NLP tasks
TASKS_CONFIG = {
    "sentiment-analysis": {
        "display_name": "Sentiment Analysis",
        "description": "Determines if a text expresses a positive, negative, or neutral sentiment",
        "models": [
            {
                "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "display_name": "RoBERTa Twitter (Recommended)",
                "description": "Optimized for short texts, social media"
            },
            {
                "name": "nlptown/bert-base-multilingual-uncased-sentiment",
                "display_name": "BERT Multilingual",
                "description": "Multilingual support, good for formal texts"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Tweet to analyze",
                "placeholder": "Enter your tweet here... (max 280 characters)",
                "max_chars": 280
            }
        },
        "output_format": "sentiment_scores"
    },

    "text-generation": {
        "display_name": "Text Generation",
        "description": "Generates text from an initial prompt",
        "models": [
            {
                "name": "gpt2",
                "display_name": "GPT-2 (Lightweight)",
                "description": "Fast, good for getting started"
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "display_name": "DialoGPT Medium",
                "description": "Specialized in conversations"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Start of tweet",
                "placeholder": "Just landed in Paris and...",
                "max_chars": 140
            },
            "max_length": {
                "type": "slider",
                "label": "Generated tweet length",
                "minimum": 10,
                "maximum": 100,
                "default": 50
            },
            "temperature": {
                "type": "slider",
                "label": "Creativity (Temperature)",
                "minimum": 0.1,
                "maximum": 2.0,
                "default": 1.0,
                "step": 0.1
            }
        },
        "output_format": "generated_text"
    },

    "question-answering": {
        "display_name": "Question Answering",
        "description": "Answers a question based on a provided context",
        "models": [
            {
                "name": "distilbert-base-cased-distilled-squad",
                "display_name": "DistilBERT SQuAD",
                "description": "Fast and efficient, trained on SQuAD"
            }
        ],
        "parameters": {
            "context": {
                "type": "textbox",
                "label": "Context/Passage",
                "placeholder": "Paste the text containing the information here...",
                "max_chars": 1000,
                "lines": 5
            },
            "question": {
                "type": "textbox",
                "label": "Question",
                "placeholder": "What is your question?",
                "max_chars": 200
            }
        },
        "output_format": "qa_result"
    },

    "zero-shot-classification": {
        "display_name": "Zero-Shot Classification",
        "description": "Classifies text according to user-defined categories",
        "models": [
            {
                "name": "facebook/bart-large-mnli",
                "display_name": "BART Large MNLI",
                "description": "Performs well for classification without training"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Text to classify",
                "placeholder": "Text to analyze...",
                "max_chars": 500
            },
            "candidate_labels": {
                "type": "textbox",
                "label": "Possible labels (comma-separated)",
                "placeholder": "politics, sport, technology, entertainment",
                "max_chars": 200
            }
        },
        "output_format": "classification_scores"
    },

    "summarization": {
        "display_name": "Twitter Thread Summarization",
        "description": "Summarizes a series of tweets or a Twitter conversation",
        "models": [
            {
                "name": "t5-small",
                "display_name": "T5 Small (Recommended)",
                "description": "Lightweight and efficient for short texts"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Twitter thread/conversation to summarize",
                "placeholder": "Paste multiple tweets separated by newlines...",
                "max_chars": 1000,
                "lines": 6
            },
            "max_length": {
                "type": "slider",
                "label": "Max summary length",
                "minimum": 20,
                "maximum": 100,
                "default": 50
            }
        },
        "output_format": "summary_text"
    }
}

# Utility function to retrieve task information
def get_task_info(task_name):
    """Returns complete information for a task."""
    return TASKS_CONFIG.get(task_name, None)

def get_available_tasks():
    """Returns a list of available tasks."""
    return list(TASKS_CONFIG.keys())

def get_models_for_task(task_name):
    """Returns available models for a given task."""
    task_info = get_task_info(task_name)
    if task_info:
        return [model["name"] for model in task_info["models"]]
    return []

def get_model_info(task_name, model_name):
    """Returns detailed information for a model."""
    task_info = get_task_info(task_name)
    if task_info:
        for model in task_info["models"]:
            if model["name"] == model_name:
                return model
    return None
