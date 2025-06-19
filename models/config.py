# models/config.py
"""
Configuration centralisée pour les tâches NLP et leurs modèles associés.
Cette approche permet une maintenance facile et une extensibilité claire.
"""

# Configuration principale des tâches NLP
TASKS_CONFIG = {
    "sentiment-analysis": {
        "display_name": "Analyse de sentiment",
        "description": "Détermine si un texte exprime un sentiment positif, négatif ou neutre",
        "models": [
            {
                "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "display_name": "RoBERTa Twitter (Recommandé)",
                "description": "Optimisé pour les textes courts, réseaux sociaux"
            },
            {
                "name": "nlptown/bert-base-multilingual-uncased-sentiment", 
                "display_name": "BERT Multilingual",
                "description": "Support multilingue, bon pour textes formels"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Texte à analyser",
                "placeholder": "Entrez votre texte ici...",
                "max_chars": 500
            }
        },
        "output_format": "sentiment_scores"
    },
    
    "text-generation": {
        "display_name": "Génération de texte", 
        "description": "Génère du texte à partir d'un prompt initial",
        "models": [
            {
                "name": "gpt2",
                "display_name": "GPT-2 (Léger)",
                "description": "Rapide, bon pour débuter"
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "display_name": "DialoGPT Medium", 
                "description": "Spécialisé dans les conversations"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Prompt de départ",
                "placeholder": "Il était une fois...",
                "max_chars": 200
            },
            "max_length": {
                "type": "slider",
                "label": "Longueur maximale",
                "minimum": 10,
                "maximum": 200,
                "default": 50
            },
            "temperature": {
                "type": "slider", 
                "label": "Créativité (Temperature)",
                "minimum": 0.1,
                "maximum": 2.0,
                "default": 1.0,
                "step": 0.1
            }
        },
        "output_format": "generated_text"
    },
    
    "question-answering": {
        "display_name": "Questions-Réponses",
        "description": "Répond à une question basée sur un contexte fourni", 
        "models": [
            {
                "name": "distilbert-base-cased-distilled-squad",
                "display_name": "DistilBERT SQuAD",
                "description": "Rapide et efficace, entraîné sur SQuAD"
            }
        ],
        "parameters": {
            "context": {
                "type": "textbox",
                "label": "Contexte/Passage",
                "placeholder": "Collez ici le texte contenant l'information...",
                "max_chars": 1000,
                "lines": 5
            },
            "question": {
                "type": "textbox", 
                "label": "Question",
                "placeholder": "Quelle est votre question ?",
                "max_chars": 200
            }
        },
        "output_format": "qa_result"
    },
    
    "zero-shot-classification": {
        "display_name": "Classification Zero-Shot",
        "description": "Classifie un texte selon des catégories définies par l'utilisateur",
        "models": [
            {
                "name": "facebook/bart-large-mnli", 
                "display_name": "BART Large MNLI",
                "description": "Performant pour la classification sans entraînement"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Texte à classifier", 
                "placeholder": "Texte à analyser...",
                "max_chars": 500
            },
            "candidate_labels": {
                "type": "textbox",
                "label": "Labels possibles (séparés par des virgules)",
                "placeholder": "politique, sport, technologie, divertissement",
                "max_chars": 200
            }
        },
        "output_format": "classification_scores"
    },
    
    "summarization": {
        "display_name": "Résumé de texte",
        "description": "Génère un résumé concis d'un texte long",
        "models": [
            {
                "name": "facebook/bart-large-cnn",
                "display_name": "BART CNN",
                "description": "Excellent pour résumer des articles"
            },
            {
                "name": "t5-small",
                "display_name": "T5 Small", 
                "description": "Plus léger, bon compromis vitesse/qualité"
            }
        ],
        "parameters": {
            "text_input": {
                "type": "textbox",
                "label": "Texte à résumer",
                "placeholder": "Collez votre texte long ici...",
                "max_chars": 2000,
                "lines": 8
            },
            "max_length": {
                "type": "slider",
                "label": "Longueur max du résumé", 
                "minimum": 30,
                "maximum": 200,
                "default": 100
            }
        },
        "output_format": "summary_text"
    }
}

# Fonction utilitaire pour récupérer les informations d'une tâche
def get_task_info(task_name):
    """Retourne les informations complètes d'une tâche."""
    return TASKS_CONFIG.get(task_name, None)

def get_available_tasks():
    """Retourne la liste des tâches disponibles."""
    return list(TASKS_CONFIG.keys())

def get_models_for_task(task_name):
    """Retourne les modèles disponibles pour une tâche donnée."""
    task_info = get_task_info(task_name)
    if task_info:
        return [model["name"] for model in task_info["models"]]
    return []

def get_model_info(task_name, model_name):
    """Retourne les informations détaillées d'un modèle."""
    task_info = get_task_info(task_name)
    if task_info:
        for model in task_info["models"]:
            if model["name"] == model_name:
                return model
    return None
