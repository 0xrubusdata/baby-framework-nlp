# models/pipeline_manager.py
"""
Gestionnaire centralisé des pipelines Hugging Face.
Optimisé pour les contraintes mémoire des Spaces gratuits.
"""

import logging
from typing import Dict, Any, Optional, Union
from transformers import pipeline, Pipeline
import torch
import gc

from .config import get_task_info, get_model_info

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Gestionnaire intelligent des pipelines HuggingFace.
    
    Fonctionnalités :
    - Cache des pipelines pour éviter les rechargements
    - Gestion mémoire optimisée 
    - Validation des paramètres selon la tâche
    - Gestion des erreurs robuste
    """
    
    def __init__(self, max_cached_pipelines: int = 2):
        """
        Initialise le gestionnaire de pipelines.
        
        Args:
            max_cached_pipelines: Nombre maximum de pipelines en cache
                                 (limité pour éviter l'OOM sur Spaces gratuits)
        """
        self._pipelines_cache: Dict[str, Pipeline] = {}
        self._max_cached_pipelines = max_cached_pipelines
        self._cache_order = []  # Pour LRU (Least Recently Used)
        
        logger.info(f"PipelineManager initialisé avec {max_cached_pipelines} pipelines max en cache")
    
    def _generate_cache_key(self, task: str, model_name: str) -> str:
        """Génère une clé unique pour le cache."""
        return f"{task}::{model_name}"
    
    def _cleanup_cache(self):
        """
        Nettoie le cache selon une stratégie LRU quand la limite est atteinte.
        Crucial pour les Spaces avec mémoire limitée.
        """
        while len(self._pipelines_cache) >= self._max_cached_pipelines:
            # Supprime le plus ancien pipeline (LRU)
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._pipelines_cache:
                logger.info(f"Libération mémoire : suppression pipeline {oldest_key}")
                del self._pipelines_cache[oldest_key]
                
                # Force le garbage collection pour libérer la mémoire GPU/CPU
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def get_pipeline(self, task: str, model_name: str) -> Pipeline:
        """
        Récupère ou crée un pipeline pour la tâche et le modèle donnés.
        
        Args:
            task: Type de tâche NLP (ex: "sentiment-analysis")
            model_name: Nom du modèle HuggingFace
            
        Returns:
            Pipeline configuré et prêt à l'usage
            
        Raises:
            ValueError: Si la tâche/modèle n'est pas supporté
            RuntimeError: Si le chargement du modèle échoue
        """
        # Validation de la tâche
        task_info = get_task_info(task)
        if not task_info:
            raise ValueError(f"Tâche non supportée : {task}. "
                           f"Tâches disponibles : {list(get_task_info().keys())}")
        
        # Validation du modèle pour cette tâche
        model_info = get_model_info(task, model_name)
        if not model_info:
            available_models = [m["name"] for m in task_info["models"]]
            raise ValueError(f"Modèle {model_name} non supporté pour {task}. "
                           f"Modèles disponibles : {available_models}")
        
        # Gestion du cache
        cache_key = self._generate_cache_key(task, model_name)
        
        # Pipeline déjà en cache
        if cache_key in self._pipelines_cache:
            logger.info(f"Pipeline trouvé en cache : {cache_key}")
            # Marque comme récemment utilisé
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._pipelines_cache[cache_key]
        
        # Nettoyage préventif du cache
        self._cleanup_cache()
        
        # Chargement du nouveau pipeline
        try:
            logger.info(f"Chargement du pipeline : {task} avec {model_name}")
            
            # Configuration optimisée selon la tâche
            pipeline_kwargs = self._get_pipeline_config(task, model_name)
            
            # Création du pipeline
            pipe = pipeline(task, model=model_name, **pipeline_kwargs)
            
            # Mise en cache
            self._pipelines_cache[cache_key] = pipe
            self._cache_order.append(cache_key)
            
            logger.info(f"Pipeline {cache_key} chargé et mis en cache avec succès")
            return pipe
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du pipeline {cache_key}: {str(e)}")
            raise RuntimeError(f"Impossible de charger le modèle {model_name} pour {task}: {str(e)}")
    
    def _get_pipeline_config(self, task: str, model_name: str) -> Dict[str, Any]:
        """
        Retourne la configuration optimisée pour un pipeline donné.
        Adaptée aux contraintes des Spaces Hugging Face.
        """
        base_config = {
            "return_all_scores": False,  # Économise de la mémoire
            "device": -1,  # Force CPU (Spaces gratuits n'ont pas de GPU)
        }
        
        # Configurations spécifiques par tâche
        task_specific_configs = {
            "sentiment-analysis": {
                "return_all_scores": True,  # Pour avoir tous les scores de sentiment
            },
            "text-generation": {
                "do_sample": True,
                "pad_token_id": 50256,  # Token de padding GPT-2
            },
            "zero-shot-classification": {
                "return_all_scores": True,
            },
            "question-answering": {
                "handle_impossible_answer": True,
            },
            "summarization": {
                "do_sample": False,  # Plus déterministe pour les résumés
            }
        }
        
        # Fusion des configurations
        config = {**base_config, **task_specific_configs.get(task, {})}
        
        # Optimisations spécifiques par modèle si nécessaire
        if "gpt2" in model_name.lower():
            config["pad_token_id"] = 50256
        
        return config
    
    def process_request(self, task: str, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite une requête complète : validation + pipeline + formatage.
        
        Args:
            task: Type de tâche NLP
            model_name: Nom du modèle à utiliser
            inputs: Paramètres d'entrée selon la tâche
            
        Returns:
            Résultats formatés selon le type de tâche
        """
        try:
            # 1. Validation des inputs
            validated_inputs = self._validate_inputs(task, inputs)
            
            # 2. Récupération du pipeline
            pipe = self.get_pipeline(task, model_name)
            
            # 3. Exécution selon le type de tâche
            raw_results = self._execute_pipeline(pipe, task, validated_inputs)
            
            # 4. Formatage des résultats
            formatted_results = self._format_results(task, raw_results)
            
            logger.info(f"Requête {task}/{model_name} traitée avec succès")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement {task}/{model_name}: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _validate_inputs(self, task: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Valide les inputs selon la configuration de la tâche."""
        task_info = get_task_info(task)
        parameters = task_info["parameters"]
        
        validated = {}
        
        for param_name, param_config in parameters.items():
            if param_name in inputs:
                value = inputs[param_name]
                
                # Validation de la longueur pour les textbox
                if param_config["type"] == "textbox" and "max_chars" in param_config:
                    max_chars = param_config["max_chars"]
                    if len(str(value)) > max_chars:
                        raise ValueError(f"{param_name} dépasse la limite de {max_chars} caractères")
                
                # Validation des sliders
                elif param_config["type"] == "slider":
                    min_val = param_config.get("minimum", 0)
                    max_val = param_config.get("maximum", 100)
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"{param_name} doit être entre {min_val} et {max_val}")
                
                validated[param_name] = value
        
        return validated
    
    def _execute_pipeline(self, pipe: Pipeline, task: str, inputs: Dict[str, Any]) -> Any:
        """Exécute le pipeline selon le type de tâche."""
        if task == "sentiment-analysis":
            return pipe(inputs["text_input"])
        
        elif task == "text-generation":
            return pipe(
                inputs["text_input"],
                max_length=inputs.get("max_length", 50),
                temperature=inputs.get("temperature", 1.0),
                do_sample=True
            )
        
        elif task == "question-answering":
            return pipe(
                question=inputs["question"],
                context=inputs["context"]
            )
        
        elif task == "zero-shot-classification":
            labels = [label.strip() for label in inputs["candidate_labels"].split(",")]
            return pipe(inputs["text_input"], labels)
        
        elif task == "summarization":
            return pipe(
                inputs["text_input"],
                max_length=inputs.get("max_length", 50),
                min_length=10,
                do_sample=False
            )
        
        else:
            raise ValueError(f"Exécution non implémentée pour la tâche: {task}")
    
    def _format_results(self, task: str, raw_results: Any) -> Dict[str, Any]:
        """Formate les résultats selon le type de tâche."""
        # Pour l'instant, retourne les résultats bruts
        # Sera implémenté dans formatters.py au Module 3
        return {
            "task": task,
            "results": raw_results,
            "status": "success"
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Retourne des informations sur l'état du cache (pour debugging)."""
        return {
            "cached_pipelines": list(self._pipelines_cache.keys()),
            "cache_size": len(self._pipelines_cache),
            "max_cache_size": self._max_cached_pipelines,
            "cache_order": self._cache_order.copy()
        }

# Instance globale (Singleton pattern)
pipeline_manager = PipelineManager(max_cached_pipelines=2)
