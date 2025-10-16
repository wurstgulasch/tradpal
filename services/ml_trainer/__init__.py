#!/usr/bin/env python3
"""
ML Trainer Service - Machine learning model training and optimization.

This service provides comprehensive ML capabilities for trading signal prediction,
including model training, hyperparameter optimization, evaluation, and management.
"""

from .api import app
from .service import MLTrainerService, EventSystem
from .client import MLTrainerServiceClient, TrainingRequest, RetrainingRequest, EvaluationRequest

__all__ = [
    "app",
    "MLTrainerService",
    "EventSystem",
    "MLTrainerServiceClient",
    "TrainingRequest",
    "RetrainingRequest",
    "EvaluationRequest"
]