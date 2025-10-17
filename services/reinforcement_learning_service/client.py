"""
Reinforcement Learning Service Client
Client for AI-powered trading decisions using reinforcement learning.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class ReinforcementLearningServiceClient:
    """Client for Reinforcement Learning Service"""

    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize the client"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

    async def close(self) -> None:
        """Close the client"""
        if self.session:
            await self.session.close()

    async def get_trading_action(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trading action recommendation from RL agent.

        Args:
            market_state: Current market state data

        Returns:
            Recommended action with confidence and reasoning
        """
        try:
            async with self.session.post(f"{self.base_url}/action",
                                       json=market_state) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get trading action: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting trading action: {e}")
            return {}

    async def update_model(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the RL model with new experience.

        Args:
            experience: Experience tuple (state, action, reward, next_state, done)

        Returns:
            Update status
        """
        try:
            async with self.session.post(f"{self.base_url}/model/update",
                                       json=experience) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to update model: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return {}

    async def start_training(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start RL model training.

        Args:
            training_config: Training configuration parameters

        Returns:
            Training task information
        """
        try:
            async with self.session.post(f"{self.base_url}/train",
                                       json=training_config) as response:
                if response.status in [200, 202]:
                    return await response.json()
                else:
                    logger.error(f"Failed to start training: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return {}

    async def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get training status for a specific task.

        Args:
            task_id: Training task identifier

        Returns:
            Training status and metrics
        """
        try:
            async with self.session.get(f"{self.base_url}/training/status/{task_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get training status: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {}

    async def stop_training(self, task_id: str) -> Dict[str, Any]:
        """
        Stop a training task.

        Args:
            task_id: Training task identifier

        Returns:
            Stop status
        """
        try:
            async with self.session.delete(f"{self.base_url}/training/{task_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to stop training: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return {}

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current RL model.

        Returns:
            Model information and metrics
        """
        try:
            async with self.session.get(f"{self.base_url}/model/info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get model info: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

    async def save_model(self, filepath: str) -> Dict[str, Any]:
        """
        Save the current RL model.

        Args:
            filepath: Path to save the model

        Returns:
            Save status
        """
        try:
            params = {"filepath": filepath}
            async with self.session.post(f"{self.base_url}/model/save",
                                       params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to save model: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return {}

    async def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load an RL model.

        Args:
            filepath: Path to load the model from

        Returns:
            Load status
        """
        try:
            params = {"filepath": filepath}
            async with self.session.post(f"{self.base_url}/model/load",
                                       params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to load model: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {}

    async def get_available_algorithms(self) -> Dict[str, Any]:
        """
        Get list of available RL algorithms.

        Returns:
            Available algorithms and their descriptions
        """
        try:
            async with self.session.get(f"{self.base_url}/algorithms") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get algorithms: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting algorithms: {e}")
            return {}

    async def get_available_actions(self) -> Dict[str, Any]:
        """
        Get list of available trading actions.

        Returns:
            Available actions and their descriptions
        """
        try:
            async with self.session.get(f"{self.base_url}/actions") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get actions: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting actions: {e}")
            return {}

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get service health status.

        Returns:
            Health status information
        """
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get health status: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {}

    async def get_q_values(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Q-values for all actions given a market state.

        Args:
            market_state: Current market state

        Returns:
            Q-values for all possible actions
        """
        try:
            async with self.session.post(f"{self.base_url}/q-values",
                                       json=market_state) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get Q-values: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting Q-values: {e}")
            return {}