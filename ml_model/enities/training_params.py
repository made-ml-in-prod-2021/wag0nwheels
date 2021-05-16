from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    train_model: str = 'log_reg'
    model_params: Dict[str, Any] = field(default_factory={'C': 1, 'penalty': 'l2', 'n_jobs': -1})
