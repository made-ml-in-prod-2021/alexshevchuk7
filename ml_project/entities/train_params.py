from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass()
class TrainingParams:
    final_estimator: str
    model_params: Dict[str, Any]
    preprocessing: str
