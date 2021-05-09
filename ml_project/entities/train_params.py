from dataclasses import dataclass, field
from typing import List


@dataclass()
class TrainingParams:
    final_estimator: str
    preprocessing: str