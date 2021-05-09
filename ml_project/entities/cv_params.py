from dataclasses import dataclass, field


@dataclass()
class CVParams:
    folds: int = field(default=9)
    random_state: int = field(default=0)