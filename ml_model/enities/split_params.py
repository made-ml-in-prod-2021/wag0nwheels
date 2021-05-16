from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.3)
    random_state: int = field(default=42)
