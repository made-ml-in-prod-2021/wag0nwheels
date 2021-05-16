from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureParams:
    features_to_drop: List[str]
    numerical_features: List[str] = field(default_factory=list)
    #['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'target']
    categorical_features: List[str] = field(default_factory=list)
    #['cp', 'slope', 'ca', 'thal']
    target_name: Optional[str] = field(default="target")
    #pass_limit: float = field(default=0.7, compare=True)
    use_log_trick: bool = field(default=False)
