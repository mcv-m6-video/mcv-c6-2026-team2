from .evaluation import main as evaluate
from .matching import main as match
from .tracking import main as track
from .train_matcher import main as train_match

__all__ = ["track", "match", "evaluate", "train_match"]
