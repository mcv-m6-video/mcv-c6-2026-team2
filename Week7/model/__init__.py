from .model_spotting import Model as BaselineModel
from .model_spotting_tdeed import Model as TDEEDModel

def get_model(args):
    if args.model_type == "baseline":
        print("Using model: BASELINE")
        return BaselineModel(args)
    elif args.model_type == "tdeed":
        print("Using model: TDEED")
        return TDEEDModel(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

__all__ = ["get_model"]