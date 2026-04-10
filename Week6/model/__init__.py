from .model_spotting import Model as BaselineModel

def get_model(args):
    if args.model_type == "baseline":
        print("Using model: BASELINE")
        return BaselineModel(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

__all__ = ["get_model"]