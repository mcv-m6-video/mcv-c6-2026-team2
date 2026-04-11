from .model_spotting import Model as BaselineModel
from .model_spotting_lstm import Model as LSTMModel

def get_model(args):
    if args.model_type == "baseline":
        print("Using model: BASELINE")
        return BaselineModel(args)
    elif args.model_type == "lstm":
        print("Using model: LSTM")
        return LSTMModel(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

__all__ = ["get_model"]