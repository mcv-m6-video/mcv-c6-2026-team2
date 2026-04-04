from .model_classification import Model as BaselineModel
from .model_classification_lstm import Model as LSTMModel
from .model_classification_lstm_attention import Model as LSTMAttnModel

def get_model(args):
    if args.model_type == "baseline":
        print("Using model: BASELINE")
        return BaselineModel(args)
    elif args.model_type == "lstm":
        print("Using model: LSTM")
        return LSTMModel(args)
    elif args.model_type == "lstm_attn":
        print("Using model: LSTM + ATTENTION")
        return LSTMAttnModel(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

__all__ = ["get_model"]