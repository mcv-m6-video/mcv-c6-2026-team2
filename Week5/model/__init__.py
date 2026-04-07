from .model_classification import Model as BaselineModel
from .model_classification_lstm import Model as LSTMModel
from .model_classification_lstm_attention import Model as LSTMAttnModel
from .model_classification_3dcnn import Model as Model3DCnn
from .model_classification_3dcnn_lstm import Model as Model3DCnnLSTM

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
    elif args.model_type == "3dcnn":
        print("Using model: 3D CNN")
        return Model3DCnn(args)
    elif args.model_type == "3dcnn_lstm":
        print("Using model: 3D NN + LSTM")
        return Model3DCnnLSTM(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

__all__ = ["get_model"]