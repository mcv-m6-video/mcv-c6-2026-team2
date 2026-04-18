from .model_spotting import Model as BaselineModel
from .model_spotting_tdeed import Model as TDEEDModel
from .model_spotting_tdeed_x3d import Model as TDEEDX3DModel
from .model_spotting_x3d import Model as X3DModel
from .model_spotting_x3d_neck import Model as X3DNeckModel

def get_model(args):
    if args.model_type == "baseline":
        print("Using model: BASELINE")
        return BaselineModel(args)
    elif args.model_type == "tdeed":
        print("Using model: TDEED")
        return TDEEDModel(args)
    elif args.model_type == "tdeed_x3d":
        print("Using model: TDEED + X3D")
        return TDEEDX3DModel(args)
    elif args.model_type == "x3d":
        print("Using model: X3D")
        return X3DModel(args)
    elif args.model_type in ["x3d_lstm", "x3d_gru"]:
        print(f"Using model: {args.model_type.upper()}")
        return X3DNeckModel(args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

__all__ = ["get_model"]