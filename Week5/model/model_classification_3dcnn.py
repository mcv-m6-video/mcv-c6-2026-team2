"""
Video-based models (R3D-18 / R(2+1)D-18 / X3D).
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from thop import profile
from fvcore.nn import FlopCountAnalysis
from torchvision.models.video import r3d_18, r2plus1d_18
from torchvision.models.video import R3D_18_Weights, R2Plus1D_18_Weights
from pytorchvideo.models.hub import x3d_m, x3d_s

#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self._pretrained = getattr(args, "pretrained", False)

            if self._feature_arch == "r3d_18":
                weights = R3D_18_Weights.DEFAULT
                features = r3d_18(weights=weights)
                feat_dim = features.fc.in_features
                features.fc = nn.Identity()
 
            elif self._feature_arch == "r2plus1d_18":
                weights = R2Plus1D_18_Weights.DEFAULT
                features = r2plus1d_18(weights=weights)
                feat_dim = features.fc.in_features
                features.fc = nn.Identity()
            elif self._feature_arch in {"x3d_s", "x3d_m"}:
                x3d_ctor = {
                    "x3d_s": x3d_s,
                    "x3d_m": x3d_m,
                }[self._feature_arch]
                features = x3d_ctor(pretrained=self._pretrained)
                feat_dim = features.blocks[-1].proj.in_features

                # Keep the projected X3D head, but swap its fixed pool for adaptive
                # pooling so clip length and frame size are not hard-coded.
                features.blocks[-1].pool.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                features.blocks[-1].proj = nn.Linear(feat_dim, args.num_classes)
                features.blocks[-1].activation = None
            else:
                raise NotImplementedError(self._feature_arch)

            self._features = features
            self._d = feat_dim

            # The torchvision backbones output features, while X3D can emit logits
            # directly once its head is replaced.
            self._fc = None if self._feature_arch.startswith("x3d_") else FCLayers(self._d, args.num_classes)

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.43216, 0.394666, 0.37645), std = (0.22803, 0.22145, 0.216989)) #Kinetics 400 
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            # x.shape: B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            # torchvision video models expect [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)

            # Video backbone processes the whole clip
            im_feat = self._features(x)

            if self._fc is not None:
                im_feat = self._fc(im_feat) #B, num_classes

            return im_feat 
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            """
            Apply temporally-consistent augmentations.
            x: [B, T, C, H, W]
            Same transform parameters are applied to all frames in a clip.
            """
            B = x.size(0)

            for b in range(B):
                clip = x[b]

                # Flip
                if torch.rand(1) < 0.5:
                    clip = torch.flip(clip, dims=[3])

                # Color jitter (probabilístico)
                if torch.rand(1) < 0.25:
                    brightness = torch.empty(1).uniform_(0.7, 1.2).item()
                    contrast = torch.empty(1).uniform_(0.7, 1.2).item()
                    saturation = torch.empty(1).uniform_(0.7, 1.2).item()
                    hue = torch.empty(1).uniform_(-0.2, 0.2).item()

                    clip = T.functional.adjust_brightness(clip, brightness)
                    clip = T.functional.adjust_contrast(clip, contrast)
                    clip = T.functional.adjust_saturation(clip, saturation)
                    clip = T.functional.adjust_hue(clip, hue)

                # Blur
                if torch.rand(1) < 0.25:
                    clip = T.functional.gaussian_blur(clip, kernel_size=5)

                x[b] = clip

            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self, clip_len, device):
            # Params
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

            print(f"Total params: {total_params:,} ({total_params/1e6:.3f} M)")
            print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.3f} M)")

            # extracted frames are 398x224 (W x H)
            dummy = torch.randn(1, clip_len, 3, 224, 398).to(device)
            was_training = self.training
            self.eval()

            # MACs
            try:
                macs, _ = profile(self, inputs=(dummy,), verbose=False)
                print(f"MACs: {macs/1e9:.3f} G")
            except Exception as e:
                print(f"THOP failed: {e}")
            
            # FLOPs
            try:
                flops = FlopCountAnalysis(self, dummy)
                flops_total = flops.total()
                print(f"FLOPs: {flops_total/1e9:.3f} G")
            except Exception as e:
                print(f"fvcore failed: {e}")

            if was_training:
                self.train()

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

        self._model.print_stats(args.clip_len, self.device)

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).float()

                with torch.amp.autocast(self.device):
                    pred = self._model(frame)
                    loss = F.binary_cross_entropy_with_logits(
                            pred, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.amp.autocast(self.device):
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.sigmoid(pred)
            
            return pred.cpu().numpy()
