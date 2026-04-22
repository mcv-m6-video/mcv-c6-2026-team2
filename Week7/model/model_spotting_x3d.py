"""
Action spotting using X3D-M as spatiotemporal backbone.
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
from pytorchvideo.models.hub import x3d_l

#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch == "x3d_l":
                features = x3d_l(pretrained=True)
                feat_dim = 192
            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features
            self._d = feat_dim

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.43216, 0.394666, 0.37645), std = (0.22803, 0.22145, 0.216989)) #Kinetics 400 
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            # x.shape: B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization kinetics-400 stats
                        
            # X3D expects (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)

            # Remove X3D classification head: use only blocks 0..4
            for i, block in enumerate(self._features.blocks):
                if i == 5:
                    break
                x = block(x) # B, C, T, H, W

            # Pool only spatial dims, keep temporal dim
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # B, C, T, 1, 1
            x = x.squeeze(-1).squeeze(-1)                    # B, C, T
            x = x.permute(0, 2, 1)                           # B, T, C

            #MLP
            im_feat = self._fc(x) #B, T, num_classes+1

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
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

            print(f"Total params: {total_params:,} ({total_params/1e6:.3f} M)")
            print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.3f} M)")

            dummy = torch.randn(1, clip_len, 3, 224, 398).to(device)

            was_training = self.training
            self.eval()

            try:
                macs, _ = profile(self, inputs=(dummy,), verbose=False)
                print(f"MACs: {macs/1e9:.3f} G")
            except Exception as e:
                print(f"THOP failed: {e}")

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
        self._soft_labels = args.soft_labels
        self._model.print_stats(args.clip_len, self.device)

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.amp.autocast(self.device):
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)

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
                logits = self._model(seq)

            # apply softmax
            pred = torch.softmax(logits, dim=-1)
            
            return pred.cpu().numpy(), logits
