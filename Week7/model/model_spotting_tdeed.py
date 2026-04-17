"""
T-DEED.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import random
import torch.nn.functional as F
import math
from thop import profile


#Local imports
from model.modules import BaseRGBModel, EDSGPMIXERLayers, FCLayers, step, process_prediction
from model.shift import make_temporal_shift

# Code adapted from: https://github.com/arturxe2/T-DEED.git

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()

            self._feature_arch = args.feature_arch
            self._temp_arch = args.temporal_arch
            self._radi_displacement = args.radi_displacement
            
            assert self._temp_arch in ['ed_sgp_mixer'], 'Only ed_sgp_mixer supported for now'
            assert 'rny' in self._feature_arch, 'Only rny supported for now'
            

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = -1
            if self._feature_arch.endswith('_gsm'):
                make_temporal_shift(features, args.clip_len, mode='gsm')
                self._require_clip_len = args.clip_len
            elif self._feature_arch.endswith('_gsf'):
                make_temporal_shift(features, args.clip_len, mode='gsf')
                self._require_clip_len = args.clip_len

            self._features = features
            self._feat_dim = self._d
            feat_dim = self._d

            #Positional encoding
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / args.clip_len, size = (args.clip_len, self._d)))
            
            if self._temp_arch == 'ed_sgp_mixer':
                self._temp_fine = EDSGPMIXERLayers(feat_dim, args.clip_len, num_layers=args.n_layers, ks = args.sgp_ks, k = args.sgp_r, concat = True)
                self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)
            else:
                raise NotImplementedError(self._temp_arch)
            
            if self._radi_displacement > 0:
                self._pred_displ = FCLayers(self._feat_dim, 1)
            
            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

            #Augmentation at test time
            self.augmentationI = T.Compose([
                T.RandomHorizontalFlip(p = 1.0)
            ])

            #Croping in case of using it
            self.croping = args.crop_dim
            if self.croping is not None and self.croping > 0:
                self.cropT = T.RandomCrop((self.croping, self.croping))
                self.cropI = T.CenterCrop((self.croping, self.croping))
            else:
                self.cropT = torch.nn.Identity()
                self.cropI = torch.nn.Identity()

        def forward(self, x):
            
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape

            x.view(-1, channels, height, width)
            if self.croping is not None and self.croping > 0:
                height = self.croping
                width = self.croping
            x = self.cropT(x) # same crop for all frames
            x = x.view(batch_size, clip_len, channels, height, width)

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)

            im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)

            if self._temp_arch == 'ed_sgp_mixer':
                im_feat = self._temp_fine(im_feat)
                if self._radi_displacement > 0:
                    displ_feat = self._pred_displ(im_feat).squeeze(-1)
                    im_feat = self._pred_fine(im_feat)
                    return {'im_feat': im_feat, 'displ_feat': displ_feat}
                im_feat = self._pred_fine(im_feat)
                return im_feat
            
            else:
                raise NotImplementedError(self._temp_arch)
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x
        
        def augmentI(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentationI(x[i])
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
                label = label.to(self.device)

                if 'labelD' in batch.keys():
                    labelD = batch['labelD'].to(self.device).float()

                with torch.amp.autocast(self.device):
                    pred = self._model(frame)
                    if 'labelD' in batch.keys():
                        predD = pred['displ_feat']
                        pred = pred['im_feat']

                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights) 
                     
                    if 'labelD' in batch.keys():
                        lossD = F.mse_loss(predD, labelD, reduction = 'none')
                        lossD = (lossD).mean()
                        loss = loss + lossD

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

            if isinstance(pred, dict):
                predD = pred['displ_feat']
                pred = pred['im_feat']
                pred = process_prediction(pred, predD)
                return pred.cpu().numpy(), pred

            logits = pred
            pred = torch.softmax(logits, dim=-1)
            return pred.cpu().numpy(), logits
        