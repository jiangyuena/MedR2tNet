import torch
import torch.nn as nn
import torchvision.models as models
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image

import torch.nn.functional as F


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.model1 = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model1.from_pretrained()
        self.model1.cuda()
        self.processor = MedCLIPProcessor()

        self.affine_aa = nn.Linear(512,2048)


    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

        a = []
        for i in images:
            inputs = self.processor(text="lungs", images=i, return_tensors="pt", padding=True)
            outputs = self.model1(**inputs)
            feats = outputs['img_embeds']
            a.append(feats)
        clip_feats = torch.stack(a, dim=0) #batch*1*512

        clip_feats1 = F.relu(self.affine_aa(clip_feats))
        clip_feats2 = clip_feats1.repeat(1, 49, 1)

        patch_feats=patch_feats+clip_feats2

        clip_feats3 = clip_feats1.squeeze(1)
        avg_feats=avg_feats+clip_feats3


        return patch_feats, avg_feats