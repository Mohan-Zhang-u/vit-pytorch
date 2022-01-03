import torch
from vit_pytorch import ViT, ViTwithTextInput

v = ViTwithTextInput(
    image_size = 256,
    patch_size = 32,
    num_classes = 1024,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    pool = 'mean',
)

img = torch.randn(1, 3, 256, 256)

preds = v(img, ["hello world!"]) # (1, num_classes)