import torch
from vit_pytorch import ViT, ViTwithTextInput
import pickle

with open('chars.pkl', 'rb') as fp:
    text_dict_list = pickle.load(fp)

v = ViTwithTextInput(
    image_size = 256,
    patch_size = 32,
    num_classes = 2048,
    dim = 2048,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    text_dict_list = text_dict_list,
    dropout = 0.1,
    emb_dropout = 0.1,
    pool = 'mean',
    text_seq_length = 64,
    text_padding_idx = 0,
)

img = torch.randn(1, 3, 256, 256)

preds = v(img, ["hello world!"]) # (1, num_classes)