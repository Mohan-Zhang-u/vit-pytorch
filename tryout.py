import torch
from vit_pytorch import ViT, ViTwithTextInput, MAE
from vit_pytorch.simmim import SimMIM
import pickle

with open('chars.pkl', 'rb') as fp:
    text_dict_list = pickle.load(fp)

v = ViTwithTextInput(
    image_size = 256,
    patch_size = 32,
    num_classes = 2048,
    dim = 1024,
    depth = 6,
    decoder_depth = 6,
    heads = 16,
    decoder_heads = 64,
    mlp_dim = 2048,
    decoder_mlp_dim = 2048,
    text_dict_list = text_dict_list,
    dropout = 0.1,
    emb_dropout = 0.1,
    text_seq_length = 64,
    text_padding_idx = 0,
)

img = torch.randn(1, 3, 256, 256)

x = v.encoding(img, ["hello world!"]) # (1, num_classes)

# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 2048
# )

# mim = SimMIM(
#     encoder = v,
#     masking_ratio = 0.5  # they found 50% to yield the best results
# )

# images = torch.randn(8, 3, 256, 256)

# loss = mim(images)
# loss.backward()

# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 2048
# )

# mae = MAE(
#     encoder = v,
#     masking_ratio = 0.25,   # the paper recommended 75% masked patches
#     decoder_dim = 1024,      # paper showed good results with just 512
#     decoder_depth = 6       # anywhere from 1 to 8
# )

# images = torch.randn(8, 3, 256, 256)

# loss = mae(images)
# loss.backward()