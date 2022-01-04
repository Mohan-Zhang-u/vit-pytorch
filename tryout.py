import torch
from vit_pytorch import ViT, ViTwithTextInput, MAE
from vit_pytorch.simmim import SimMIM
import pickle
from PIL import Image
import numpy as np
from einops import rearrange
import torchvision.transforms as T

with open('chars.pkl', 'rb') as fp:
    text_dict_list = pickle.load(fp)

image_size = 256
patch_size = 32

v = ViTwithTextInput(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 2048,
    dim = 1024,
    depth = 6,
    decoder_depth = 6,
    language_transform_depth = 6,
    heads = 16,
    decoder_heads = 64,
    language_transform_heads = 16,
    mlp_dim = 2048,
    decoder_mlp_dim = 2048,
    language_transform_dim = 2048,
    text_dict_list = text_dict_list,
    dropout = 0.1,
    emb_dropout = 0.1,
    text_seq_length = 64,
    text_padding_idx = 0,
)


# inference_transform = T.Compose(
#     [
#         T.Resize(size=(image_size, image_size)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Imagenet
#     ]
# )

# orig_img = Image.open('1.jpg')
# img = inference_transform(orig_img) # torch.Size([3, 256, 256])
# img = torch.stack([img, img]) # torch.Size([2, 3, 256, 256])

# https://github.com/lucidrains/vit-pytorch/issues/159
# https://forums.fast.ai/t/is-normalizing-the-input-image-by-imagenet-mean-and-std-really-necessary/51338/3
# img0 = torch.randn(1, 3, image_size, image_size)
# img = Image.open('1.jpg')
# img = img.resize((image_size, image_size),Image.ANTIALIAS)
# img.save('p.jpg')
# img = torch.tensor([np.asarray(img)]) # torch.Size([1, 256, 256, 3]) range from 0-255, need to convert to [0,1) as instructed here https://pytorch.org/vision/stable/transforms.html#torchvision-transforms
# transform shape
# img = rearrange(img, 'b h w c -> b c h w') # convert to correct shape. torch.Size([1, 3, 256, 256])

# # simulate Rearrange layer
# a = rearrange(img, 'b c (hn p1) (wn p2) -> b (hn wn) (p1 p2 c)', p1=patch_size, p2=patch_size) # torch.Size([1, 64, 3072])
# # simulate internal converter
# b = rearrange(a, 'b (hn wn) (p1 p2 c) -> b c (hn p1) (wn p2)', p1=patch_size, p2=patch_size, hn=int(image_size/patch_size), wn=int(image_size/patch_size)) # torch.Size([1, 3, 256, 256])
# # convert shape back to b h w c
# img_back = rearrange(b, 'b c h w -> b h w c') # torch.Size([1, 256, 256, 3])
# img_back = np.array(img_back)[0] # (256, 256, 3)
# img_back = Image.fromarray(img_back)
# img_back.save("b.jpg")

img = torch.randn(1, 3, image_size, image_size)
x = v.encoding(img, ["hello world!"]) # (1, num_classes)
decoded_x = v.decoding(x)
print(1)

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
