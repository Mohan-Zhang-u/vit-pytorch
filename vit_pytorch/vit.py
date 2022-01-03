import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img) # x.shape [1, 64, 1024]
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
class ViTwithTextInput(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, decoder_depth, heads, decoder_heads, mlp_dim, decoder_mlp_dim, text_dict_list, channels = 3, dim_head = 64, decoder_dim_head=64, dropout = 0., decoder_dropout = 0., emb_dropout = 0., text_seq_length = 64, text_padding_idx = 0, img_tensor_loss_weight = 1., text_tensor_loss_weight = 0.1, vector_loss_weight = 1.):
        super().__init__()
        # init text embedding layer
        self.text_dict_list = text_dict_list # a list of all possible characters (literally a dictionary).
        self.text_seq_length = text_seq_length
        self.text_padding_idx = text_padding_idx
        self.num_classes = num_classes
        self.dim = dim # both encoder dim and decoder dim.
        self.text_dict_length = len(text_dict_list)
        self.text_embedding_layer = torch.nn.Embedding(num_embeddings=self.text_dict_length, embedding_dim=dim, padding_idx=text_padding_idx, max_norm=1., norm_type=2.) # https://stats.stackexchange.com/questions/177905/should-i-normalize-word2vecs-word-vectors-before-using-them
        
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.h_n_patches = self.image_height // self.patch_height
        self.w_n_patches = self.image_width // self.patch_width
        self.num_patches = self.h_n_patches * self.w_n_patches
        patch_dim = channels * self.patch_height * self.patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + text_seq_length + 1, dim)) # we still keep the trainable pos_embedding as it is in the original ViT paper.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.decoder_transformer = Transformer(dim, decoder_depth, decoder_heads, decoder_dim_head, decoder_mlp_dim, decoder_dropout)

        self.to_pixels = nn.Linear(dim, patch_dim)
        
        # loss functions
        self.img_tensor_loss = torch.nn.MSELoss(reduction='mean')
        self.img_tensor_loss_weight = img_tensor_loss_weight
        self.text_tensor_loss = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.) # https://discuss.pytorch.org/t/loss-functions-for-batches/20488/7 batch size included.
        self.text_tensor_loss_weight = text_tensor_loss_weight # should be small, as it does not directly affect our desired output.
        self.vector_loss = torch.nn.MSELoss(reduction='mean')
        self.vector_loss_weight = vector_loss_weight
        
        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        
    def text_to_indices(self, text):
        # here, text is a list of strings
        indices = []
        for string in text:
            sub_indices = []
            for char_s in string:
                sub_indices.append(self.text_dict_list.index(char_s))
            sub_indices = sub_indices[:self.text_seq_length] # cannot go beyond text_seq_length
            while len(sub_indices) < self.text_seq_length: # padding
                sub_indices.append(self.text_padding_idx)
            indices.append(sub_indices)
        return torch.LongTensor(indices)
    
    def indices_to_text(self, indices):
        texts = []
        for text_indices in indices:
            text = ''
            for char_indices in text_indices:
                idx = char_indices.item()
                text += self.text_dict_list[idx]
            texts.append(text)
        return texts
    
    def get_cls_from_x(self, x):
        return x[:, 0:1]
    
    def get_img_patches_from_x(self, x):
        return x[:, 1:1+self.num_patches]
    
    def get_text_patches_from_x(self, x):
        return x[:, 1+self.num_patches:]
    
    def get_style_vector(self, x):
        return x[:, :1+self.num_patches] # [cls] + patches. pos encoding was added before the encoder transformer.
    
    def get_semantic_vector(self, x):
        return x[:, 1+self.num_patches:] # text patches.
    
    def convert_to_img(self, decoded_x):
        img_patches = self.get_img_patches_from_x(decoded_x)
        pix = self.to_pixels(img_patches) # torch.Size([img.shape[0], num_patches, patch_dim])
        img_back = rearrange(pix, 'b (hn wn) (p1 p2 c) -> b c (hn p1) (wn p2)', p1=self.patch_height, p2=self.patch_width, hn=self.h_n_patches, wn=self.w_n_patches) # torch.Size([img.shape[0], c, self.image_height, self.image_width])
        return img_back
    
    def get_text_mins(self, decoded_x):
        text_patches = self.get_text_patches_from_x(decoded_x) # torch.Size(img.shape[0],text_seq_length, dim)
        cos_sim = torch.matmul(text_patches, torch.transpose(self.text_embedding_layer.weight, 0, 1)) # torch.Size(img.shape[0],text_seq_length, self.text_dict_length)
        return torch.min(cos_sim, 2) # torch.Size(img.shape[0],text_seq_length)
    
    def convert_to_text(self, decoded_x):
        min_vals, min_indices = self.get_text_mins(decoded_x)
        return self.indices_to_text(min_indices)
    
    def compute_img_tensor_loss(self, orig_img_tensor, pred_img_tensor):
        return self.img_tensor_loss_weight * self.img_tensor_loss(orig_img_tensor, pred_img_tensor)
    
    def compute_vector_loss(self, orig_vector, pred_vector):
        return self.vector_loss_weight * self.vector_loss(orig_vector, pred_vector)
    
    def compute_text_tensor_loss(self, orig_text, pred_text_tensor):
        """
        orig_text are list of strings.
        pred_text_tensor should be a result of get_text_patches_from_x
        """
        target = self.text_to_indices(orig_text)
        cos_sim = torch.matmul(pred_text_tensor, torch.transpose(self.text_embedding_layer.weight, 0, 1))
        # shift batch size to be d_K
        cos_sim = rearrange(cos_sim, 'b n c -> n c b')
        target = rearrange(target, 'b n -> n b')
        return self.text_tensor_loss_weight * self.text_tensor_loss(cos_sim, target)
        
    def encoding(self, img, text):
        x = self.to_patch_embedding(img) # torch.Size(img.shape[0], num_patches, dim)
        indices = self.text_to_indices(text) # torch.Size(img.shape[0], text_seq_length)
        x_text = self.text_embedding_layer(indices) # torch.Size(img.shape[0], text_seq_length, dim)
        x = torch.cat((x, x_text), dim=1) # torch.Size(img.shape[0], num_patches + text_seq_length, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) # torch.Size(img.shape[0], 1, dim)
        x = torch.cat((cls_tokens, x), dim=1) # torch.Size(img.shape[0],1 +  num_patches + text_seq_length, dim)
        x += self.pos_embedding[:, :(n + 1)] # torch.Size(img.shape[0],1 +  num_patches + text_seq_length, dim)
        x = self.dropout(x) # torch.Size(img.shape[0],1 + num_patches + text_seq_length, dim)

        x = self.encoder_transformer(x) # torch.Size(img.shape[0],1 +  num_patches + text_seq_length, dim)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # torch.Size(img.shape[0], dim)

        # x = self.to_latent(x) # torch.Size(img.shape[0], dim)
        # x = self.mlp_head(x) # torch.Size(img.shape[0], num_classes)
        # # style_vector = x[:,:int(self.num_classes/2)] # torch.Size(img.shape[0], num_classes/2)
        # # semantic_vector = x[:,int(self.num_classes/2):] # torch.Size(img.shape[0], num_classes/2)
        return x

    def decoding(self, x):
        decoded_x = self.decoder_transformer(x) # torch.Size(img.shape[0],1 +  num_patches + text_seq_length, dim)
        return decoded_x