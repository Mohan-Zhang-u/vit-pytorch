import torch
from torch import nn
import torch.nn.functional as F
import codecs
import pickle
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torchvision.transforms as T
from PIL import Image
import mzutils

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

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LSATransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViTwithTextInputHorizontal(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, decoder_depth, language_transform_depth, heads, decoder_heads, language_transform_heads, mlp_dim, decoder_mlp_dim, language_transform_dim, text_dict_list, normalize=True, pad_center_crop=False, channels = 3, dim_head = 64, decoder_dim_head=64, language_transform_dim_head=64, dropout = 0., decoder_dropout = 0., language_transform_dropout = 0., emb_dropout = 0., text_seq_length = 64, unknown_char_loc='unknown.txt', text_padding_idx = 0, use_SPT=False, img_loss_name = 'BCEWithLogitsLoss', img_loss_weight = 0.001, intermediate_feature_loss_weight = 0.05, text_loss_weight = 0.1, vector_loss_weight = 1.):
        super().__init__()
        # init text embedding layer
        self.cls_token_len = 1
        self.intermediate_feature_vector_len = 1 + 1 + 2 + 2
        # shape of x should be torch.Size(img.shape[0], self.cls_token_len + num_patches + self.intermediate_feature_vector_len + text_seq_length, dim)
        self.text_dict_list = text_dict_list # a list of all possible characters (literally a dictionary).
        self.text_seq_length = text_seq_length
        self.unknown_char_loc = unknown_char_loc
        self.unknown_chars = []
        self.text_padding_idx = text_padding_idx
        self.num_classes = num_classes
        self.dim = dim # both encoder dim and decoder dim.
        self.text_dict_length = len(text_dict_list)
        self.one_hot_dim = torch.nn.functional.one_hot(torch.tensor(list(range(self.dim))), num_classes = self.dim).float() # torch.Size(img.shape[0], text_seq_length, dim) TODO: here, we are using it as our text_embedding. We pad len(self.text_dict_list) to self.dim, which is a huge waste of compute. Can we improve this?
        self.processor = mzutils.ImageExperimentProcessor(image_size=image_size, normalize=normalize, pad_center_crop=pad_center_crop)
        
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.h_n_patches = self.image_height // self.patch_height
        self.w_n_patches = self.image_width // self.patch_width
        self.num_patches = self.h_n_patches * self.w_n_patches
        patch_dim = channels * self.patch_height * self.patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if use_SPT:
            self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
                nn.Linear(patch_dim, dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.cls_token_len + self.num_patches + self.intermediate_feature_vector_len + text_seq_length, dim)) # we still keep the trainable pos_embedding as it is in the original ViT paper.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if use_SPT:
            self.encoder_transformer = LSATransformer(dim, depth, heads, dim_head, mlp_dim, dropout) #TODO: check whether we do it here or just decoder also needs change?
        else:
            self.encoder_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.decoder_transformer = Transformer(dim, decoder_depth, decoder_heads, decoder_dim_head, decoder_mlp_dim, decoder_dropout)
        self.l1_to_l2_style_transformer = Transformer(dim, language_transform_depth, language_transform_heads, language_transform_dim_head, language_transform_dim, language_transform_dropout)
        self.l2_to_l1_style_transformer = Transformer(dim, language_transform_depth, language_transform_heads, language_transform_dim_head, language_transform_dim, language_transform_dropout)

        self.to_pixels = nn.Linear(dim, patch_dim)
        
        # loss functions
        self.img_loss_name = img_loss_name
        if self.img_loss_name == 'BCEWithLogitsLoss':
            self.img_loss = torch.nn.BCEWithLogitsLoss(reduction='sum') # we need to add an sigmoid layer at the end of img prediction.
            self.sigmoid_regularize_func = nn.Sigmoid()
        elif self.img_loss_name == 'BCELoss':
            self.img_loss = torch.nn.BCELoss(reduction='sum') # we need to clip the predction to [0,1]
            print("For numerical stability, we suggest BCEWithLogitsLoss over this!")
        elif self.img_loss_name == 'MSELoss':
            self.img_loss = torch.nn.MSELoss(reduction='sum')
        else:
            raise ValueError('Unknown loss, should be either BCEWithLogitsLoss, BCELoss or MSELoss')
        self.img_loss_weight = img_loss_weight
        self.intermediate_feature_loss = torch.nn.MSELoss(reduction='mean')
        self.intermediate_feature_loss_weight = intermediate_feature_loss_weight
        self.text_loss = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.) # https://discuss.pytorch.org/t/loss-functions-for-batches/20488/7 batch size included.
        self.text_loss_weight = text_loss_weight # should be small, as it does not directly affect our desired output.
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
                # sub_indices.append(self.text_dict_list.index(char_s))
                if char_s in self.text_dict_list:
                    sub_indices.append(self.text_dict_list.index(char_s))
                else:
                    sub_indices.append(self.text_dict_list.index('𖡄')) # unknown character.
                    if char_s not in self.unknown_chars:
                        print('unknown:', char_s)
                        self.unknown_chars.append(char_s)
                        with open(self.unknown_char_loc, 'wb') as fp:
                            pickle.dump(self.unknown_chars, fp)
            sub_indices = sub_indices[:self.text_seq_length] # cannot go beyond text_seq_length
            while len(sub_indices) < self.text_seq_length: # padding
                sub_indices.append(self.text_padding_idx)
            indices.append(sub_indices)
        indices = torch.LongTensor(indices).to(self.to_pixels.weight.device)
        return indices
    
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
        return x[:, 0:self.cls_token_len]
    
    def get_img_patches_from_x(self, x):
        return x[:, self.cls_token_len:self.cls_token_len+self.num_patches]
    
    def get_intermediate_patches_from_x(self, x):
        return x[:, self.cls_token_len + self.num_patches: self.cls_token_len + self.num_patches + self.intermediate_feature_vector_len]
    
    def get_text_patches_from_x(self, x):
        return x[:, self.cls_token_len+self.num_patches + self.intermediate_feature_vector_len:]
    
    def get_style_vector(self, x):
        return x[:, :self.cls_token_len + self.num_patches + self.intermediate_feature_vector_len] # [cls] + patches + intermediate_feature_vector. pos encoding was added before the encoder transformer.
    
    def get_semantic_vector(self, x):
        return x[:, self.cls_token_len + self.num_patches + self.intermediate_feature_vector_len:] # text patches.
    
    def cat_style_and_semantic_vectors(self, style_vector, semantic_vector):
        return torch.cat((style_vector, semantic_vector), dim=1)
    
    def convert_to_img(self, decoded_x, mode='loss_compute'):
        """[summary]

        Args:
            decoded_x ([type]): [description]
            mode (str): can be 'loss_compute' or 'inference'.

        Returns:
            a image vector.
        """
        img_patches = self.get_img_patches_from_x(decoded_x)
        pix = self.to_pixels(img_patches) # torch.Size([img.shape[0], num_patches, patch_dim])
        img_back = rearrange(pix, 'b (hn wn) (p1 p2 c) -> b c (hn p1) (wn p2)', p1=self.patch_height, p2=self.patch_width, hn=self.h_n_patches, wn=self.w_n_patches) # torch.Size([img.shape[0], c, self.image_height, self.image_width])
        # if self.img_loss_name == 'BCEWithLogitsLoss':
        #     img_back = self.sigmoid_regularize_func(img_back) # we need to add an sigmoid layer at the end of img prediction.
        if self.img_loss_name == 'BCELoss':
            img_loss = mzutils.differentiable_clamp(img_back, min=1e-8, max=1-1e-8) # we need to clip the predction to [1e-8,1-1e-8] for numerical stability.
        if mode == 'inference' and self.img_loss_name == 'BCEWithLogitsLoss':
            img_back = self.sigmoid_regularize_func(img_back) # we need to add an sigmoid layer at the end of img prediction to shape the prediction to [0,1].
        return img_back
    
    def get_text_maxs(self, decoded_x):
        text_patches = self.get_text_patches_from_x(decoded_x) # torch.Size(img.shape[0],text_seq_length, dim)
        cos_sim = torch.matmul(text_patches, self.one_hot_dim) # torch.Size(img.shape[0],text_seq_length, dim)
        return torch.max(cos_sim, 2) # torch.Size(img.shape[0],text_seq_length)
    
    def convert_to_text(self, decoded_x):
        max_vals, max_indices = self.get_text_maxs(decoded_x)
        return self.indices_to_text(max_indices)
    
    def compute_img_loss(self, orig_img, pred_img, device='cpu'):
        """
        orig_img is a list of PIL.Image.Image
        """
        processed_imgs, size = self.processor.preprocess_transform_imgs(orig_img, device = device)
        return self.img_loss_weight * self.img_loss(pred_img, processed_imgs)
    
    def compute_vector_loss(self, orig_vector, pred_vector):
        return self.vector_loss_weight * self.vector_loss(orig_vector, pred_vector)
    
    def compute_text_loss(self, orig_text, pred_text_tensor):
        """
        orig_text are list of strings.
        pred_text_tensor should be a result of get_text_patches_from_x(decoded_x) # torch.Size([img.shape[0],text_seq_length, self.dim])
        """
        if self.one_hot_dim.device != pred_text_tensor.device:
            self.one_hot_dim = self.one_hot_dim.to(pred_text_tensor.device)
        target = self.text_to_indices(orig_text)
        cos_sim = torch.matmul(pred_text_tensor, self.one_hot_dim) # torch.Size(img.shape[0], text_seq_length, dim)
        # shift batch size to be d_K
        cos_sim = rearrange(cos_sim, 'b n c -> n c b') 
        target = rearrange(target, 'b n -> n b')
        return self.text_loss_weight * self.text_loss(cos_sim, target)
    
    def compute_intermediate_feature_loss(self, decoded_x, language_labels, original_img_sizes, target_image_sizes, text_lens):
        decoded_intermediate_patches = self.get_intermediate_patches_from_x(decoded_x)
        if self.one_hot_dim.device != decoded_intermediate_patches.device:
            self.one_hot_dim = self.one_hot_dim.to(decoded_intermediate_patches.device)
        target_indices = self.intermediate_feature_indices(language_labels, original_img_sizes, target_image_sizes, text_lens, device=self.to_pixels.weight.device)
        cos_sim = torch.matmul(decoded_intermediate_patches, self.one_hot_dim) # torch.Size(img.shape[0], self.intermediate_feature_vector_len, dim)
        pred_max_vals, pred_max_indices = torch.max(cos_sim, 2) # torch.Size(img.shape[0],self.intermediate_feature_vector_len)
        return self.intermediate_feature_loss_weight * self.intermediate_feature_loss(target_indices.float(), pred_max_indices.float())
    
    def intermediate_feature_indices(self, language_labels, original_img_sizes, target_image_sizes, text_lens, device='cpu'):
        """[summary]

        Args:
            language_labels (list of str): 'l1' or 'l2'
            original_img_sizes ([type]): [description]
            target_image_sizes ([type]): [description]
            texts ([type]): [description]
        Returns:
            [torch.Tensor]: 
        """
        assert len(original_img_sizes) == len(target_image_sizes) and len(target_image_sizes) == len(text_lens)
        indices = []
        for i in range(len(original_img_sizes)):
            b_indices = []
            if language_labels[i] == 'l1':
                b_indices.append(0)
            elif language_labels[i] == 'l2':
                b_indices.append(self.dim-1)
            else:
                raise ValueError('language_label should be either l1 or l2')
            b_indices.append(min(text_lens[i], self.dim-1))
            b_indices.append(min(original_img_sizes[i][0], self.dim-1))
            b_indices.append(min(original_img_sizes[i][1], self.dim-1))
            b_indices.append(min(target_image_sizes[i][0], self.dim-1))
            b_indices.append(min(target_image_sizes[i][1], self.dim-1))
            indices.append(b_indices)
        return torch.tensor(indices).to(device)
    
    def intermediate_feature_vectors(self, language_labels, original_img_sizes, target_image_sizes, text_lens, device='cpu'):
        """[summary]

        Args:
            language_labels (list of str): 'l1' or 'l2'
            original_img_sizes ([type]): [description]
            target_image_sizes ([type]): [description]
            texts ([type]): [description]
        Returns:
            [torch.Tensor]: torch.Size(img.shape[0], self.intermediate_feature_vector_len, dim)
        """
        indices = self.intermediate_feature_indices(language_labels, original_img_sizes, target_image_sizes, text_lens, device=self.to_pixels.weight.device)
        intermediate_features = torch.nn.functional.one_hot(indices, num_classes = self.dim).float().to(device)
        return intermediate_features
        
    def encoding(self, imgs, texts, language_labels, target_image_sizes):
        """[summary]

        Args:
            imgs ([type]): a list of PIL.Image.Image
            texts ([type]): a list of strings
            target_image_sizes ([type]): a list of (w, h)

        Returns:
            [type]: [description]
        """
        assert len(imgs) == len(texts) and len(texts) == len(target_image_sizes)
        img_tensors, original_img_sizes = self.processor.preprocess_transform_imgs(imgs, device=self.to_pixels.weight.device)
        img_tensors = img_tensors
        x = self.to_patch_embedding(img_tensors) # torch.Size(img.shape[0], num_patches, dim)
        intermediate_features = self.intermediate_feature_vectors(language_labels, original_img_sizes, target_image_sizes, [len(text) for text in texts], device=self.to_pixels.weight.device) # torch.Size(img.shape[0], self.intermediate_feature_vector_len, dim)
        indices = self.text_to_indices(texts) # torch.Size(img.shape[0], text_seq_length)
        x_text = torch.nn.functional.one_hot(indices, num_classes = self.dim) # torch.Size(img.shape[0], text_seq_length, dim) TODO: here, we pad len(self.text_dict_list) to self.dim, which is a huge waste of compute. Can we improve this?
        x = torch.cat((x, intermediate_features, x_text), dim=1) # torch.Size(img.shape[0], num_patches + self.intermediate_feature_vector_len + text_seq_length, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) # torch.Size(img.shape[0], self.cls_token_len, dim)
        x = torch.cat((cls_tokens, x), dim=1) # torch.Size(img.shape[0], self.cls_token_len + num_patches + self.intermediate_feature_vector_len + text_seq_length, dim)
        x += self.pos_embedding[:, :(n + 1)] # torch.Size(img.shape[0], self.cls_token_len + num_patches + self.intermediate_feature_vector_len + text_seq_length, dim)
        x = self.dropout(x) # torch.Size(img.shape[0], self.cls_token_len + num_patches + self.intermediate_feature_vector_len + text_seq_length, dim)

        x = self.encoder_transformer(x) # torch.Size(img.shape[0], self.cls_token_len + num_patches + self.intermediate_feature_vector_len + text_seq_length, dim)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # torch.Size(img.shape[0], dim)

        # x = self.to_latent(x) # torch.Size(img.shape[0], dim)
        # x = self.mlp_head(x) # torch.Size(img.shape[0], num_classes)
        # # style_vector = x[:,:int(self.num_classes/2)] # torch.Size(img.shape[0], num_classes/2)
        # # semantic_vector = x[:,int(self.num_classes/2):] # torch.Size(img.shape[0], num_classes/2)
        return x

    def decoding(self, x):
        decoded_x = self.decoder_transformer(x) # torch.Size(img.shape[0],1 +  num_patches + text_seq_length, dim)
        return decoded_x
    
    def compute_loss(self, l1_imgs, l1_texts, l1_language_labels, l2_imgs, l2_texts, l2_language_labels):
        """[summary]

        Args:
            l1_imgs ([type]): a list of PIL.Image.Image
            l1_texts ([type]): a list of strings
            l1_language_labels ([type]): a list of 'l1' or 'l2'
            l2_imgs ([type]): [description]
            l2_texts ([type]): [description]
            l2_language_labels ([type]): [description]

        Returns:
            [type]: losses
        """
        l1_img_sizes =  [img.size for img in l1_imgs]
        l2_img_sizes =  [img.size for img in l2_imgs]
        l1_text_sizes = [len(text) for text in l1_texts]
        l2_text_sizes = [len(text) for text in l2_texts]
        # l1 reconstruction.
        l1_x = self.encoding(l1_imgs, l1_texts, l1_language_labels, l1_img_sizes)
        l1_decoded_x = self.decoding(l1_x)
        # l1 reconstruction loss
        l1_self_pred_img = self.convert_to_img(l1_decoded_x)
        l1_self_pred_text_patches = self.get_text_patches_from_x(l1_decoded_x)
        loss1 = self.compute_img_loss(l1_imgs, l1_self_pred_img, device=self.to_pixels.weight.device)
        loss2 = self.compute_text_loss(l1_texts, l1_self_pred_text_patches)
        loss17 = self.compute_intermediate_feature_loss(l1_decoded_x, l1_language_labels, l1_img_sizes, l1_img_sizes, l1_text_sizes)
        # l1_recon_loss = loss1 + loss2 + loss 17
        
        # l2 reconstruction.
        l2_x = self.encoding(l2_imgs, l2_texts, l2_language_labels, l2_img_sizes)
        l2_decoded_x = self.decoding(l2_x)
        # l2 reconstruction loss
        l2_self_pred_img = self.convert_to_img(l2_decoded_x)
        l2_self_pred_text_patches = self.get_text_patches_from_x(l2_decoded_x)
        loss3 = self.compute_img_loss(l2_imgs, l2_self_pred_img, device=self.to_pixels.weight.device)
        loss4 = self.compute_text_loss(l2_texts, l2_self_pred_text_patches)
        loss18 = self.compute_intermediate_feature_loss(l2_decoded_x, l2_language_labels, l2_img_sizes, l2_img_sizes, l2_text_sizes)
        # l2_recon_loss = loss3 + loss4 + loss18
        
        # Z vector consistency loss.
        l1_zc = self.get_semantic_vector(l1_x) # zc1
        l1_zs = self.get_style_vector(l1_x) # zs1
        l1_zsp = self.l1_to_l2_style_transformer(l1_zs) # zs1'
        l1_zspp = self.l2_to_l1_style_transformer(l1_zsp) # zs1''
        
        l2_zc = self.get_semantic_vector(l2_x) # zc2
        l2_zs = self.get_style_vector(l2_x) # zs2
        l2_zsp = self.l2_to_l1_style_transformer(l2_zs) # zs2'
        l2_zsppp = self.l1_to_l2_style_transformer(l2_zsp) # zs2''
        
        loss5 = self.compute_vector_loss(l1_zs, l1_zspp)
        loss6 = self.compute_vector_loss(l2_zs, l2_zsppp)
        loss7 = self.compute_vector_loss(l1_zs, l2_zsp)
        loss8 = self.compute_vector_loss(l2_zs, l1_zsp)
        
        # twist language text loss.
        l1_img_l2_text_x = self.encoding(l1_imgs, l2_texts, l2_language_labels, l2_img_sizes)
        zs3 = self.get_style_vector(l1_img_l2_text_x)
        zc3 = self.get_semantic_vector(l1_img_l2_text_x)
        loss9 = self.compute_vector_loss(l1_zs, zs3)
        zs3_p = self.l1_to_l2_style_transformer(zs3)
        # same_as_loss8 = self.compute_vector_loss(l2_zs, zs3_p)
        loss10 = self.compute_vector_loss(l2_zc, zc3)
        l1_img_l2_text_x_p = self.cat_style_and_semantic_vectors(zs3_p, zc3)
        l1_img_l2_text_decoded_x = self.decoding(l1_img_l2_text_x_p)
        l1_img_l2_text_pred_img = self.convert_to_img(l1_img_l2_text_decoded_x)
        l1_img_l2_text_pred_text_patches = self.get_text_patches_from_x(l1_img_l2_text_decoded_x)
        loss11 = self.compute_img_loss(l2_imgs, l1_img_l2_text_pred_img, device=self.to_pixels.weight.device)
        loss12 = self.compute_text_loss(l2_texts, l1_img_l2_text_pred_text_patches)
        loss19 = self.compute_intermediate_feature_loss(l1_img_l2_text_decoded_x, l2_language_labels, l1_img_sizes, l2_img_sizes, l2_text_sizes)
        
        l2_img_l1_text_x = self.encoding(l2_imgs, l1_texts, l1_language_labels, l1_img_sizes)
        zs4 = self.get_style_vector(l2_img_l1_text_x)
        zc4 = self.get_semantic_vector(l2_img_l1_text_x)
        loss13 = self.compute_vector_loss(l2_zs, zs4)
        zs4_p = self.l2_to_l1_style_transformer(zs4)
        # same_as_loss7 = self.compute_vector_loss(l1_zs, zs4_p)
        loss14 = self.compute_vector_loss(l1_zc, zc4)
        l2_img_l1_text_x_p = self.cat_style_and_semantic_vectors(zs4_p, zc4)
        l2_img_l1_text_decoded_x = self.decoding(l2_img_l1_text_x_p)
        l2_img_l1_text_pred_img = self.convert_to_img(l2_img_l1_text_decoded_x)
        l2_img_l1_text_pred_text_patches = self.get_text_patches_from_x(l2_img_l1_text_decoded_x)
        loss15 = self.compute_img_loss(l1_imgs, l2_img_l1_text_pred_img, device=self.to_pixels.weight.device)
        loss16 = self.compute_text_loss(l1_texts, l2_img_l1_text_pred_text_patches)
        loss20 = self.compute_intermediate_feature_loss(l2_img_l1_text_decoded_x, l1_language_labels, l2_img_sizes, l1_img_sizes, l1_text_sizes)
        return loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, loss10, loss11, loss12, loss13, loss14, loss15, loss16, loss17, loss18, loss19, loss20
        
    def forward(self, l1_imgs, l2_texts, l2_language_labels, target_image_sizes):
        """[summary]

        Args:
            l1_imgs ([type]): a list of PIL.Image.Image
            l2_texts ([type]): a list of strings
            l2_language_labels ([type]): a list of 'l1' or 'l2'
            target_image_sizes ([type]): a list of (w, h)

        Returns:
            [type]: a list of PIL.Image.Image
        """
        assert len(l1_imgs) == len(l2_texts) and len(l2_texts) == len(l2_language_labels) and len(l2_language_labels) == len(target_image_sizes)
        l1_img_l2_text_x = self.encoding(l1_imgs, l2_texts, l2_language_labels, target_image_sizes)
        zs3 = self.get_style_vector(l1_img_l2_text_x)
        zc3 = self.get_semantic_vector(l1_img_l2_text_x)
        assert l2_language_labels.count(l2_language_labels[0]) == len(l2_language_labels) # all elements in l2_language_labels are equal.
        if l2_language_labels[0] == 'l1':
            zs3_p = self.l2_to_l1_style_transformer(zs3)
        elif l2_language_labels[0] == 'l2':
            zs3_p = self.l1_to_l2_style_transformer(zs3)
        else:
            raise ValueError('l2_language_label must be either "l1" or "l2"')
        l1_img_l2_text_x_p = self.cat_style_and_semantic_vectors(zs3_p, zc3)
        l1_img_l2_text_decoded_x = self.decoding(l1_img_l2_text_x_p)
        l1_img_l2_text_pred_img = self.convert_to_img(l1_img_l2_text_decoded_x, mode='inference')
        converted_imgs = self.processor.postprocess_transform_tensors(l1_img_l2_text_pred_img, target_image_sizes)
        return converted_imgs
    
    def forward_reconstruct(self, l1_imgs, l1_texts, l1_language_labels):
        """[summary]

        Args:
            l1_imgs ([type]): a list of PIL.Image.Image
            l1_texts ([type]): a list of strings
            l1_language_labels ([type]): a list of 'l1' or 'l2'

        Returns:
            [type]: a list of PIL.Image.Image and a list of strings
        """
        assert len(l1_imgs) == len(l1_texts) and len(l1_texts) == len(l1_language_labels)
        l1_img_sizes =  [img.size for img in l1_imgs]
        l1_text_sizes = [len(text) for text in l1_texts]
        x = self.encoding(l1_imgs, l1_texts, l1_language_labels, l1_img_sizes)
        decoded_x = self.decoding(x)
        pred_img = self.convert_to_img(decoded_x, mode='inference')
        converted_imgs = self.processor.postprocess_transform_tensors(pred_img, l1_img_sizes)
        converted_texts = self.convert_to_text(decoded_x)
        return converted_imgs, converted_texts
