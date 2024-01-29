import torch
import torch.nn as nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
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
        # B, (ph*pw), L/ph*W/pw, C
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # B, (ph*pw), L/ph*W/pw, C --> B, (ph*pw), L/ph*W/pw, inner_dim(dim_head*heads)*3
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        # B, (ph*pw), L/ph*W/pw, inner_dim(dim_head*heads) --> B, (ph*pw), heads, L/ph*W/pw, dim_head

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head] matmul --> [B, (ph*pw), heads, L/ph*W/pw, L/ph*W/pw]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        # [B, (ph*pw), heads, L/ph*W/pw, L/ph*W/pw] X [B, (ph*pw), heads, L/ph*W/pw, dim_head] -->
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head]
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head] --> [B, (ph*pw), L/ph*W/pw, dim_head*heads]
        return self.to_out(out)  # [B, (ph*pw), L/ph*W/pw, dim_head*heads] --> [B, (ph*pw), L/ph*W/pw, C]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        # B, (ph*pw), L/ph*W/pw, C
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def conv_nxn_bn_group(inp, oup, kernal_size=3, stride=1, groups=4):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class ViTBlock(nn.Module):
    def __init__(self, channel, kernel_size, patch_size, groups, depth, mlp_dim, dropout=0.):
        super().__init__()
        '''
        ViTBlock: simplified ViT block, merging channel and dim
        channel:(channels of input), 
        depth:(num of transformer block)[2,4,3],
        kernel_size:(kernel size of convlutional neural networks)
        patch_size:(patch size of transformer)
        heads:(heads number/kernel number)
        att_dim:(nodes of mlp in attention module)
        mlp_dim:(nodes of mlp in feedfward module)
        groups:(groups for convolution)
        dropout
        '''

        self.ph, self.pw = patch_size

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        # Transformer(dim(channels of input), depth(num of transformer block)[2,4,3], 
        #             4(heads number/kernel number), 8(length of mlp in attention),
        #             mlp_dim(nodes of mlp, extension), dropout)
        self.merge_conv = conv_nxn_bn_group(2 * channel, channel, kernel_size, stride=1, groups=groups)
    
    def forward(self, x):
        # input size: B, 4*C, L, W / B, C, D, L, W not included
        y = x.clone()
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # B, C, L, W --> B, (ph*pw), L/ph*W/pw, C
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # x: B, 4*C, L, W 

        # Fusion
        x = torch.cat((x, y), 1)
        # x: B, 4*2C, L, W
        x = self.merge_conv(x)
        # x: B, 4*C, L, W
        return x
    

class ViTBlockV2(nn.Module):
    def __init__(self, channel, kernel_size, patch_size, groups, depth, mlp_dim, dropout=0.):
        super().__init__()
        '''
        ViTBlock: simplified ViT block, merging channel and dim
        channel:(channels of input),
        depth:(num of transformer block)[2,4,3],
        kernel_size:(kernel size of convlutional neural networks)
        patch_size:(patch size of transformer)
        heads:(heads number/kernel number)
        att_dim:(nodes of mlp in attention module)
        mlp_dim:(nodes of mlp in feedfward module)
        groups:(groups for convolution)
        dropout
        '''

        self.ph, self.pw = patch_size

        self.transformer = Transformer(channel*self.ph*self.pw, depth, 8, 64, mlp_dim, dropout)
        # Transformer(dim(channels of input), depth(num of transformer block)[2,4,3], 
        #             4(heads number/kernel number), 8(length of mlp in attention),
        #             mlp_dim(nodes of mlp, extension), dropout)
        self.merge_conv = conv_nxn_bn_group(2 * channel, channel, kernel_size, stride=1, groups=groups)
    
    def forward(self, x):
        # input size: B, 4*C, L, W / B, C, D, L, W not included
        y = x.clone()
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b  (h w) (ph pw d)', ph=self.ph, pw=self.pw)
        # B, C, L, W --> B, (ph*pw), L/ph*W/pw, C
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) (ph pw d) -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # x: B, 4*C, L, W 

        # Fusion
        x = torch.cat((x, y), 1)
        # x: B, 4*2C, L, W
        x = self.merge_conv(x)
        # x: B, 4*C, L, W
        return x
    

class ParrellelAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        v = x.clone()
        qk = self.to_qkv(x).chunk(2, dim=-1)
        q, k= map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qk)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class ParrellelTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + ff(x) # + x # not in the original similifying tranformer blocks
        return x
    


class ParrellelViTBlock(nn.Module):
    def __init__(self, channel, kernel_size, patch_size, groups, depth, mlp_dim, dropout=0.):
        super().__init__()
        '''
        ViTBlock: simplified ViT block, merging channel and dim
        channel:(channels of input), 
        depth:(num of transformer block)[2,4,3],
        kernel_size:(kernel size of convlutional neural networks)
        patch_size:(patch size of transformer)
        heads:(heads number/kernel number)
        att_dim:(nodes of mlp in attention module)
        mlp_dim:(nodes of mlp in feedfward module)
        groups:(groups for convolution)
        dropout
        '''

        self.ph, self.pw = patch_size

        self.transformer = ParrellelTransformer(channel, depth, 4, 8, mlp_dim, dropout)
        # Transformer(dim(channels of input), depth(num of transformer block)[2,4,3], 
        #             4(heads number/kernel number), 8(length of mlp in attention),
        #             mlp_dim(nodes of mlp, extension), dropout)
        self.merge_conv = conv_nxn_bn_group(2 * channel, channel, kernel_size, stride=1, groups=groups)
    
    def forward(self, x):
        # input size: B, 4*C, L, W / B, C, D, L, W not included
        y = x.clone()
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # x: B, 4*C, L, W 

        # Fusion
        x = torch.cat((x, y), 1)
        # x: B, 4*2C, L, W
        x = self.merge_conv(x)
        # x: B, 4*C, L, W
        return x