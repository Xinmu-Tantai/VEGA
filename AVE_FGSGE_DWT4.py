import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleConvMlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super(MultiScaleConvMlp, self).__init__()

        self.dwconv1 = nn.Conv2d(in_features, in_features // 2, 1, 1)
        self.dwconv2 = nn.Conv2d(in_features, in_features // 4, kernel_size=3, stride=1, padding=1)
        self.dwconv3 = nn.Conv2d(in_features, in_features // 4, kernel_size=7, stride=1, padding=3)

        self.act = act_layer()

        self.fc1 = nn.Conv2d(in_features, in_features * 4, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_features * 4, in_features, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = blc2bchw(x, H, W)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = bchw2blc(x, H, W)
        return x


def blc2bchw(x, h, w):
    b, l, c = x.shape
    assert l == h * w, "in blc to bchw, h*w != l."
    return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def bchw2blc(x, h, w):
    b, c, _, _ = x.shape
    return x.permute(0, 2, 3, 1).view(b, -1, c).contiguous()


def window_partition(x, window_size): 
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C).contiguous()
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C).contiguous()
    return windows


def window_reverse(windows, window_size, H, W): 
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1).contiguous()
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).contiguous()
    return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.): 

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class WindowMSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., use_relative_pe=False):

        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.use_relative_pe = use_relative_pe
        if self.use_relative_pe: 
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))   
 
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))   
            coords_flatten = torch.flatten(coords, 1)  
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]   
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()   
            relative_coords[:, :, 0] += self.window_size[0] - 1   
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)   
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): 
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_relative_pe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)   
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()   
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LocalToGlobalRefinement(nn.Module): 
    def __init__(self, dim, heads):
        super(LocalToGlobalRefinement, self).__init__()
        self.dim = dim
 
        self.local_aggregation = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
 
        self.global_h_refine = nn.Conv2d(dim, dim, kernel_size=2, stride=2, padding=0)
        self.global_w_refine = nn.Conv2d(dim, dim, kernel_size=2, stride=2, padding=0)
 
        self.refine_attn = WindowMSA(dim, (4, 4), heads)
 
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, local_feat, global_feat_h, global_feat_w, border_index, H, W): 
        local_key = self.local_aggregation(local_feat.permute(0, 3, 1, 2).contiguous())   
 
        local_bhwc = local_key.permute(0, 2, 3, 1).contiguous()   
        local_h_border = torch.index_select(local_bhwc, 1, border_index)   
        local_w_border = torch.index_select(local_bhwc, 2, border_index)   
 
        local_h_border = local_h_border.permute(0, 3, 1, 2).contiguous()
        local_w_border = local_w_border.permute(0, 3, 2, 1).contiguous()

        local_h_down = self.global_h_refine(local_h_border)  
        local_w_down = self.global_w_refine(local_w_border)    

        global_h_refined = self.fusion(torch.cat([global_feat_h, local_h_down], dim=1))
        global_w_refined = self.fusion(torch.cat([global_feat_w, local_w_down], dim=1))
 
        b_, c_, h_, w_ = global_h_refined.shape
 
        global_h_bhwc = global_h_refined.permute(0, 2, 3, 1).contiguous()
        global_h_windows = window_partition(global_h_bhwc, [1, w_]).view(-1, 1 * w_, c_).contiguous()
        global_h_attn = self.refine_attn(global_h_windows)
        global_h_attn = window_reverse(global_h_attn, [1, w_], h_, w_).permute(0, 3, 1, 2).contiguous()
 
        global_w_bhwc = global_w_refined.permute(0, 2, 3, 1).contiguous()
        global_w_windows = window_partition(global_w_bhwc, [1, w_]).view(-1, 1 * w_, c_).contiguous()
        global_w_attn = self.refine_attn(global_w_windows)
        global_w_attn = window_reverse(global_w_attn, [1, w_], h_, w_).permute(0, 3, 2, 1).contiguous()

        return global_h_attn, global_w_attn


class LSRFormerBidirectional(nn.Module): 

    def __init__(self, dim, heads):
        super(LSRFormerBidirectional, self).__init__()

        self.dim = dim
        self.channel_ratio = 2
 
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(dim, dim // self.channel_ratio, 2, 2, 0, groups=dim // 8),
            nn.BatchNorm2d(dim // self.channel_ratio))
 
        self.h_conv = nn.Conv2d(dim // self.channel_ratio, dim // self.channel_ratio, 2, 2, 0)
        self.w_conv = nn.Conv2d(dim // self.channel_ratio, dim // self.channel_ratio, 2, 2, 0)
 
        self.global_attn = WindowMSA(dim // self.channel_ratio, (4, 4), heads)
        self.local_attn = WindowMSA(dim // self.channel_ratio, (4, 4), heads, use_relative_pe=True)
        self.local_to_global = LocalToGlobalRefinement(dim // self.channel_ratio, heads)
        self.global_reinjection = nn.Sequential(
            nn.Conv2d(dim // self.channel_ratio * 2, dim // self.channel_ratio, kernel_size=1),
            nn.BatchNorm2d(dim // self.channel_ratio),
            nn.GELU()
        )
 
        self.conv_out = nn.Conv2d(dim // self.channel_ratio, dim, 3, 1, 1, groups=dim // 8)
 
        self.mlp = MultiScaleConvMlp(in_features=dim // self.channel_ratio, act_layer=nn.GELU, drop=0.1)
        self.norm1 = nn.LayerNorm(dim // self.channel_ratio)
        self.norm2 = nn.LayerNorm(dim // self.channel_ratio)

        self.apply(self._init_weights)

        self.enable_glb2loc = True    
        self.enable_loc2glb = True   

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def get_index(self, real_h: int): 
        if real_h <= 1:
            return [0]
        if real_h <= 4:
            return [0, real_h - 1]

        index = []
        windows = real_h // 4
        for i in range(windows):
            if i == 0:
                index.append(4 - 1)
            elif i == windows - 1:
                index.append(real_h - 4)
            else:
                index.append(i * 4)
                index.append(i * 4 + 3)
 
        if len(index) == 1:
            index = [0, index[0]] if index[0] != 0 else [0, real_h - 1]

        return index


    def forward(self, x): 
        x_reduction = self.conv_reduce(x)  
        x_reduction = x_reduction.permute(0, 2, 3, 1).contiguous()   
 
        H, W = x_reduction.shape[1], x_reduction.shape[2]
        pad_r = int(((4 - W % 4) % 4) / 2)
        pad_b = pad_l = pad_t = pad_r
        if pad_r > 0:
            x_reduction = F.pad(x_reduction, (0, 0, pad_l, pad_r, pad_t, pad_b), mode='reflect')
 
        border_index = torch.Tensor(self.get_index(x_reduction.shape[2])).int().to(x_reduction.device)
 
        if border_index.numel() < 2: 
            local_windows = window_partition(x_reduction, [4, 4]).view(-1, 16, x_reduction.shape[3]).contiguous()
            local_windows = self.local_attn(local_windows)
            local_features = window_reverse(local_windows, [4, 4], x_reduction.shape[1], x_reduction.shape[2]).contiguous()
            
 
        x_h = torch.index_select(x_reduction, 1, border_index).permute(0, 3, 1, 2).contiguous()
        x_h = self.h_conv(x_h).permute(0, 2, 3, 1).contiguous()
        b_, h_, w_, c_ = x_h.shape
        x_h_windows = window_partition(x_h, [1, w_]).view(-1, 1 * w_, c_).contiguous()
 
        x_w = torch.index_select(x_reduction, 2, border_index).permute(0, 3, 2, 1).contiguous()
        x_w = self.w_conv(x_w).permute(0, 2, 3, 1).contiguous()
        x_w_windows = window_partition(x_w, [1, w_]).view(-1, 1 * w_, c_).contiguous()
 
        x_total = torch.cat([x_h_windows, x_w_windows], dim=0)
        x_h_attn, x_w_attn = torch.chunk(self.global_attn(x_total), 2, 0)
 
        x_h_global = window_reverse(x_h_attn, [1, w_], h_, w_).permute(0, 3, 1, 2).contiguous() 
        x_w_global = window_reverse(x_w_attn, [1, w_], h_, w_).permute(0, 3, 1, 2).contiguous()
 
        x_h_guide = F.interpolate(x_h_global, scale_factor=2, mode='bilinear', align_corners=True)
        x_w_guide = F.interpolate(x_w_global, scale_factor=2, mode='bilinear', align_corners=True)
        x_h_guide = x_h_guide.permute(0, 2, 3, 1).contiguous() 
        x_w_guide = x_w_guide.permute(0, 3, 2, 1).contiguous()

  
        if getattr(self, "enable_glb2loc", True):
            x_reduction.index_add_(1, border_index, x_h_guide)
            x_reduction.index_add_(2, border_index, x_w_guide)
 
        local_windows = window_partition(x_reduction, [4, 4]).view(-1, 16, x_reduction.shape[3]).contiguous()
        local_windows = self.local_attn(local_windows)
        local_features = window_reverse(local_windows, [4, 4], x_reduction.shape[1], x_reduction.shape[2]).contiguous()
  
        if getattr(self, "enable_loc2glb", True):
            global_h_refined, global_w_refined = self.local_to_global(
                local_features, x_h_global, x_w_global, border_index, H, W
            )

            global_h_refined_up = F.interpolate(global_h_refined, scale_factor=2, mode='bilinear', align_corners=True)
            global_w_refined_up = F.interpolate(global_w_refined, scale_factor=2, mode='bilinear', align_corners=True)

            global_h_refined_up = global_h_refined_up.permute(0, 2, 3, 1).contiguous()
            global_w_refined_up = global_w_refined_up.permute(0, 2, 3, 1).contiguous()

            local_features.index_add_(1, border_index, global_h_refined_up)
            local_features.index_add_(2, border_index, global_w_refined_up)

 
        if pad_r > 0:
            x_reduction = x_reduction[:, pad_t:H + pad_t, pad_l:W + pad_t, :].contiguous()
            local_features = local_features[:, pad_t:H + pad_t, pad_l:W + pad_t, :].contiguous()
 
        bb, hh, ww, cc = local_features.shape
        local_windows_flat = local_features.view(bb, hh * ww, cc).contiguous()
 
        local_windows_flat = local_windows_flat + self.mlp(self.norm1(local_windows_flat), hh, ww)
 
        local_features_final = local_windows_flat.view(bb, hh, ww, cc).contiguous() + x_reduction
        local_features_final = local_features_final.permute(0, 3, 1, 2).contiguous()
 
        out = F.interpolate(local_features_final, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv_out(out)

        return out

# -----------------------------------------------------------------------------
# FG-SGE: Frequency-guided Structure Graph Evidence Module
# -----------------------------------------------------------------------------
# The following classes extend the original LSRFormerBidirectional module with:
# 1) feature-level Haar DWT frequency decomposition;
# 2) adaptive low/mid/high frequency fusion;
# 3) visual evidence node construction;
# 4) dynamic multi-relation structure graph construction;
# 5) graph-guided evidence refinement;
# 6) graph-enhanced visual evidence field output.
# -----------------------------------------------------------------------------


class HaarDWT2D(nn.Module):
    """Differentiable 2D Haar DWT implemented with tensor slicing.

    Input:  x  [B, C, H, W]
    Output: LL, LH, HL, HH, each with spatial size ceil(H/2) x ceil(W/2).

    The scaling factor 0.5 keeps the transform numerically stable and makes the
    inverse-energy scale comparable to the input feature magnitude.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        if H < 2 or W < 2:
            zero = torch.zeros_like(x)
            return x, zero, zero, zero

        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class FrequencyAdaptiveDWTFusion(nn.Module):
    """Feature-level 2D DWT decomposition and adaptive four-subband fusion.

    A one-level 2D DWT decomposes a feature map into four sub-bands:
        LL : low-frequency approximation, mainly global layout and large regions;
        LH : vertical-direction detail responses;
        HL : horizontal-direction detail responses;
        HH : diagonal high-frequency detail responses.

    We keep the four sub-bands explicitly instead of directly naming them as
    low/mid/high. The semantic use is performed after projection and adaptive
    fusion: LL mainly supports scene-level evidence, while LH/HL/HH provide
    directional structural evidence for boundaries, linear structures, textures,
    and small objects.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dwt = HaarDWT2D()

        self.ll_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.lh_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.hl_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.hh_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        hidden = max(dim // reduction, 4)
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )

        self.spatial_freq_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape
        out_size = (H, W)

        # Strict one-level 2D DWT: LL, LH, HL, HH.
        ll, lh, hl, hh = self.dwt(x)

        # Restore each sub-band to the input feature resolution for node-level
        # evidence construction and graph compatibility computation.
        ll = self.ll_proj(self._resize(ll, out_size))
        lh = self.lh_proj(self._resize(lh, out_size))
        hl = self.hl_proj(self._resize(hl, out_size))
        hh = self.hh_proj(self._resize(hh, out_size))

        # Image-adaptive four-subband weights: [B, 4, 1, 1].
        # Order: LL, LH, HL, HH.
        weight = torch.softmax(self.freq_gate(x), dim=1)
        freq = (
            weight[:, 0:1] * ll +
            weight[:, 1:2] * lh +
            weight[:, 2:3] * hl +
            weight[:, 3:4] * hh
        )

        # Residual spatial-frequency fusion.
        x_sf = x + self.spatial_freq_fuse(torch.cat([x, freq], dim=1))

        # Four-dimensional local frequency descriptor for graph construction:
        # [B, 4, H, W], corresponding to LL/LH/HL/HH response strengths.
        freq_desc = torch.cat([
            ll.abs().mean(dim=1, keepdim=True),
            lh.abs().mean(dim=1, keepdim=True),
            hl.abs().mean(dim=1, keepdim=True),
            hh.abs().mean(dim=1, keepdim=True),
        ], dim=1)

        # Return flattened weights in order [LL, LH, HL, HH].
        return x_sf, freq_desc, weight.flatten(1)


class DynamicMultiRelationGraphRefinement(nn.Module):
    """Dynamic multi-relation graph construction and graph-guided refinement.

    Relations:
        adj  : spatial adjacency, preserving local spatial continuity;
        sem  : semantic similarity, linking repeated or same-class regions;
        co   : scene-level co-occurrence, linking complementary land-cover parts;
        freq : frequency compatibility, linking similar morphology/texture patterns.

    To avoid O((HW)^2) memory on large feature maps, the graph is built on an
    adaptively pooled evidence map when HW > max_graph_nodes, and the graph-refined
    evidence is then upsampled back to the original feature resolution.
    """

    def __init__(
        self,
        dim,
        heads=4,
        k=8,
        max_graph_nodes=256,
        freq_dim=4,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads."
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.k = k
        self.max_graph_nodes = max_graph_nodes

        self.node_norm = nn.LayerNorm(dim)
        self.freq_proj = nn.Linear(freq_dim, dim)
        self.co_context = nn.Linear(dim, dim)

        self.relations = ["adj", "sem", "co", "freq"]
        self.q_proj = nn.ModuleDict({r: nn.Linear(dim, dim) for r in self.relations})
        self.k_proj = nn.ModuleDict({r: nn.Linear(dim, dim) for r in self.relations})
        self.v_proj = nn.ModuleDict({r: nn.Linear(dim, dim) for r in self.relations})

        self.attn_drop = nn.Dropout(attn_drop)
        self.fuse = nn.Sequential(
            nn.Linear(dim * len(self.relations), dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(dim, dim),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(dim * 4, dim),
        )
        self.map_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def _target_hw(self, H, W):
        n = H * W
        if n <= self.max_graph_nodes:
            return H, W
        scale = math.sqrt(float(self.max_graph_nodes) / float(n))
        gh = max(1, int(round(H * scale)))
        gw = max(1, int(round(W * scale)))
        while gh * gw > self.max_graph_nodes:
            if gh >= gw and gh > 1:
                gh -= 1
            elif gw > 1:
                gw -= 1
            else:
                break
        return gh, gw

    def _make_positions(self, B, H, W, device, dtype):
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([yy, xx], dim=-1).view(1, H * W, 2)
        return pos.expand(B, -1, -1).contiguous()

    def _flatten_map(self, x):
        return x.flatten(2).transpose(1, 2).contiguous()

    def _mask_self(self, score):
        B, N, _ = score.shape
        if N <= 1:
            return score
        eye = torch.eye(N, device=score.device, dtype=torch.bool).unsqueeze(0)
        return score.masked_fill(eye, float("-inf"))

    def _topk(self, score, exclude_self=True):
        B, N, _ = score.shape
        if N == 1:
            idx = torch.zeros(B, 1, 1, dtype=torch.long, device=score.device)
            val = torch.ones(B, 1, 1, dtype=score.dtype, device=score.device)
            return idx, val

        if exclude_self:
            score = self._mask_self(score)
            k = min(self.k, N - 1)
        else:
            k = min(self.k, N)
        val, idx = torch.topk(score, k=k, dim=-1)
        val = torch.nan_to_num(val, neginf=-1e4, posinf=1e4)
        return idx, val

    def _gather_neighbors(self, x, idx):
        # x: [B, N, C], idx: [B, N, K] -> [B, N, K, C]
        B, N, C = x.shape
        K = idx.shape[-1]
        idx = idx.unsqueeze(-1).expand(B, N, K, C)
        x_expand = x.unsqueeze(1).expand(B, N, N, C)
        return torch.gather(x_expand, dim=2, index=idx)

    def _cosine_score(self, x):
        x = F.normalize(x, dim=-1)
        return torch.bmm(x, x.transpose(1, 2))

    def _build_relations(self, nodes, freq_nodes, pos):
        B, N, C = nodes.shape

        # 1) Spatial adjacency relation.
        dist = torch.cdist(pos, pos, p=2)
        sigma = 0.25
        adj_score = torch.exp(-(dist ** 2) / (sigma ** 2))
        adj_idx, adj_val = self._topk(adj_score, exclude_self=True)

        # 2) Semantic similarity relation.
        sem_score = self._cosine_score(nodes)
        sem_idx, sem_val = self._topk(sem_score, exclude_self=True)

        # 3) Scene-level co-occurrence relation, conditioned by global context.
        g = nodes.mean(dim=1)
        co_nodes = nodes + self.co_context(g).unsqueeze(1)
        co_q = F.normalize(self.q_proj["co"](co_nodes), dim=-1)
        co_k = F.normalize(self.k_proj["co"](co_nodes), dim=-1)
        co_score = torch.bmm(co_q, co_k.transpose(1, 2))
        co_idx, co_val = self._topk(co_score, exclude_self=True)

        # 4) Frequency compatibility relation.
        freq_score = self._cosine_score(freq_nodes)
        freq_idx, freq_val = self._topk(freq_score, exclude_self=True)

        return {
            "adj": (adj_idx, adj_val),
            "sem": (sem_idx, sem_val),
            "co": (co_idx, co_val),
            "freq": (freq_idx, freq_val),
        }

    def _relation_attention(self, nodes, idx, edge_bias, relation):
        B, N, C = nodes.shape
        K = idx.shape[-1]

        q = self.q_proj[relation](nodes).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k_all = self.k_proj[relation](nodes)
        v_all = self.v_proj[relation](nodes)

        k = self._gather_neighbors(k_all, idx).view(B, N, K, self.heads, self.head_dim)
        v = self._gather_neighbors(v_all, idx).view(B, N, K, self.heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4).contiguous()  # [B, heads, N, K, head_dim]
        v = v.permute(0, 3, 1, 2, 4).contiguous()

        logits = (q.unsqueeze(3) * k).sum(dim=-1) * self.scale
        logits = logits + edge_bias.unsqueeze(1)
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * v).sum(dim=3)
        out = out.transpose(1, 2).reshape(B, N, C)
        return out

    def forward(self, feat_map, freq_desc):
        B, C, H, W = feat_map.shape
        gh, gw = self._target_hw(H, W)

        if (gh, gw) != (H, W):
            graph_feat = F.adaptive_avg_pool2d(feat_map, (gh, gw))
            graph_freq = F.adaptive_avg_pool2d(freq_desc, (gh, gw))
        else:
            graph_feat = feat_map
            graph_freq = freq_desc

        nodes = self._flatten_map(graph_feat)  # [B, N, C]
        nodes = self.node_norm(nodes)
        freq_nodes = self._flatten_map(graph_freq)  # [B, N, 3]
        freq_nodes = self.freq_proj(freq_nodes)
        pos = self._make_positions(B, gh, gw, feat_map.device, feat_map.dtype)

        relation_graphs = self._build_relations(nodes, freq_nodes, pos)
        relation_outs = []
        for relation in self.relations:
            idx, val = relation_graphs[relation]
            relation_outs.append(self._relation_attention(nodes, idx, val, relation))

        fused = self.fuse(torch.cat(relation_outs, dim=-1))
        refined_nodes = nodes + fused
        refined_nodes = refined_nodes + self.ffn(refined_nodes)

        refined_map = refined_nodes.transpose(1, 2).view(B, C, gh, gw).contiguous()
        if (gh, gw) != (H, W):
            refined_map = F.interpolate(refined_map, size=(H, W), mode="bilinear", align_corners=False)

        graph_enhanced_map = feat_map + self.map_fuse(torch.cat([feat_map, refined_map], dim=1))
        graph_evidence = self._flatten_map(graph_enhanced_map)

        return graph_enhanced_map, graph_evidence, relation_graphs


class FGSGE_LSRFormerBidirectional(LSRFormerBidirectional):
    """LSRFormerBidirectional enhanced with FG-SGE.

    Args:
        dim: input/output channel dimension.
        heads: attention heads used in the original bidirectional module.
        graph_heads: heads for multi-relation graph attention.
        graph_k: top-k neighbors retained for each relation.
        max_graph_nodes: maximum nodes used for dynamic graph construction.
        return_evidence: if True, forward returns a dict containing the normal
            output feature, graph-enhanced evidence tokens, and graph metadata.
            if False, forward returns only the output tensor for drop-in use.
    """

    def __init__(
        self,
        dim,
        heads,
        graph_heads=4,
        graph_k=8,
        max_graph_nodes=256,
        return_evidence=False,
    ):
        super().__init__(dim=dim, heads=heads)
        reduced_dim = dim // self.channel_ratio
        self.return_evidence = return_evidence

        self.freq_fusion = FrequencyAdaptiveDWTFusion(reduced_dim)
        self.graph_refine = DynamicMultiRelationGraphRefinement(
            dim=reduced_dim,
            heads=graph_heads,
            k=graph_k,
            max_graph_nodes=max_graph_nodes,
            freq_dim=4,
        )

        # Re-initialize only the newly added modules.
        self.freq_fusion.apply(self._init_weights)
        self.graph_refine.apply(self._init_weights)

    def forward(self, x):
        # ------------------------------------------------------------------
        # 1. Original channel/spatial reduction.
        # ------------------------------------------------------------------
        x_reduction = self.conv_reduce(x)  # [B, C/2, H/2, W/2]

        # ------------------------------------------------------------------
        # 2. Strict LL/LH/HL/HH DWT decomposition + adaptive sub-band fusion.
        # ------------------------------------------------------------------
        x_reduction, freq_desc, freq_weight = self.freq_fusion(x_reduction)

        # The original bidirectional module uses BHWC layout internally.
        x_reduction = x_reduction.permute(0, 2, 3, 1).contiguous()

        # ------------------------------------------------------------------
        # 3. Original global-local bidirectional evidence interaction.
        # ------------------------------------------------------------------
        H, W = x_reduction.shape[1], x_reduction.shape[2]
        pad_r = int(((4 - W % 4) % 4) / 2)
        pad_b = pad_l = pad_t = pad_r
        if pad_r > 0:
            x_reduction = F.pad(x_reduction, (0, 0, pad_l, pad_r, pad_t, pad_b), mode="reflect")
            freq_desc = F.pad(freq_desc, (pad_l, pad_r, pad_t, pad_b), mode="reflect")

        border_index = torch.Tensor(self.get_index(x_reduction.shape[2])).int().to(x_reduction.device)

        x_h = torch.index_select(x_reduction, 1, border_index).permute(0, 3, 1, 2).contiguous()
        x_h = self.h_conv(x_h).permute(0, 2, 3, 1).contiguous()
        b_, h_, w_, c_ = x_h.shape
        x_h_windows = window_partition(x_h, [1, w_]).view(-1, 1 * w_, c_).contiguous()

        x_w = torch.index_select(x_reduction, 2, border_index).permute(0, 3, 2, 1).contiguous()
        x_w = self.w_conv(x_w).permute(0, 2, 3, 1).contiguous()
        x_w_windows = window_partition(x_w, [1, w_]).view(-1, 1 * w_, c_).contiguous()

        x_total = torch.cat([x_h_windows, x_w_windows], dim=0)
        x_h_attn, x_w_attn = torch.chunk(self.global_attn(x_total), 2, 0)

        x_h_global = window_reverse(x_h_attn, [1, w_], h_, w_).permute(0, 3, 1, 2).contiguous()
        x_w_global = window_reverse(x_w_attn, [1, w_], h_, w_).permute(0, 3, 1, 2).contiguous()

        x_h_guide = F.interpolate(x_h_global, scale_factor=2, mode="bilinear", align_corners=True)
        x_w_guide = F.interpolate(x_w_global, scale_factor=2, mode="bilinear", align_corners=True)
        x_h_guide = x_h_guide.permute(0, 2, 3, 1).contiguous()
        x_w_guide = x_w_guide.permute(0, 3, 2, 1).contiguous()

        if getattr(self, "enable_glb2loc", True):
            x_reduction.index_add_(1, border_index, x_h_guide)
            x_reduction.index_add_(2, border_index, x_w_guide)

        local_windows = window_partition(x_reduction, [4, 4]).view(-1, 16, x_reduction.shape[3]).contiguous()
        local_windows = self.local_attn(local_windows)
        local_features = window_reverse(local_windows, [4, 4], x_reduction.shape[1], x_reduction.shape[2]).contiguous()

        if getattr(self, "enable_loc2glb", True):
            global_h_refined, global_w_refined = self.local_to_global(
                local_features, x_h_global, x_w_global, border_index, H, W
            )

            global_h_refined_up = F.interpolate(global_h_refined, scale_factor=2, mode="bilinear", align_corners=True)
            global_w_refined_up = F.interpolate(global_w_refined, scale_factor=2, mode="bilinear", align_corners=True)

            global_h_refined_up = global_h_refined_up.permute(0, 2, 3, 1).contiguous()
            global_w_refined_up = global_w_refined_up.permute(0, 2, 3, 1).contiguous()

            local_features.index_add_(1, border_index, global_h_refined_up)
            local_features.index_add_(2, border_index, global_w_refined_up)

        if pad_r > 0:
            x_reduction = x_reduction[:, pad_t:H + pad_t, pad_l:W + pad_t, :].contiguous()
            local_features = local_features[:, pad_t:H + pad_t, pad_l:W + pad_t, :].contiguous()
            freq_desc = freq_desc[:, :, pad_t:H + pad_t, pad_l:W + pad_t].contiguous()

        bb, hh, ww, cc = local_features.shape
        local_windows_flat = local_features.view(bb, hh * ww, cc).contiguous()
        local_windows_flat = local_windows_flat + self.mlp(self.norm1(local_windows_flat), hh, ww)
        local_features_final = local_windows_flat.view(bb, hh, ww, cc).contiguous() + x_reduction
        local_features_final = local_features_final.permute(0, 3, 1, 2).contiguous()

        # ------------------------------------------------------------------
        # 4. Visual evidence node construction + dynamic multi-relation graph
        #    construction + graph-guided evidence refinement.
        # ------------------------------------------------------------------
        graph_feature, graph_evidence, relation_graphs = self.graph_refine(local_features_final, freq_desc)

        # ------------------------------------------------------------------
        # 5. Output graph-enhanced visual evidence field in original resolution.
        # ------------------------------------------------------------------
        out = F.interpolate(graph_feature, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.conv_out(out)

        if self.return_evidence:
            # out_evidence is the final graph-enhanced evidence field after
            # restoring the original channel dimension, suitable for F-SAMGA
            # when the text branch expects dim-dimensional visual tokens.
            out_evidence = out.flatten(2).transpose(1, 2).contiguous()
            return {
                "out": out,
                "out_evidence": out_evidence,
                "evidence": graph_evidence,
                "graph_feature": graph_feature,
                "freq_weight": freq_weight,  # [B, 4], ordered as [LL, LH, HL, HH]
                "relation_graphs": relation_graphs,
            }
        return out
