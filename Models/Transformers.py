import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)" 
                % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = Attention(hidden_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, dim_feedforward),
                                nn.ReLU(),
                                nn.Linear(dim_feedforward, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, attention_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.fc(src)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)
    def forward(self, src, attention_mask=None):
        for layer in self.layers:
            src = layer(src, attention_mask=attention_mask)
        return src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers=1, nhead=1, dropout=0.1, dim_feedforward=128, max_seq_length=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.encoder = TransformerEncoder(TransformerLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, attention_mask=None):
        seq_length = input.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
        positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand(input.size())
        input = input + positions_embedding
        input = self.norm(input)
        hidden = self.encoder(input, attention_mask=attention_mask)
        # Return ([CLS] token embedding, full hidden sequence, last hidden layer)
        return hidden[:, 0, :], hidden, hidden

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, context_size, nhead=1, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.src_cross_attention = Attention(hidden_size, nhead, dropout, ctx_dim=context_size)
        self.context_cross_attention = Attention(context_size, nhead, dropout, ctx_dim=hidden_size)

        # NOTE: hidden_size + context_size must be divisible by nhead
        self.self_attention = Attention(hidden_size + context_size, nhead, dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + context_size, hidden_size + context_size),
            nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(hidden_size + context_size)
        self.norm2 = nn.LayerNorm(hidden_size + context_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, context, src_key_mask=None, ctx_key_mask=None, time_key_mask=None):
        """
        src:     (B, T, Ds)
        context: (B, T, Dc)  (after you pad-align them in DVlog_Transformer)

        src_key_mask:  additive key mask for src keys      -> (B,1,1,T)
        ctx_key_mask:  additive key mask for context keys  -> (B,1,1,T)
        time_key_mask: additive key mask for self-attn keys -> (B,1,1,T)
        """
        # src queries attend to context keys -> mask context keys
        new_src = self.src_cross_attention(src, context, attention_mask=ctx_key_mask)          # (B,T,Ds)

        # context queries attend to src keys -> mask src keys
        new_context = self.context_cross_attention(context, src, attention_mask=src_key_mask) # (B,T,Dc)

        cross_src = torch.cat((new_src, new_context), dim=2)  # (B,T,Ds+Dc)

        # IMPORTANT: mask PAD keys here so PAD tokens don't affect valid tokens
        cross_src_1 = self.self_attention(cross_src, cross_src, attention_mask=time_key_mask)
        cross_src = cross_src + self.dropout1(cross_src_1)
        cross_src = self.norm1(cross_src)

        cross_src_2 = self.fc(cross_src)
        cross_src = cross_src + self.dropout2(cross_src_2)
        cross_src = self.norm2(cross_src)

        return cross_src

VISUAL_DIM = 136
ACOUSTIC_DIM = 25

class DVlog_Transformer(nn.Module):
    def __init__(self, visual_model, acoustic_model, cross_n_heads=1, dropout=0.1, fusion_dim=128):
        super(DVlog_Transformer, self).__init__()
        self.visual_model = visual_model
        self.acoustic_model = acoustic_model
        self.L_AV = CrossAttentionLayer(ACOUSTIC_DIM, VISUAL_DIM, nhead=cross_n_heads, dropout=dropout)

        total_dim = VISUAL_DIM + ACOUSTIC_DIM + (VISUAL_DIM + ACOUSTIC_DIM)  # 322
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, 2)
        )
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _make_additive_key_mask(valid_mask_bool: torch.Tensor) -> torch.Tensor:
        """
        valid_mask_bool: (B,T) True for valid tokens
        returns additive mask: (B,1,1,T) with 0 for valid, -1e9 for PAD
        """
        return (~valid_mask_bool).float().unsqueeze(1).unsqueeze(2) * -1e9

    @staticmethod
    def _masked_mean_pool(seq: torch.Tensor, valid_mask_bool: torch.Tensor) -> torch.Tensor:
        """
        seq: (B,T,D), valid_mask_bool: (B,T)
        """
        den = valid_mask_bool.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B,1)
        return (seq * valid_mask_bool.unsqueeze(-1)).sum(dim=1) / den

    @staticmethod
    def _pad_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        x: (B,T,D) -> pad T to target_len with zeros
        """
        B, T, D = x.shape
        if T == target_len:
            return x
        pad = x.new_zeros((B, target_len - T, D))
        return torch.cat([x, pad], dim=1)

    @staticmethod
    def _pad_bool_to_len(m: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        m: (B,T) bool -> pad to (B,target_len) with False
        """
        B, T = m.shape
        if T == target_len:
            return m
        pad = torch.zeros((B, target_len - T), dtype=torch.bool, device=m.device)
        return torch.cat([m, pad], dim=1)

    def forward(self, visual, acoustic, return_features=False, attention_mask=None, token_type_ids=None):
        # valid tokens: assume padded frames are all-zeros in the RAW inputs
        vis_valid = (visual.abs().sum(dim=-1) != 0)      # (B,Tv)
        ac_valid  = (acoustic.abs().sum(dim=-1) != 0)    # (B,Ta)

        vis_key_mask = self._make_additive_key_mask(vis_valid)  # (B,1,1,Tv)
        ac_key_mask  = self._make_additive_key_mask(ac_valid)   # (B,1,1,Ta)

        # modality transformers (mask PAD keys)
        _, full_vis, _ = self.visual_model(visual, attention_mask=vis_key_mask)       # (B,Tv,136)
        _, full_ac,  _ = self.acoustic_model(acoustic, attention_mask=ac_key_mask)    # (B,Ta,25)

        if return_features:
            # padding-aware per-modality embeddings
            vis_emb = self._masked_mean_pool(full_vis, vis_valid)  # (B,136)
            ac_emb  = self._masked_mean_pool(full_ac,  ac_valid)   # (B,25)
            return ac_emb, vis_emb

        # ----- Align lengths so CrossAttentionLayer is always valid -----
        Tv = full_vis.size(1)
        Ta = full_ac.size(1)
        T  = max(Tv, Ta)

        full_vis_T = self._pad_to_len(full_vis, T)     # (B,T,136)
        full_ac_T  = self._pad_to_len(full_ac,  T)     # (B,T,25)

        vis_valid_T = self._pad_bool_to_len(vis_valid, T)  # (B,T)
        ac_valid_T  = self._pad_bool_to_len(ac_valid,  T)  # (B,T)

        vis_key_mask_T = self._make_additive_key_mask(vis_valid_T)  # (B,1,1,T)
        ac_key_mask_T  = self._make_additive_key_mask(ac_valid_T)   # (B,1,1,T)

        # for self-attn over the fused time axis: mask timesteps that are PAD in BOTH streams
        fused_valid_T = (vis_valid_T | ac_valid_T)                  # (B,T)
        fused_key_mask_T = self._make_additive_key_mask(fused_valid_T)  # (B,1,1,T)

        # cross-attention (src=acoustic, context=visual) with proper masks
        noverbal = self.L_AV(
            src=full_ac_T,
            context=full_vis_T,
            src_key_mask=ac_key_mask_T,     # when context attends to src keys
            ctx_key_mask=vis_key_mask_T,    # when src attends to context keys
            time_key_mask=fused_key_mask_T  # for self-attn on concatenated sequence
        )  # (B,T,161)

        # pooled embeddings
        vis_emb = self._masked_mean_pool(full_vis_T, vis_valid_T)   # (B,136)
        ac_emb  = self._masked_mean_pool(full_ac_T,  ac_valid_T)    # (B,25)

        # noverbal is over aligned T; use fused_valid_T or ac_valid_T depending on what you want
        # fused_valid_T is safer since noverbal contains both modalities
        lav_emb = self._masked_mean_pool(noverbal, fused_valid_T)   # (B,161)

        fused = torch.cat([vis_emb, ac_emb, lav_emb], dim=1)        # (B,322)
        out = self.fc(fused)                                        # (B,2)
        return self.softmax(out)

