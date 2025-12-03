import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttentionFromScratch(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout = 0.1):
        super().__init__()
        assert embedding_size%num_heads == 0, "Model embedding size is not compatible with the number of heads"
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dimension = embedding_size//num_heads

        self.Wq = nn.Linear(embedding_size, embedding_size)
        self.Wk = nn.Linear(embedding_size, embedding_size)
        self.Wv = nn.Linear(embedding_size, embedding_size)
        self.Wo = nn.Linear(embedding_size, embedding_size)

        self.drop = nn.Dropout(dropout)
    
    def forward(self, q, k = None, v = None):
        
        if k == None:
            k = q
        if v == None:
            v = k

        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)

        #print("Q size: ", Q.size)

        batch_times_column, num_rows_q, embed = Q.size()
        num_rows_k = K.size(1)

        #split them into heads 
        Q = Q.view(batch_times_column, num_rows_q, self.num_heads, self.head_dimension).transpose(1,2)
        K = K.view(batch_times_column, num_rows_k, self.num_heads, self.head_dimension).transpose(1,2)
        V = V.view(batch_times_column, num_rows_k, self.num_heads, self.head_dimension).transpose(1,2)

        attn_scores = torch.matmul(Q,K.transpose(-2,-1))/(self.head_dimension**0.5)

        attn_scores = torch.softmax(attn_scores, dim=-1) #attn_scores.shape = (batch x columns, num_heads, num_rows_q, num_rows_k)

        attn_scores = self.drop(attn_scores)

       
        weighted_sum = torch.matmul(attn_scores, V)  #weighted_sum.shape = (batch x colums, num_heads, num_rows_k, d_k)
        #print("Q shape:", Q.shape, flush=True)
        #print("K shape:", K.shape, flush = True)
        #print("V shape:", V.shape, flush = True)
        #print("attn_scores shape:", attn_scores.shape, flush=True)
        #print("weighted_sum shape:", weighted_sum.shape, flush=True)
        #print("batch x col = ", batch_times_column, " num rows k = ", num_rows_k, " embedding size = ", self.embedding_size)

        transposed = weighted_sum.transpose(1,2)
        #print("transposed size = ", transposed.shape)
        output = weighted_sum.transpose(1,2).contiguous().view(batch_times_column, num_rows_q, self.embedding_size)

        output = self.Wo(output)

        return output, attn_scores  #return un tuple meme si on n'utilise pas les raw scores pour rester compatibles avec les implementations pytorch
    

    

class LocalSlidingWindowAttention(nn.Module):
    ''' Sparse sliding window attention codee a la main en creant les windows 
    '''
    def __init__(self, embedding_size, num_heads, window_size=20, dropout=0.1):
        super().__init__()

        assert embedding_size % num_heads == 0, "Model embedding size is not compatible with the number of heads"
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.window = window_size

        self.Wq = nn.Linear(embedding_size, embedding_size)
        self.Wk = nn.Linear(embedding_size, embedding_size)
        self.Wv = nn.Linear(embedding_size, embedding_size)
        self.Wo = nn.Linear(embedding_size, embedding_size)

        self.drop = nn.Dropout(dropout)
        print("window_size", window_size, flush = True)

    def forward(self, q, k=None, v=None):
        if k is None: k = q
        if v is None: v = k

        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)


        batch_times_colums, num_rows_q, E = Q.shape   # (batch_times_column, num_rows, embed)
        num_rows_k = K.size(1)

        Q = Q.view(batch_times_colums, num_rows_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_times_colums, num_rows_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_times_colums, num_rows_k, self.num_heads, self.head_dim).transpose(1, 2)
    

        #il faut faire du padding pour que chaque token ait une window entiere 
        pad = self.window
        K_padded = F.pad(K, (0, 0, pad, pad))  # on pad K et V dans l'avant derniere dimension (celle qui represente le nombre de lignes)
        V_padded = F.pad(V, (0, 0, pad, pad))

        #on cree une sliding window
        # K_padded a  une dimension (Bxc, num_heads, num_rows_k + 2*pad, dim_k)
        # K_windowed a une dimension (BxC, num_heads, num_rows_k, 2*pad +1, dim_k)
        K_win = K_padded.unfold(dimension=2, size=2 * pad + 1, step=1)
        V_win = V_padded.unfold(dimension=2, size=2 * pad + 1, step=1)



        # 4. Compute local attention scores
        # Q: (B, H, N, 1, D)
        Q_exp = Q.unsqueeze(-2)

        # Dot product within local window
        # attn_scores: (B, H, N, window*2+1)

        #print("Q shape:", Q.shape, flush=True)
        #print("K shape:", K.shape, flush = True)
        #print("K padded shape:", K_padded.shape, flush = True)
        #print("K windowed shape:", K_win.shape, flush = True)
        #print("K_win transposed shape:", K_win.transpose(-2,-1).shape, flush = True)
        #print("V windowed shape:", V_win.shape, flush = True)
        #print("Q_exp shape:", Q_exp.shape, flush = True)
        #print("batch x col = ", batch_times_colums, " num rows k = ", num_rows_k, "num_rows_q = ", num_rows_q,  " embedding size = ", self.embedding_size)

        attn_scores = torch.matmul(Q_exp, K_win).squeeze(-2)
        attn_scores /= (self.head_dim ** 0.5)
        #print("attn_scores shape:", attn_scores.shape, flush = True)

        

        # 5. Softmax on the window only (not full sequence)
        attn_probs = self.drop(F.softmax(attn_scores, dim=-1))

        # 6. Weighted sum inside the window
        # attn_probs: (B, H, N, W)
        # V_win:      (B, H, N, W, D)
        out = torch.matmul(attn_probs.unsqueeze(-2), V_win.transpose(-2,-1)).squeeze(-2)

        # 7. Combine heads
        out = out.transpose(1,2).contiguous().view(batch_times_colums, num_rows_q, self.embedding_size)

        # 8. Final projection
        out = self.Wo(out)
        #print("out shape:", out.shape, flush = True)
        return out, attn_probs

class EinsteinLocalAttention(nn.Module):
    """
    Efficient sliding-window (local) attention without unfold.
    """
    def __init__(self, embed_dim, num_heads, window_size=20, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window = window_size

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k = None, v = None):
        """
        x: (B, N, embed_dim)
        returns: (B, N, embed_dim)
        """
        if k  == None:
            k = q 
        if v == None:
            v = k

        B, N, C = q.shape
        H = self.num_heads
        D = self.head_dim

        # 1. Linear projections
        Q = self.Wq(q).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K = self.Wk(k).view(B, N, H, D).transpose(1, 2)
        V = self.Wv(v).view(B, N, H, D).transpose(1, 2)

        out = torch.zeros_like(Q)  # (B, H, N, D)
        scale = D ** 0.5

        # 2. Efficient local attention via slicing
        for i in range(N):
            start = max(0, i - self.window // 2)
            end = min(N, i + self.window // 2 + 1)

            # Slice only the local window
            q_i = Q[:, :, i, :].unsqueeze(-2)       # (B, H, 1, D)
            k_window = K[:, :, start:end, :]        # (B, H, w, D)
            v_window = V[:, :, start:end, :]        # (B, H, w, D)

            # Compute attention scores: (B, H, 1, w)
            attn_scores = torch.einsum('bhid,bhjd->bhij', q_i, k_window) / scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.drop(attn_probs)

            # Weighted sum: (B, H, 1, D)
            out[:, :, i, :] = torch.einsum('bhij,bhjd->bhid', attn_probs, v_window).squeeze(-2)

        # 3. Combine heads
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.Wo(out)
        return out, attn_probs

class LocalSlidingWindowAttentionOptimized(nn.Module):
    ''' Ici, au lieu de decouper les windows a la main, on utilise un mask et la methode scaled dot product attention de pytorch
        Training plus rapide mais moins performant
    '''
    def __init__(self, embedding_size, num_heads, window_size=20, dropout=0.1):
        super().__init__()
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.window = window_size

        self.Wq = nn.Linear(embedding_size, embedding_size)
        self.Wk = nn.Linear(embedding_size, embedding_size)
        self.Wv = nn.Linear(embedding_size, embedding_size)
        self.Wo = nn.Linear(embedding_size, embedding_size)

        self.drop = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None):
        if k is None: k = q
        if v is None: v = k

        B, N, E = q.shape
        Q = self.Wq(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.Wk(k).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(v).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Create sliding window mask
        device = q.device
        idxs = torch.arange(N, device=device)
        # (N, N) mask where True means "mask this"
        mask = (idxs[None, :] - idxs[:, None]).abs() > self.window
        # Expand mask to (B*H, N, N) for scaled_dot_product_attention
        attn_mask = mask[None, :, :].expand(self.num_heads, -1, -1)

        # scaled_dot_product_attention expects (B*H, N, D)
        Q_flat = Q.reshape(B * self.num_heads, N, self.head_dim)
        K_flat = K.reshape(B * self.num_heads, N, self.head_dim)
        V_flat = V.reshape(B * self.num_heads, N, self.head_dim)

        # Compute attention with mask
        out = F.scaled_dot_product_attention(
            Q_flat, K_flat, V_flat,
            attn_mask=attn_mask.repeat(B, 1, 1),  # (B*H, N, N)
            dropout_p=self.drop.p,
            is_causal=False
        )

        # Reshape back and project
        out = out.view(B, self.num_heads, N, self.head_dim).transpose(1, 2).contiguous()
        out = out.view(B, N, E)
        out = self.Wo(out)
        #print("output shape = ", out.shape, flush = True)
        return out, idxs
    

class SparseAttention(nn.Module):
    """Implementation tiree d'un article """
    def __init__(self, d_model, n_heads, dropout=0.1, local_window=8):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.local_window = local_window
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k) ** -0.5
        
    def create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # Local attention window
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, start:end] = 1
        
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, q, k = None, v = None, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        if k == None:
            k = q
        if v == None:
            v = k

     
        Q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, q.device)
        scores = scores.masked_fill(sparse_mask == 0, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out), attn 
    



class BlockSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size=32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size

        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        H = self.n_heads
        D = self.head_dim
        BZ = self.block_size

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        # Split into blocks
        assert N % BZ == 0, "Sequence length must be divisible by block size"
        n_blocks = N // BZ

        # Output buffer
        out = torch.zeros_like(q)

        for b in range(n_blocks):
            # Current block range
            q_idx = slice(b * BZ, (b + 1) * BZ)

            # Determine which K/V blocks this block attends to
            attend_blocks = []

            # local block (itself + neighbors)
            attend_blocks.append(b)
            if b > 0:
                attend_blocks.append(b - 1)
            if b < n_blocks - 1:
                attend_blocks.append(b + 1)

            # global block 0 attends to everything
            if b == 0:
                attend_blocks = list(range(n_blocks))

            # Gather K/V for selected blocks
            k_idx = [slice(i * BZ, (i + 1) * BZ) for i in attend_blocks]
            k_sel = torch.cat([k[:, :, idx, :] for idx in k_idx], dim=2)  # (B,H,S,D)
            v_sel = torch.cat([v[:, :, idx, :] for idx in k_idx], dim=2)

            # Compute attention
            q_chunk = q[:, :, q_idx, :]                        # (B,H,BZ,D)
            att = torch.matmul(q_chunk, k_sel.transpose(-2, -1)) / math.sqrt(D)
            att = torch.softmax(att, dim=-1)                  # (B,H,BZ,S)
            out_chunk = torch.matmul(att, v_sel)              # (B,H,BZ,D)

            # Write back
            out[:, :, q_idx, :] = out_chunk

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out(out)