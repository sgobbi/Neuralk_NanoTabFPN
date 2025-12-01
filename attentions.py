import torch
from torch import nn


class LocalAttention(nn.Module):
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
    