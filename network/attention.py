import torch
import torch.nn as nn

def attention(query, key, value, key_mask=None, temperature=1.0):
    """
    @param query:        b,d,h,n
    @param key:          b,d,h,m
    @param value:        b,d,h,m
    @param key_mask:     b,1,1,m
    @param temperature:
    @return:
    """
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query / temperature, key) / dim ** .5 # b,head,n0,n1
    if key_mask is not None: scores = scores.masked_fill(key_mask == 0, -1e7)
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class SpecialLayerNorm(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.norm=nn.LayerNorm(in_dim)

    def forward(self,x):
        x = self.norm(x.permute(0,2,1))
        return x.permute(0,2,1)

class AttentionBlock(nn.Module):
    def __init__(self,in_dim,att_dim,out_dim,head_num=4,temperature=1.0,bias=True,skip_connect=True,norm='layer'):
        super().__init__()
        self.conv_key=nn.Conv1d(in_dim,att_dim,1,bias=bias)
        self.conv_query=nn.Conv1d(in_dim,att_dim,1,bias=bias)
        self.conv_feats=nn.Conv1d(in_dim,out_dim,1,bias=bias)
        self.conv_merge=nn.Conv1d(out_dim,out_dim,1,bias=bias)

        self.head_att_dim=att_dim//head_num
        self.head_out_dim=out_dim//head_num
        self.head_num=head_num
        self.temperature=temperature
        if norm=='layer':
            self.norm=SpecialLayerNorm(out_dim)
        elif norm=='instance':
            self.norm = nn.InstanceNorm1d(out_dim)
        else:
            raise NotImplementedError
        self.skip_connect = skip_connect
        if skip_connect:
            assert(in_dim==out_dim)

    def forward(self, feats_query, feats_key, key_mask=None):
        '''
        :param feats_query: b,f,n0
        :param feats_key: b,f,n1
        :param key_mask: b,1,n1
        :return: b,f,n0
        '''
        b,f,n0=feats_query.shape
        b,f,n1=feats_key.shape

        query=self.conv_query(feats_query).reshape(b, self.head_att_dim, self.head_num, n0)  # b,had,hn,n0
        key=self.conv_key(feats_key).reshape(b, self.head_att_dim, self.head_num, n1)        # b,had,hn,n1
        feats=self.conv_feats(feats_key).reshape(b, self.head_out_dim, self.head_num, n1)    # b,hod,hn,n1
        if key_mask is not None: key_mask = key_mask.reshape(b, 1, 1, n1) # b,1,1,n1
        feats_out, weights = attention(query, key, feats, key_mask, self.temperature)
        feats_out = feats_out.reshape(b,self.head_out_dim*self.head_num,n0) # b,hod*hn,n0
        feats_out = self.conv_merge(feats_out)
        if self.skip_connect: feats_out=feats_out+feats_query
        feats_out = self.norm(feats_out)
        return feats_out