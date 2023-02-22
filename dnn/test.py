import torch as t 
import numpy as np
'''''
bs, frame_len = 2,2
decoder_len = frame_len
pos=t.Tensor([[0,1],[3,0]])
m_mask = pos.ne(0).type(t.float)
mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
print(m_mask.shape)
print(mask.shape)
print(m_mask)
print(mask)
'''''
data=t.from_numpy(np.random.randint(1,100,size=(5,10)))
print(data.shape)
