import torch
import numpy as np

# softmax(Q * K^t) * V
if __name__ == "__main__":
    m = 2048
    k2 = 256
    n = 64
    b0 = 1
    b1 = 1
    mask_type = torch.bool
    write_mask_type = torch.bool
    scale = 10000
    q = (torch.rand(b0,b1,m,n).to(torch.float32) - 0.5) * scale
    k = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * scale
    v = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * scale
    mask = (torch.rand(b0,b1,m,k2) > 0.5)
    #mask = (torch.ones(b0,b1,m,k2))
    #mask = torch.tril(mask, diagonal = 1)
    ref_mask = (mask.to(torch.float32))

    np.save("npys/attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("npys/attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("npys/attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("npys/attn_mask.npy", mask.detach().to(dtype=write_mask_type, device="cpu").numpy())
    ## ^ Comment out to exclude mask

    # Post attention func
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, ref_mask) # Or attn_mask=None
    #out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None) # Or attn_mask=None
    np.save("npys/attn_ref.npy", out.detach().to(device="cpu").numpy())
