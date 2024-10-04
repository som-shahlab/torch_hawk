import torch

import torch_hawk

def test_main():
    dtype = torch.float16
    device = torch.device('cuda')

    n = 200
    r = 5
    k = 2

    c = n // r

    # x = torch.ones((n, k), dtype=torch.float16).cuda()
    # w = torch.ones(k, 4, dtype=torch.float16).cuda()

    x = torch.rand((n, k), dtype=dtype).to(device)
    w = torch.rand((k, 4), dtype=dtype).to(device)

    s = torch.ones(n, dtype=torch.uint8).to(device)
    for i in range(c):
        s[r * i : r * (i + 1)] = i

    d_o = torch.rand((n, k), dtype=dtype).to(device)



    x_copy = x.clone().detach()
    w_copy = w.clone().detach()

    x.requires_grad = True
    w.requires_grad = True

    x_copy.requires_grad = True
    w_copy.requires_grad = True


    o1 = torch_hawk.conv1d(x_copy, w_copy, s)
    o2 = torch.nn.functional.conv1d(x.reshape((c, n // c, k)).transpose(1, 2), w.unsqueeze(1), padding=3, groups=k)[:, :, :-3].transpose(2, 1).reshape(n , k)

    if dtype == torch.float32:
        rtol=1e-6
        atol=1e-6
    else:
        rtol = 1e-3
        atol = 1e-3

    assert torch.allclose(o1, o2, rtol=rtol, atol=atol)

    total = (o1 * d_o).sum()
    total.backward()


    total = (o2 * d_o).sum()
    total.backward()

    assert x.grad is not None
    assert x_copy.grad is not None

    assert torch.allclose(x_copy.grad, x.grad, rtol=rtol, atol=atol)

    assert w.grad is not None
    assert w_copy.grad is not None

    print(w_copy.grad)
    print(w.grad)

    assert torch.allclose(w_copy.grad, w.grad, rtol=rtol, atol=atol)
