import torch

import torch_hawk

def test_main():

    dtype = torch.float32
    device = torch.device('cuda')

    if dtype == torch.float32:
        rtol=1e-6
        atol=1e-6
    else:
        rtol = 1e-3
        atol = 1e-3

    n = 20
    k = 2

    a = torch.ones((n, k), dtype=dtype, device=device)
    x = torch.ones((n, k), dtype=dtype, device=device)

    a = torch.rand((n, k), dtype=dtype, device=device)
    # x = torch.rand((n, k), dtype=dtype, device=device)

    d_o = torch.ones((n, k), dtype=dtype, device=device)

    # d_o = torch.rand((n, k), dtype=dtype, device=device)

    a_copy = a.clone().detach()
    x_copy = x.clone().detach()

    a.requires_grad = True
    x.requires_grad = True

    a_copy.requires_grad = True
    x_copy.requires_grad = True


    result = []
    current = torch.zeros(k, dtype=torch.float32, device=device)

    for i in range(n):
        current = a[i, :] * current + x[i, :]
        result.append(current.type(dtype))

    o2 = torch.stack(result, dim=0)

    #print(o2)

    o1 = torch_hawk.linear_recurrence(a_copy, x_copy)

    #print(o1)

    assert torch.allclose(o1, o2, rtol=rtol, atol=atol)

    total = (o1 * d_o).sum()
    total.backward()


    total = (o2 * d_o).sum()
    total.backward()

    assert x.grad is not None
    assert x_copy.grad is not None

    #print(x.grad)

    #print(x_copy.grad)

    assert torch.allclose(x_copy.grad, x.grad, rtol=rtol, atol=atol)


    assert a.grad is not None
    assert a_copy.grad is not None

    #print(a.grad)

    #print(a_copy.grad)

    assert torch.allclose(a_copy.grad, a.grad, rtol=rtol, atol=atol)
