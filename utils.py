import torch


def sign(x):
    ans = x.clone()
    ans[x < 0.] = -1.
    ans[x >= 0.] = 1.
    return ans


class Sign(torch.autograd.Function):
    def __init__(self, *args, **kwargs):
        super(Sign, self).__init__(*args, **kwargs)

    def forward(self, x):
        return sign(x)

    def backward(self, grad_output):
        return grad_output.clone()


def smoothBinerazer(x, is_embedding=False):
    if not is_embedding:
        ans = x.abs().mean() * Sign()(x)
    else:
        if len(x.size()) == 2:
            ans = torch.mean(x.abs(), dim=1).resize(x.size()[0],1).repeat(1, x.size()[1]) * Sign()(x)
        else:
            ans = torch.mean(x.abs(), dim=2).resize(x.size()[0],x.size()[1],1).repeat(1, 1, x.size()[2]) * Sign()(x)
    return ans
