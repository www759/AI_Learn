import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)

y0 = torch.mul(a, b)
y1 = torch.add(a, b)

loss = torch.cat([y0, y1], dim=0)
grad_tensors = torch.tensor([1., 2.])

loss.backward(gradient=grad_tensors)

print(w.grad)


x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)

grad_1 = torch.autograd.grad(y, x, create_graph=True)
print(grad_1)

grad_2 = torch.autograd.grad(grad_1[0], x)
print(grad_2)

