
import torch

a = torch.randn(2, 4, 4)
print(a)
print()
print(a[..., 0])
print()
print(a[:, :, 0])

nx = 33
num_x = int(nx//7) + 1
num_y = int(nx//7) + 1

bot=(0,0)
top=(1,1)
x_bot, y_bot = bot
x_top, y_top = top

x_arr = torch.linspace(x_bot, x_top, steps=num_x)
y_arr = torch.linspace(y_bot, y_top, steps=num_y)
xx, yy = torch.meshgrid(x_arr, y_arr)
print(xx)
print(yy)
print(xx.shape)
mesh = torch.stack([xx, yy], dim=2)
print(mesh.shape)
#print(mesh)
print()
print(mesh[...,0])
print()
print(mesh[...,1])