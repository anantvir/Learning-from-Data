import torch
import numpy as np

device_to_run = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32, device = device_to_run, requires_grad=True)

# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)

x = torch.empty(size=(3,3))
x = torch.rand(size=(3,3))


x = torch.eye(5,5)
x = torch.ones(5,5)
x = torch.linspace(start=0.1, end=1,steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=4)
#print(x)

tensor = torch.arange(4)
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
print(tensor)