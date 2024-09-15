import torch

x= torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)
torch.add(x,y,out=z1)

z = x+y
z= x/y
#x.add_(y)
z = x.pow(3)
z = x**3

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)


z = x*y

z = torch.dot(x,y)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_batch_mm = torch.bmm(tensor1, tensor2) # Shape -> (batch, n ,p)
#print("Shape of batch matrix multiplication :", out_batch_mm.shape)
#print(out_batch_mm)

# Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

# even though matrix and vector are not the same shape, but still we can broadcast the vector to match the shape of matrix
z = x1 - x2
z = x1**x2

# Other usefule tensor operations

values, indices = torch.max(x, dim=0)
#print(values)
#print("Indices : ",indices)

mean_x = torch.mean(x.float(), dim=0)
#print(mean_x)

# Tensor Indexing
batch_size = 32
features = 25

x = torch.rand((batch_size, features))
#print(x[0].shape)
#print(x[:,0])
#print(x[2,0:10])
x = torch.arange(9)

x_3X3 = x.view(3,3)
x_3X3 = x.reshape(3,3)
print(x_3X3.shape)