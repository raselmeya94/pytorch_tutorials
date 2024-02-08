import torch

device="cuda" if torch.cuda.is_available() else "cpu"
# Initializing Tensor

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32 , device=device)

print(my_tensor)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.dtype)


# others common initialization methods

empty_tensor= torch.empty(size=(3,3))  
print(empty_tensor)                     # output is 3*3 dimention matrix all elements are random.
zeros_tensor= torch.zeros(size=(3,3))
print(zeros_tensor)                     # output is 3*3 dimention matrix all elements are zero.
ones_tensor= torch.ones(size=(3,3))
print(ones_tensor)                      # output is 3*3 dimention matrix all elements are ones.
eye_tensor= torch.eye(3,3)
print(eye_tensor)                       # output is 3*3 dimention matrix all diagonal elements are ones.



# uniform and normal distribution 

x=torch.empty((3,3)).normal_(mean=10, std=1)
print(x)
y=torch.empty(size=(3,3)).uniform_(0,1)
print(y)

# data types conversion into other types( int, float, double)
tensor= torch.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())


# array to tensor conversion and vice-versa
import numpy as np
np_array=np.zeros([2,3])
print(np_array)
print(np_array.shape)

tensor_np=torch.from_numpy(np_array)
print(tensor_np)


# addition 
x=torch.tensor([1,2,3,4])
y=torch.tensor([10,11,12,13])

add_xy=torch.empty(4)
torch.add( x, y, out=add_xy)
print(add_xy)
#  another way 2
add_xy= torch.add( x, y)
print(add_xy)
# 3
add_xy=x+y
print(add_xy)


# substraction
x=torch.tensor([1,2,3,4])
y=torch.tensor([10,11,12,13])

sub_xy=torch.empty(4)
torch.subtract( x, y, out=sub_xy)
print(sub_xy)
#  another way 2
sub_xy= torch.sub( x, y)
print(sub_xy)
# 3
sub_xy=x-y
print(sub_xy)

# division

div_xy= torch.true_divide(x,y)
print(div_xy)


# exponentiation
x=torch.tensor([[10,10], [10,10]])
exp_x= x.pow(2)
print(x)
print(exp_x)

# another way 
exp_x2= x**2
print(exp_x2)


# Matrix Multiplication

x1=torch.tensor([[10,10], [5,5]])
x2=torch.tensor([[2,2], [4,4]])
matrix_mul=torch.mm(x1,x2)
print(matrix_mul)
