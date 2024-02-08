import torch

device="cuda" if torch.cuda.is_available() else "cpu"

# create a 4 * 4 matrix with all elements is 10
x=torch.empty(4,4).fill_(10)
# print(x)
sum_x=torch.sum(x, dim=0) # column wise
# print(sum_x)

values, indices= torch.max(x, dim=0)
# print(values, indices)

x=torch.randn(4,4) # negative positive combine matrix
print(x)
abs_x=torch.abs(x) # all convert into positive.
print(abs_x)

# Tensor Shape, Re-Shaping

x=torch.arange(10)
print(x)   # one dimension

#output : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# if i change this one-dim matrix into any shape then must be careful about total elements are equal which dim you convert it 

# Here is 10 elements 
# that means 
#  1 * 10 || 10 *1   
#  2 * 5 || 5 * 2
# possible dimension
x_2x5= x.reshape(2,5)
print(x_2x5)

# output of 2 * 5:

# [[0, 1, 2, 3, 4],
# [5, 6, 7, 8, 9]]

# output of 5 * 2:
# [[0, 1],
# [2, 3],
# [4, 5],
# [6, 7],
# [8, 9]]

x_5x2= x.reshape(5,2)
print(x_5x2)

