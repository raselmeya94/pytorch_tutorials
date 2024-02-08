import torch

device="cuda" if torch.cuda.is_available() else "cpu"

# # create a 4 * 4 matrix with all elements is 10
# x=torch.empty(4,4).fill_(10)
# # print(x)
# sum_x=torch.sum(x, dim=0) # column wise
# # print(sum_x)

# values, indices= torch.max(x, dim=0)
# # print(values, indices)

# x=torch.randn(4,4) # negative positive combine matrix
# print(x)
# abs_x=torch.abs(x) # all convert into positive.
# print(abs_x)

# argmax


