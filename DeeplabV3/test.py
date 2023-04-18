# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)

def sum(y):
    x = 0
    for _ in range(3):
        x += 1
    y += x
    return y


if __name__ == "__main__":
    sum(2)