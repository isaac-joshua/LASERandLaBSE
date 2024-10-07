# import torch

# print("Number of GPU: ", torch.cuda.device_count())
# print("GPU Name: ", torch.cuda.get_device_name())


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)


import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("Number of GPUs: ", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} Name: ", torch.cuda.get_device_name(i))
else:
    print("CUDA is not available. No GPU found.")

    