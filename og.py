import torch
import torch.nn as nn

device = "mps"
with open("data.txt", "r") as file:
    raw_data = file.readlines()

#Get PFMs from text file
data = []
for datum_start in range(1, len(raw_data), 5):
    nucleotides = []
    for i in range(4):
        words = raw_data[datum_start + i].split()
        frequencies = [int(word) for word in words[2:-1]] 
        nucleotides.append(frequencies)
    tens = torch.tensor(nucleotides, dtype=torch.float32).to(device) + 0.25
    #PFM to PPM
    tens = tens / tens.sum(dim=0, keepdim=True)
    data.append(tens)

max_len = max([datum.shape[1] for datum in data])
data = data[:5]

#example calculation of affinity
#this whole project will assume uniform DNA background frequencies
#first create PWM from PFM (log-odds)
pwms = [torch.log2(datum / 0.25) for datum in data]

def output_to_pfm_scorer(x):
    assert x.dim() == 1, "Input must be a 1D tensor"
    assert x.size(0) % 4 == 0, "Input length must be divisible by 4"
    n = x.size(0) / 4
    # Reshape to (n, 4)
    reshaped = x.view(n, 4)
    # Find argmax along each row (dim=1)
    argmax_indices = torch.argmax(reshaped, dim=1)
    # Create one-hot tensor (n, 4)
    one_hot = torch.zeros_like(reshaped)
    one_hot[torch.arange(n), argmax_indices] = 1
    # Transpose to (4, n)
    output = one_hot.t()
    return output

exit()

#Get affinity labels

#Sample sequences

#Train on the sequences
affin_net = nn.Sequential(
    nn.Linear(max_len*4, max_len*8),
    nn.ReLU(),
    nn.Linear(max_len*8, max_len*8),
    nn.ReLU(),
    nn.Linear(max_len*8, max_len*4)
    #then you need to apply softmax to every group of 4
).to(device)

