import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "mps"  # Use "cuda" if available, or "cpu" for fallback

# Load raw data from file
with open("data.txt", "r") as file:
    raw_data = file.readlines()

# Parse PFMs from text file
data = []
for datum_start in range(1, len(raw_data), 5):
    nucleotides = []
    for i in range(4):
        words = raw_data[datum_start + i].split()
        frequencies = [int(word) for word in words[2:-1]]
        nucleotides.append(frequencies)
    tens = torch.tensor(nucleotides, dtype=torch.float32).to(device) + 0.25  # Add pseudocount
    tens = tens / tens.sum(dim=0, keepdim=True)  # Normalize to PPM
    data.append(tens)

data = data[:100]
# Determine max length and optionally limit number of PFMs for testing
max_len = max([datum.shape[1] for datum in data])
# data = data[:5]  # Uncomment to limit to first 5 PFMs

# Pad all PPMs to max_len with uniform (0.25) columns
padded_data = []
for tens in data:
    l = tens.shape[1]
    if l < max_len:
        pad_width = max_len - l
        tens = torch.nn.functional.pad(tens, (0, pad_width), mode='constant', value=0.25)
    padded_data.append(tens)
data = padded_data

# Create PWMs (log-odds scores) from PPMs assuming uniform background (0.25)
pwms = [torch.log2(ppm / 0.25) for ppm in data]

# Helper function to sample a one-hot sequence from a PPM (multinomial per position)
def sample_sequence_from_ppm(ppm):
    _, length = ppm.shape
    bases = []
    for pos in range(length):
        probs = ppm[:, pos]
        base = torch.multinomial(probs, num_samples=1).item()
        bases.append(base)
    one_hot = torch.zeros(4, length, device=device)
    one_hot[bases, torch.arange(length)] = 1
    return one_hot

# Helper function to sample a random uniform one-hot sequence
def sample_random_sequence(length):
    bases = torch.randint(0, 4, (length,), device=device)
    one_hot = torch.zeros(4, length, device=device)
    one_hot[bases, torch.arange(length)] = 1
    return one_hot

# Helper function to compute hard affinity (score) of a one-hot sequence against PWM
def compute_affinity(one_hot_seq, pwm):
    return torch.sum(one_hot_seq * pwm)

# Helper function to convert sequence indices to string
def indices_to_string(indices):
    base_map = ['A', 'C', 'G', 'T']
    return ''.join(base_map[i] for i in indices)

# Generate training examples: for each PPM, sample sequences from it to get high-affinity targets,
# pair with random input sequences
num_samples_per_ppm = 5  # Adjustable
training_examples = []
print("Making training samples")
for i in tqdm(range(len(data))):
    ppm = data[i]
    pwm = pwms[i]
    for _ in range(num_samples_per_ppm):
        sampled_seq = sample_sequence_from_ppm(ppm)
        desired_aff = compute_affinity(sampled_seq, pwm)
        input_seq = sample_random_sequence(max_len)
        training_examples.append((ppm, pwm, input_seq, desired_aff))

# Define the neural network: refines a sequence toward a desired affinity for a given PPM
input_size = max_len * 8 + 1  # Flattened pos-major seq (4*len) + pos-major PPM (4*len) + desired (1)
hidden_size = max_len * 8
output_size = max_len * 4  # Logits for refined sequence (pos-major: len positions x 4 bases)
affin_net = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
).to(device)

# Train the network
optimizer = torch.optim.Adam(affin_net.parameters(), lr=0.001)
num_epochs = 10  # Adjustable
print("About to train")
for epoch in range(num_epochs):
    total_loss = 0.0
    for ppm, pwm, input_seq, desired in training_examples:
        optimizer.zero_grad()
        # Flatten inputs in position-major order: transpose (4,len) -> (len,4) then flatten
        input_seq_flat = input_seq.t().contiguous().view(-1)
        ppm_flat = ppm.t().contiguous().view(-1)
        input_flat = torch.cat((input_seq_flat, ppm_flat, torch.tensor([desired], device=device)))
        # Forward pass
        output_logits = affin_net(input_flat)
        # Compute soft affinity for differentiable loss
        logits_reshaped = output_logits.view(max_len, 4)  # (positions, bases)
        soft_prob = torch.softmax(logits_reshaped, dim=1)  # Probabilities per position
        soft_ppm = soft_prob.t()  # (bases, positions) for compatibility with PWM
        aff_out = torch.sum(soft_ppm * pwm)
        # MSE loss on affinities
        loss = (aff_out - desired) ** 2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(training_examples)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# After training, generate and visualize examples for each PPM
for i in range(len(data)):
    print(f"\n--- Example for PPM {i+1} ---")
    example_ppm = data[i]
    example_pwm = pwms[i]
    # Use a desired affinity from training (e.g., first sample for this PPM)
    example_desired = training_examples[i * num_samples_per_ppm][3]
    example_input_seq = sample_random_sequence(max_len)
    # Compute input affinity (hard)
    aff_in = compute_affinity(example_input_seq, example_pwm)
    # Prepare input
    input_seq_flat = example_input_seq.t().contiguous().view(-1)
    ppm_flat = example_ppm.t().contiguous().view(-1)
    example_input_flat = torch.cat((input_seq_flat, ppm_flat, torch.tensor([example_desired], device=device)))
    # Generate output
    example_output = affin_net(example_input_flat)
    logits_reshaped = example_output.view(max_len, 4)
    # Get sequences as strings
    input_indices = torch.argmax(example_input_seq, dim=0).cpu().tolist()  # argmax over bases
    input_string = indices_to_string(input_indices)
    output_indices = torch.argmax(logits_reshaped, dim=1).cpu().tolist()  # argmax over bases
    generated_string = indices_to_string(output_indices)
    # Compute output affinities (soft for training consistency, and hard for comparison)
    soft_prob = torch.softmax(logits_reshaped, dim=1)
    soft_ppm = soft_prob.t()
    aff_out_soft = torch.sum(soft_ppm * example_pwm)
    output_one_hot = torch.zeros(4, max_len, device=device)
    output_one_hot[output_indices, torch.arange(max_len)] = 1
    aff_out_hard = compute_affinity(output_one_hot, example_pwm)
    # Print results
    print(f"Input sequence: {input_string}")
    print(f"Generated sequence: {generated_string}")
    print(f"Desired affinity: {example_desired:.4f}")
    print(f"Input affinity (hard): {aff_in:.4f}")
    print(f"Generated affinity (soft): {aff_out_soft:.4f}")
    print(f"Generated affinity (hard): {aff_out_hard:.4f}")
    # Visualize
    # Pause to view before next
    input("Press Enter to continue...")

