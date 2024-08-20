import pickle
import re
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.linear_model import LinearRegression
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Set up the argument parser
parser = argparse.ArgumentParser(description='Untrained GPT')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model and training parameters
batch_size = 12
block_size = 32


max_iters = 100000
print_progress_iters = 500

learning_rate = 3e-4
eval_iters = 1

n_embd = 300
n_head = 4
n_layer = 4
dropout = 0.2

print(f"Model Training on Device: {device}")

# Load the dataset
dataset = load_dataset("yahma/alpaca-cleaned", trust_remote_code=True)

# Access the train split
train_data = dataset['train']

# Function to save metrics to JSON
def save_metrics_to_json(metrics, filename='report.json'):
    try:
        with open(filename, 'r') as f:
            report = json.load(f)  # Load existing data
            
            # Check if the loaded data is a dictionary (from a previous run)
            if isinstance(report, dict):
                report = [report]  # Convert it to a list of one dictionary
    except FileNotFoundError:
        report = []  # Initialize an empty list if file doesn't exist

    report.append(metrics)  # Append new metrics

    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)  # Write combined data back to the file


# Create a random validation split from the train data
def create_validation_split(train_data, split_ratio=0.1):
    total_size = len(train_data)
    val_size = int(total_size * split_ratio)
    indices = list(range(total_size))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    def split_dataset(indices):
        return [{'input': train_data[i]['input'], 'output': train_data[i]['output']} for i in indices]

    val_data = split_dataset(val_indices)
    train_data = split_dataset(train_indices)
    return train_data, val_data

train_data, val_data = create_validation_split(train_data)

# Print some example entries
print("Example from train split:")
print(train_data[0]['output'])

print("\nExample from validation split:")
print(val_data[0]['output'])

# Character mapping setup
chars = ""

with open("vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s if c in string_to_int]
decode = lambda l: ''.join([int_to_string[i] for i in l])

def clean_data(data):
    cleaned_content = re.sub(r'\{.*?\}', '', data)  # Removes content within {}
    cleaned_content = re.sub(r'\[.*?\]', '', cleaned_content)  # Removes content within []
    cleaned_content = re.sub(r'\<.*?\>', '', cleaned_content)  # Removes content within <>

    # Keep only letters, punctuation marks, and spaces
    cleaned_content = re.sub(r'[^\w\s.,!?\'\";:()-]', '', cleaned_content)  # Remove symbols except punctuation

    return cleaned_content

# Batch processing
def get_batch(split):
    if split == 'train':
        data = train_data
    elif split == 'val':
        data = val_data
    else:
        raise ValueError("Invalid split name provided.")

    indices = torch.randint(len(data), (batch_size,))
    x = []
    y = []

    for i in indices:
        # Convert text to token IDs and truncate/pad to block_size
        text = data[i]['output']
        token_ids = encode(text)
        
        if len(token_ids) < block_size:
            # Pad if necessary
            token_ids.extend([0] * (block_size - len(token_ids)))
        else:
            # Truncate if necessary
            token_ids = token_ids[:block_size]
        
        x.append(token_ids)

        # Prepare target sequence
        target_ids = token_ids[1:] + [0]  # Shift right for the target sequence
        y.append(target_ids)

    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    x, y = x.to(device), y.to(device)

    return x, y

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop index to the last block_size tokens
            index_cond = index[:, -block_size:]
            # Get predictions
            logits = self(index_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append the sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # Calculate loss if targets are provided
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits

model = GPTLanguageModel(vocab_size)

try:
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
        print('loaded successfully!')
except:
    print('Error: Unable to load previous model parameters')

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# To store loss values
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []

def estimate_loss():
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = []
            all_targets = []
            all_preds = []
            for _ in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses.append(loss.item())
                
                # Calculate accuracy and F1-score
                preds = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
                targets = Y.cpu().numpy().flatten()
                
                all_preds.extend(preds)
                all_targets.extend(targets)
            
            accuracy = accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average='weighted')
            
            out[split] = {
                'loss': sum(losses) / len(losses),
                'accuracy': accuracy,
                'f1_score': f1,
            }
    model.train()
    return out

metrics = estimate_loss()

# Training loop
for iter in range(max_iters):
   
    if iter % print_progress_iters ==0:
        print(f"Step {iter}: Train loss {metrics['train']['loss']:.4f}, Val loss {metrics['val']['loss']:.4f}")
        print(f"Train Accuracy: {metrics['train']['accuracy']:.4f}, F1-score: {metrics['train']['f1_score']:.4f}")
        print(f"Val Accuracy: {metrics['val']['accuracy']:.4f}, F1-score: {metrics['val']['f1_score']:.4f}")

        with open('model-01.pkl', 'wb') as f:
            pickle.dump(model, f)
            print("Model Saved Successfully")

    if iter % eval_iters == 0:
        metrics = estimate_loss()
        save_metrics_to_json(metrics, 'report.json')

        
        
       

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



prompt = ""
while prompt != "exit":
    prompt = input("Prompt (type 'exit' to terminate program): ")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=250)[0].tolist())
    print(generated_chars)