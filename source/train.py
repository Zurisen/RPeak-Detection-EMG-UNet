import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model import UNet_1D
import os
from datasets import Dataset_ECG
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import wandb  ## optional

n_epochs = 30
batch_size = 32
learning_rate = 1e-3
wandb.init(
    # set the wandb project
    project="UNET-1D",

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "UNet",
    "dataset": "ECG",
    "epochs": n_epochs,
    }
)

if not os.path.exists("../checkpoints/"):
    print("Creating checkpoints directory...")
    os.makedirs("../checkpoints/")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet_1D().to(device)
model.initialize_weights_uniform()
print(f"Imported model into {device} and initialized weights")

loss_fun = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"Initialized loss function and optimizer")

dataset = Dataset_ECG("../data/", device=device)
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler)
print(f"Imported train and test datasets")
print(f"Train Dataset: {len(train_sampler)} samples || {len(train_loader)} batches")
print(f"Test Dataset: {len(test_sampler)} samples || {len(test_loader)} batches")

print("-------------- Starting Training Loop --------------")
for epoch in range(n_epochs):
    print(f"Epoch: {epoch}")
    for data, target in tqdm(train_loader, leave=False):
        ## Train step
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        
    print(f"Loss: {loss.item()}")
    wandb.log({"loss": loss.item()})
    torch.save(model.state_dict(), f"../checkpoints/epoch_{epoch}.pt")
    ## Evaluation step
    # model.eval()
    
    

