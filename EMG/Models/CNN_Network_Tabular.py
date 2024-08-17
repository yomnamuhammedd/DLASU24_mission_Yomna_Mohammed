import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

x_train_tabular = np.load('/Users/yomnaamuhammedd/PycharmProjects/ASU-Racing Team Task Model1/X_train_tabular.npy')
y_train_tabular =np.load('/Users/yomnaamuhammedd/PycharmProjects/ASU-Racing Team Task Model1/y_train_tabular.npy')


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if device == 'mps':
    torch.manual_seed(42)

print(f'Using Device {device}')

print(x_train_tabular.shape)
print(y_train_tabular.shape)

def createLoader(x,y,batch_size):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 10

x_train, x_val, y_train, y_val = train_test_split(x_train_tabular, y_train_tabular,
                                                  test_size=0.05, random_state=42)
                  
train_loader = createLoader(x_train, y_train,BATCH_SIZE)
val_loader = createLoader(x_val, y_val,BATCH_SIZE)


class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(24)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=24, out_channels=64, kernel_size=3, padding='same')
        self.dropout1 = nn.Dropout(0.2)

        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 14)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.pool2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def train_model(self, dataloader, loss_fn, optimizer):
        self.train()
        num_batches = len(dataloader)
        training_error = 0.0

        for batch in dataloader:
            inputs, labels = batch

            optimizer.zero_grad()  # Clear previous gradients

            outputs = self(inputs)  # Forward pass

            loss = loss_fn(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            training_error += loss.item()

        training_error /= num_batches  # Average error
        return training_error

    def validate_model(self, dataloader, loss_fn):
        self.eval()  
        num_batches = len(dataloader)
        validation_error = 0.0

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch

                outputs = self(inputs)  # Forward pass

                loss = loss_fn(outputs, labels)

                validation_error += loss.item()

        validation_error /= num_batches  # Average error
        return validation_error

    def plot_losses(self,train_losses, val_losses):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
        plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

model = CNNNetwork()

loss_fn = nn.MSELoss()
optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    train_loss = model.train_model(train_loader, loss_fn, optimizer)
    val_loss = model.validate_model(val_loader, loss_fn)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")

epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('LinearNetworkPlot-Tabular.png')
plt.show()
