student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'

from Helper_codes.validator import *

python_code = extract_python("./Q3.ipynb")
with open(f'python_code_Q3_{student_number}.py', 'w') as file:
    file.write(python_code)

from Helper_codes.ae_helper import get_data
from sklearn.model_selection import train_test_split

X, Y, y = get_data()

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(X, Y, y, test_size=0.2, random_state=17)
X_train, X_val, Y_train, Y_val, y_train, y_val = train_test_split(X_train, Y_train, y_train, test_size=0.1, random_state=17)

X_train.shape

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.ReLU(True)
        )

    def forward(self, x):
        _x = x
        _x = self.encoder(_x)
        _x = self.decoder(_x)
        return _x

from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MnistNextDigitDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        return self.X[i], self.Y[i], self.y[i]

train_dataloader = DataLoader(
    MnistNextDigitDataset(X_train, Y_train, y_train),
    batch_size=512,
    shuffle=True
)
val_dataloader = DataLoader(
    MnistNextDigitDataset(X_val, Y_val, y_val),
    batch_size=512,
    shuffle=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
criterion = nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 30

min_val_loss = float('inf')
min_state_dict = model.state_dict()

for epoch in range(num_epochs):
    train_loss, val_loss = 0, 0

    model.train()

    for i, (images, next_images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, 28*28).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        batch_loss = criterion(outputs, next_images.reshape(-1, 28*28).to(device))
        train_loss += batch_loss.item() * images.size(0)
        batch_loss.backward()
        optimizer.step()

    model.eval()

    for i, (images, next_images, labels) in enumerate(val_dataloader):
        images = images.reshape(-1, 28*28).to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, next_images.reshape(-1, 28*28).to(device))
        val_loss += batch_loss.item() * images.size(0)

    train_loss /= len(train_dataloader.dataset)
    val_loss /= len(val_dataloader.dataset)
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        min_state_dict = model.state_dict()

    print(f"[Epoch {epoch}]\t"
          f"Train Loss: {train_loss:.4f}\t"
          f"Validation Loss: {val_loss:.4f}")

model.load_state_dict(min_state_dict)

import matplotlib.pyplot as plt

examples_number = 40

plot_width = 4
plot_height = 10
size_scale = 2

fig, ax = plt.subplots(plot_height, plot_width*2, figsize=(plot_width*2*size_scale, plot_height*size_scale))

for i, idx in zip(range(0, 2*examples_number-1, 2), torch.randint(0, len(X_train), [examples_number])):
    i1, i2 = i//(plot_width*2), i%(plot_width*2)
    ax[i1, i2].imshow(X_train[idx].reshape(28, 28), cmap='gray')
    ax[i1, i2].set_title('input')
    ax[i1, i2].set_xticks([])
    ax[i1, i2].set_yticks([])
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(X_train[idx].reshape(1, -1)).to(device))
    ax[i1, i2+1].imshow(output.reshape(28, 28), cmap='gray')
    ax[i1, i2+1].set_title('output')
    ax[i1, i2+1].set_xticks([])
    ax[i1, i2+1].set_yticks([])

plt.show()

