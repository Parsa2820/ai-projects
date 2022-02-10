student_number = 98102284
Name = 'Parsa'
Last_Name = 'Mohammadian'


from Helper_codes.validator import *

python_code = extract_python("./Q2.ipynb")
with open(f'python_code_Q2_{student_number}.py', 'w') as file:
    file.write(python_code)


import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from Helper_codes.ae_helper import init_mnist_subset_directories
from functools import reduce


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


mnist_dataset = datasets.MNIST(
    root='data', train=True, download=True, transform=transforms.ToTensor())


p1 = torch.tensor([3.], requires_grad=True)
p2 = torch.tensor([7.], requires_grad=True)


L = 3 * p1**3 - 7 * p2**2 + torch.sin(p1) * p2**2




L.backward()


print(f"P_1 grad: {p1.grad.item()}\nP_2 grad: {p2.grad.item()}")


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        p = torch.rand(1).item()
        if p < self.p:
            flipped_x = torch.flip(x, [2])
            return flipped_x
        return x


class RandomColorSwap(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        p = torch.rand(1).item()
        if p < self.p:
            m = torch.max(x)
            color_swapped_x = torch.sub(m, x)
            return color_swapped_x
        return x


trans = transforms.Compose([
    RandomHorizontalFlip(p=0.7),
    RandomColorSwap()
])


num_imgs = 10
fig, axs = plt.subplots(2, num_imgs, figsize=(25, 5))
for i, idx in enumerate(torch.randint(0, len(mnist_dataset), [num_imgs])):
    x, y = mnist_dataset[idx]
    axs[0, i].imshow(x[0], cmap='gray')
    axs[1, i].imshow(trans(x)[0], cmap='gray')
    for k in range(2):
        axs[k, i].set_yticks([])
        axs[k, i].set_xticks([])

axs[0, 0].set_ylabel("Original")
axs[1, 0].set_ylabel("Transformed")


dataset_path = "new_mnist"
init_mnist_subset_directories(mnist_dataset, dataset_path)


class MNISTDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.folders = os.listdir(root_dir)
        self.path_label_tuple_by_idx = {}
        for folder in self.folders:
            path = os.path.join(root_dir, folder)
            label = int(folder)
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                idx = int(file.replace('data_', '').replace('.pth', ''))
                self.path_label_tuple_by_idx[idx] = (file_path, label)
        self.len = len(self.path_label_tuple_by_idx)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path_label = self.path_label_tuple_by_idx.get(idx)
        if path_label is None:
            raise IndexError(f'Index {idx} not found in dataset')
        path, label = path_label
        image = self.transform(torch.load(path))
        return image, label


my_dataset = MNISTDataset(root_dir=dataset_path, transform=RandomColorSwap())
len(my_dataset)


num_imgs = 10
fig, axs = plt.subplots(1, num_imgs, figsize=(50, 5))
for i, idx in enumerate(torch.randint(0, len(my_dataset), [num_imgs])):
    image, label = my_dataset[i]
    axs[i].imshow(image[0], cmap='gray')
    axs[i].set_xlabel(f'Label: {label}')
    axs[i].set_yticks([])
    axs[i].set_xticks([])
plt.show()


class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        _x = x
        _x = self.linear1(_x)
        _x = self.relu(_x)
        _x = self.linear2(_x)
        return _x


model = DigitRecognizer().to(device)
model


transform_compose = transforms.Compose([
    transforms.ToTensor(),
])

mnist_dataset = datasets.MNIST(
    root='dataset', train=True, download=True, transform=transform_compose)
dataset_size = len(mnist_dataset)
train_dataset, val_dataset = torch.utils.data.random_split(
    mnist_dataset, [int(dataset_size * 0.8), int(dataset_size * 0.2)])
test_dataset = datasets.MNIST(
    root='dataset', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 10

train_loss_arr, val_loss_arr = [], []
for epoch in range(num_epochs):
    train_loss, val_loss = 0, 0

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        train_loss += batch_loss.item() * images.size(0)
        batch_loss.backward()
        optimizer.step()

    model.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        val_loss += batch_loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_loss_arr.append(train_loss)
    val_loss_arr.append(val_loss)

    print(f"[Epoch {epoch}]\t"
          f"Train Loss: {train_loss:.4f}\t"
          f"Validation Loss: {val_loss:.4f}")


x = torch.arange(num_epochs)
plt.figure(figsize=(10, 5))
plt.plot(x, train_loss_arr, label='Train Loss')
plt.plot(x, val_loss_arr, label='Validation Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
pass


true_predictions = 0
total_predictions = 0
test_loss = 0
wrong_predictions = []
for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        true_predictions += (predicted == labels).sum().item()
        batch_loss = criterion(outputs, labels)
        test_loss += batch_loss.item() * images.size(0)
        for idx in torch.where(predicted != labels)[0].tolist():
            wrong_predictions.append(
                (images[idx], labels[idx], predicted[idx]))
test_loss /= len(test_loader)
print(f'Accuracy: {int(true_predictions/total_predictions*100)}')
print(f'Test Loss: {test_loss:.4f}')


num_imgs = 8
fig, axs = plt.subplots(1, num_imgs, figsize=(50, 5))
for i, idx in enumerate(torch.randint(0, len(wrong_predictions), [num_imgs])):
    image, label, predicted_label = wrong_predictions[i]
    image = image.reshape(28, 28).cpu()
    axs[i].imshow(image, cmap='gray')
    axs[i].set_xlabel(
        f'Label: {label}, Predicted: {predicted_label}', fontsize=20)
    axs[i].set_yticks([])
    axs[i].set_xticks([])
plt.show()


