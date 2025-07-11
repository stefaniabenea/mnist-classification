import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleMLP
from utils import evaluate, get_wrong_predictions, get_sample_batch
from visualize import plot_sample_images, plot_wrong_predictions

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_dataset = datasets.MNIST(root='./data', train = True, download = True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train = False, download = True, transform=transform)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle=False)

print(f"Size of training set: {len(train_dataset)}, Size of test dataset {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss/len(train_dataloader)
    accuracy = evaluate(model, test_dataloader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}")

print(f"Final test accuracy: {accuracy:.2f}%")

wrong_images, wrong_labels, wrong_predictions =get_wrong_predictions(model, test_dataloader, device)
plot_wrong_predictions(wrong_images,wrong_labels, wrong_predictions)
sample_images, sample_labels, sample_predictions = get_sample_batch(test_dataloader, model, device)
plot_sample_images(sample_images, sample_labels, sample_predictions)

torch.save(model.state_dict(),'mnist_model.pth')



