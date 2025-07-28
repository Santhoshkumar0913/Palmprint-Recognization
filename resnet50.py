import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# Define hyperparameters
batch_size = 16
num_epochs = 50  # Reduced for faster convergence
learning_rate = 0.0001

# Define dataset paths
dataset = 'Palmprint'
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')  

# Data transformations
image_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load dataset
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=0)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=0)

print(f"Training samples: {train_data_size}, Validation samples: {valid_data_size}")

# Load ResNet-50 model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Unfreeze all layers for fine-tuning
for param in resnet50.parameters():
    param.requires_grad = True

# Modify the fully connected layer for 99-class classification
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Increased dropout for regularization
    nn.Linear(512, 99),
    nn.LogSoftmax(dim=1)
)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)

# Define loss function and optimizer
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training and validation function
def train_and_valid(model, loss_function, optimizer, scheduler, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"Epoch: {epoch+1}/{epochs}")

        model.train()
        train_loss, train_acc = 0.0, 0.0
        valid_loss, valid_acc = 0.0, 0.0

        for inputs, labels in train_data:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        # Validation phase
        with torch.no_grad():
            model.eval()
            for inputs, labels in valid_data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(outputs, dim=1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        # Save the best model
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

            # Ensure the directory exists before saving
            model_path = 'models_resnet50'
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{model_path}/Palmprint_best_model.pt')
            print(f"✅ Model saved at epoch {best_epoch} with accuracy {best_acc:.4f}")

        epoch_end = time.time()

        print(f"Epoch: {epoch+1}, Training Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.2f}%, "
              f"Validation Loss: {avg_valid_loss:.4f}, Accuracy: {avg_valid_acc*100:.2f}%, "
              f"Time: {epoch_end-epoch_start:.2f}s")

        # Step the scheduler
        scheduler.step()

    return model, history, best_acc, best_epoch

# Run training inside `if __name__ == "__main__"` for Windows compatibility
if __name__ == "__main__":
    trained_model, history, best_acc, best_epoch = train_and_valid(resnet50, loss_func, optimizer, scheduler, num_epochs)

    # Save training history
    model_path = 'models_resnet50'
    os.makedirs(model_path, exist_ok=True)
    torch.save(history, f'{model_path}/Palmprint_history.pt')

    # Convert history to numpy array
    history = np.array(history)

    # Plot loss curves
    plt.plot(history[:, 0:2])
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 8)
    plt.title(f'Best Accuracy: {best_acc:.4f}, Best Epoch: {best_epoch}')
    plt.savefig(f'{model_path}/loss_curve.png')
    plt.close()

    # Plot accuracy curves
    plt.plot(history[:, 2:4])
    plt.legend(['Train Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title(f'Best Accuracy: {best_acc:.4f}, Best Epoch: {best_epoch}')
    plt.savefig(f'{model_path}/accuracy_curve.png')
    plt.close()
