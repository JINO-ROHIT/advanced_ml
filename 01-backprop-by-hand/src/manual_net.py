import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


torch.manual_seed(2024)

# Hyperparameters
n_embd = 10
n_hidden = 64
n_classes = 10
batch_size = 64
learning_rate = 0.01
num_epochs = 3

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize parameters
g = torch.Generator().manual_seed(2024)

# Layer 1
W1 = torch.randn((28 * 28, n_hidden), generator=g) * 0.1
b1 = torch.randn(n_hidden, generator=g) * 0.1

# Layer 2
W2 = torch.randn((n_hidden, n_classes), generator=g) * 0.1
b2 = torch.randn(n_classes, generator=g) * 0.1

parameters = [W1, W2, b1, b2]

for p in parameters:
    p.requires_grad = True

print(f"Total Parameters: {sum(p.nelement() for p in parameters)}")

# Training loop with manual backpropagation
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        data = data.view(data.size(0), -1)  # Flatten the input
        h = data @ W1 + b1
        h_relu = torch.relu(h)
        logits = h_relu @ W2 + b2
        loss = F.cross_entropy(logits, target)
        
        total_loss += loss.item()

        # Backward pass (manual)
        # Gradients for output layer
        dlogits = F.softmax(logits, dim=1) # we need softmax here since we are using cross_entropy instead of nn.crossentropy that has softmax inbuilt
        # using target instead of batch size here since batch size can be uneven
        dlogits[range(len(target)), target] -= 1
        dlogits /= len(target) 

        # Gradients for W2 and b2
        dW2 = h_relu.t() @ dlogits
        db2 = dlogits.sum(0)

        # Gradients for h_relu
        dh_relu = dlogits @ W2.t()

        # Gradients for h (applying ReLU derivative)
        dh = dh_relu * (h > 0).float()

        # Gradients for W1 and b1
        dW1 = data.t() @ dh
        db1 = dh.sum(0)

        # Update parameters, cannot do an inplace directly
        # W1 -= learning_rate * dW1
        # b1 -= learning_rate * db1
        # W2 -= learning_rate * dW2
        # b2 -= learning_rate * db2
        grads = [dW1, dW2, db1, db2]
        for p, grad in zip(parameters, grads):
            p.data += -learning_rate * grad

        if batch_idx % 100 == 0:
            print(f'Epoch 1, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    print(f'Epoch 1, Average Loss: {total_loss / len(train_loader):.4f}')

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for data, target in train_loader:
        data = data.view(data.size(0), -1)
        h = torch.relu(data @ W1 + b1)
        logits = h @ W2 + b2
        _, predicted = torch.max(logits, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy on training set: {100 * correct / total:.2f}%')