import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# 1. Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Generate dummy data for demonstration
def generate_dummy_data(num_samples=1000, input_size=10, num_classes=2):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

# 3. Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# 4. Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test data: {accuracy:.2f}%')
    return accuracy

# Main execution block
if __name__ == "__main__":
    # Hyperparameters
    input_size = 10
    hidden_size = 50
    num_classes = 2
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate data
    X_train, y_train = generate_dummy_data(num_samples=1000, input_size=input_size, num_classes=num_classes)
    X_test, y_test = generate_dummy_data(num_samples=200, input_size=input_size, num_classes=num_classes)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = SimpleNN(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("\nStarting model training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print("Model training complete!")

    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, test_loader)

    # Save the model
    model_save_path = "./simple_nn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Example of loading and using the model
    loaded_model = SimpleNN(input_size, hidden_size, num_classes).to(device)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()
    print("\nModel loaded and ready for inference.")
