from __future__ import print_function
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow
import wandb
from torchvision import datasets, transforms
from azureml.core import Run
from sklearn.metrics import confusion_matrix

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_data(args, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), 
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                batch_idx / len(train_loader), loss.item()))

    # Log training metrics        
    mlflow.log_metrics({
        "Epoch": epoch,
        "Train Loss": loss.item()})
    wandb.log({
        "Epoch": epoch,
        "Train Loss": loss.item()})

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0%})\n'.format(
        test_loss, correct, len(test_loader.dataset),
        correct / len(test_loader.dataset)))

    # Log testing metrics
    mlflow.log_metrics({
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})
    wandb.log({
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})

def evaluate():
    
    # Facing issues getting model outputs into confusion matrix. Mocking some evaluation metrics for now.

    # Log confusion matrix
    y_true = [2, 0, 2, 2, 0, 1, 3, 4, 5, 6, 8, 7, 9, 4, 5, 2, 3, 1, 7, 8, 9, 8, 6, 4]
    y_pred = [0, 0, 2, 2, 0, 2, 3, 3, 6, 8, 8, 7, 9, 6, 5, 2, 8, 1, 7, 3, 9, 8, 6, 4]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Using AML APIs
    cm = {
        "schema_type": "confusion_matrix",
        "schema_version": "1.0.0",
        "data": {
            "class_labels": labels,
            "matrix": confusion_matrix(y_true, y_pred).tolist()
        }
    }
    run = Run.get_context()
    run.log_confusion_matrix('Confusion Matrix', json.dumps(cm))

    # Using Wandb
    cm = wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=labels)
    wandb.log({"conf_mat": cm})

def main():

    # Set experiment/project
    mlflow.set_experiment('pytorch-mnist')
    wandb.init(project='pytorch-mnist')

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Log parameters
    mlflow.log_params(vars(args))
    wandb.config.update(args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader, test_loader = load_data(args, kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    evaluate()

if __name__ == '__main__':
    main()