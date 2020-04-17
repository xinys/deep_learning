import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse

from model import *


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('-l', '--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('-b', '--batch_size', default=10, type=int, help='batch_size')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
def data_prepare():
    # to change
    mean = [0.7726, 0.6524, 0.8035]
    std = [0.0795, 0.1099, 0.0811]
    # to change
    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # to change
    trainset = torchvision.datasets.CIFAR10(root='.\data', train=True, download=False, transform=transforms_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='.\data', train=False, download=False, transform=transforms_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # to change

    return trainloader, testloader

# Model
def model_prepare():
    net = UNet()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)
    criterion = nn.BCELoss()

    return net, optimizer, scheduler, criterion

# Train
def train(net, trainloader, optimizer, scheduler, criterion):
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        total = 0
        running_correct = 0.0
        print("Starting epoch {} / {}".format(epoch+1, args.epochs))
        scheduler.step()
        for batch_index, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_correct += (predict == labels.data).sum().item()
            print("Batch %3d || Loss: %.3f | Acc: %.3f (%d / %d)" % (batch_index, running_loss / (batch_index + 1), 100. * running_correct / total, running_correct, total))

# Test
def test(net, testloader, criterion):
    net.eval()
    test_loss = 0.0
    total = 0
    test_correct = 0.0
    with torch.no_grad():
        for batch_index, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predict == labels.data).sum().item()
            print("Batch %3d || Loss: %.3f | Acc: %.3f (%d / %d)" % (batch_index, test_loss / (batch_index + 1), 100. * test_correct / total, test_correct, total))

if __name__ == '__main__':
    trainloader, testloader = data_prepare()
    net, optimizer, scheduler, criterion = model_prepare()
    train(net, trainloader, optimizer, scheduler, criterion)
    test(net, testloader, criterion)



