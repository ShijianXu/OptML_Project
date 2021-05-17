# PyTorch CIFAR10, L-BFGS
import torch
from torch.nn.modules.linear import Linear
from torch.optim import optimizer
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

from model import Net
from lbfgsnew import LBFGSNew


def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def train(model, trainloader, device, opt, nb_epochs, lr=0.001):
    model.train()

    criterion = nn.CrossEntropyLoss()
    print("Using optimizer: ", opt)

    #TODO adjust optimizer hyperparameters
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif opt == 'lbfgs':
        optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=2, line_search_fn=True, batch_mode=True)        # history_size, max_iter all hyper_parameter
    else:
        raise NotImplementedError


    for epoch in range(nb_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            if opt == 'lbfgs':
                # Def Closure
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                optimizer.step(closure)
                outputs=model(inputs)
                loss=criterion(outputs,labels)

            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')



def main():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    #TODO adjust batch size
    batch_size = 128
    nb_epochs = 100
    lr = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = 10
    
    net = models.vgg16_bn()
    #net = models.resnet18()
    #net = models.resnet50()
    
    net.classifier[6] = nn.Linear(4096, num_classes)
    print(net)

    net.to(device)
    opt = 'sgd' # ['lbfgs' | 'adam']
    train(net, trainloader, device, opt, nb_epochs, lr=lr)
    test(net, testloader, device)


if __name__ == "__main__":
    main()