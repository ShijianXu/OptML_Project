# PyTorch CIFAR10, L-BFGS
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import json

from model import LogisticRegression
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

    return 100 * correct / total


def train(model_name, model, trainloader, testloader, device, opt, nb_epochs, lr=0.001):
    history_loss = []
    history_acc = []

    criterion = nn.CrossEntropyLoss()
    print("Using optimizer: ", opt)

    #TODO adjust optimizer hyperparameters
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'lbfgs':
        optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=2, line_search_fn=True, batch_mode=True)
        #optimizer = optim.LBFGS(model.parameters())
    else:
        raise NotImplementedError


    for epoch in range(nb_epochs):
        # Train for each epoch
        model.train()

        running_loss = 0.0
        for batch_idx, data in enumerate(trainloader, 0):
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
            #if batch_idx % 100 == 99:    # print every 100 mini-batches
            #    print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 100))
            #    running_loss = 0.0

        # Test for each epoch
        epoch_loss = running_loss / (batch_idx+1)
        epoch_acc = test(model, testloader, device)

        print("Epoch {} train loss: {}, test acc: {}".format(epoch+1, epoch_loss, epoch_acc))
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)
        
    print('Finished Training')
    with open('history_loss_mnist' + '_' + model_name + '_' + opt + '.json', 'w') as f:
        json.dump(history_loss, f)
    with open('history_acc_mnist' + '_' + model_name + '_' + opt + '.json', 'w') as f:
        json.dump(history_acc, f)


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

    # MNIST Dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # CIFAR10 Dataset
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = 10
    
    model_names = ['LR']    #'vgg16_bn', 'resnet18']
    optim_names = ['sgd', 'adam', 'lbfgs']

    for model_name in model_names:     
        for opt in optim_names:
            ## !!! TODO every iter should create a new model !!!
            print("creating model: ", model_name)
            print("using optimizer: ", opt)

            if model_name == 'vgg16_bn':
                model = models.vgg16_bn()
                model.classifier[6] = nn.Linear(4096, num_classes)
            elif model_name == 'resnet18':
                model = models.resnet18()
                model.fc = nn.Linear(512, num_classes)
            elif model_name == 'LR':
                model = LogisticRegression(784, num_classes)
                
            model.to(device)
            train(model_name, model, trainloader, testloader, device, opt, nb_epochs, lr=lr)


if __name__ == "__main__":
    main()