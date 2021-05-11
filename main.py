# PyTorch CIFAR10, L-BFGS
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

from model import Net
from lbfgsnew import LBFGSNew


def test(model, testloader, device):
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


def train(model, trainloader, device):
    #TODO adjust nb_epochs
    nb_epochs = 10
    criterion = nn.CrossEntropyLoss()

    #TODO adjust optimizer hyperparameters
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = LBFGSNew(model.parameters(), history_size=20, max_iter=10, line_search_fn=True, batch_mode=True)        # history_size, max_iter all hyper_parameter


    for epoch in range(nb_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Def Closure
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if loss.requires_grad:
                    loss.backward()
                return loss

            loss = optimizer.step(closure)


            """
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            """

        
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')



def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    #TODO adjust batch size
    batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()
    net.to(device)

    train(net, trainloader, device)
    test(net, testloader, device)


if __name__ == "__main__":
    main()