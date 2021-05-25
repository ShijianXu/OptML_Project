import json
import matplotlib.pyplot as plt

"""
with open('history_acc_resnet18_adam.json') as f:
    acc_adam = json.load(f)
with open('history_acc_resnet18_lbfgs.json') as f:
    acc_lbfgs = json.load(f)
with open('history_acc_resnet18_sgd.json') as f:
    acc_sgd = json.load(f)
"""

"""
with open('history_acc_vgg16_bn_adam.json') as f:
    acc_adam = json.load(f)
with open('history_acc_vgg16_bn_lbfgs.json') as f:
    acc_lbfgs = json.load(f)
with open('history_acc_vgg16_bn_sgd.json') as f:
    acc_sgd = json.load(f)
"""

"""
with open('history_loss_vgg16_bn_adam.json') as f:
    loss_adam = json.load(f)
with open('history_loss_vgg16_bn_lbfgs.json') as f:
    loss_lbfgs = json.load(f)
with open('history_loss_vgg16_bn_sgd.json') as f:
    loss_sgd = json.load(f)
"""

"""
with open('./results/history_loss_resnet18_adam.json') as f:
    loss_adam = json.load(f)
with open('./results/history_loss_resnet18_lbfgs.json') as f:
    loss_lbfgs = json.load(f)
with open('./results/history_loss_resnet18_sgd.json') as f:
    loss_sgd = json.load(f)
"""

"""
with open('./results/history_loss_LR_adam.json') as f:
    loss_adam = json.load(f)
with open('./results/history_loss_LR_lbfgs.json') as f:
    loss_lbfgs = json.load(f)
with open('./results/history_loss_LR_sgd.json') as f:
    loss_sgd = json.load(f)
"""


with open('./results/history_acc_LR_adam.json') as f:
    loss_adam = json.load(f)
with open('./results/history_acc_LR_lbfgs.json') as f:
    loss_lbfgs = json.load(f)
with open('./results/history_acc_LR_sgd.json') as f:
    loss_sgd = json.load(f)


"""
with open('./results/history_loss_mnist_LR_adam.json') as f:
    loss_adam = json.load(f)
with open('./results/history_loss_mnist_LR_lbfgs.json') as f:
    loss_lbfgs = json.load(f)
with open('./results/history_loss_mnist_LR_sgd.json') as f:
    loss_sgd = json.load(f)
"""

"""
with open('./results/history_acc_mnist_LR_adam.json') as f:
    loss_adam = json.load(f)
with open('./results/history_acc_mnist_LR_lbfgs.json') as f:
    loss_lbfgs = json.load(f)
with open('./results/history_acc_mnist_LR_sgd.json') as f:
    loss_sgd = json.load(f)
"""



x = range(1, len(loss_lbfgs)+1)
plt.plot(x, loss_adam, label='Adam')
plt.plot(x, loss_sgd, label='SGD-Momentum')
plt.plot(x, loss_lbfgs, label='L-BFGS')
plt.xlabel('epochs')
plt.ylabel('Test Accuracy')
plt.title('Logistic Regression Test Accuracy on CIFAR10')
plt.legend()
plt.savefig("LR_acc.png")
plt.show()