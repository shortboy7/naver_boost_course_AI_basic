import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

training_epochs = 15
batch_size = 100

root = './data'
mnist_train = dset.MNIST(root = root,train = True,transform = transforms.ToTensor(),download=True)
mnist_test = dset.MNIST(root=root,train=False,transform = transforms.ToTensor(),download = True)

train_loader =  DataLoader(mnist_train,batch_size = batch_size,shuffle=True)
test_loader = DataLoader(mnist_test,batch_size = batch_size,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
linear = torch.nn.Linear(784,10,bias = True).to(device)
torch.nn.init.normal_(linear.weight)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(),lr=0.1)

for epoch in range(training_epochs):
    for i ,(imgs,labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device),labels.to(device)
                imgs = imgs.view(-1,28*28)
                
                outputs = linear(imgs)
                loss = criterion(outputs,labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                _,argmax = torch.max(outputs,1)
                accuracy = (labels == argmax).float().mean()
                
                if (i+1)%100 == 0:
                    print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                          epoch+1, training_epochs, i+1, len(train_loader), loss.item(), accuracy.item()*100))

linear.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i ,(imgs, labels) in enumerate(test_loader):
        imgs,labels =imgs.to(device), labels.to(device)
        imgs = imgs.view(-1,28*28)
        outputs = linear(imgs)
        _,argmax = torch.max(outputs,1)
        total += imgs.size(0)
        correct += (labels==argmax).sum().item()
    print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total *100))