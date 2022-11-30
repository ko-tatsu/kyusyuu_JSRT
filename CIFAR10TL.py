from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import time
import os 
import copy


from torchvision import datasets
data_path = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#traing用のdatasetを作成
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

#validation用のdatasetを作成
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64, shuffle=False)

from torchvision import models

feature_extract = True
num_classes = 10

model_ft = models.alexnet(pretrained=True)
print(model_ft)
print("-----------------------------------------")  
for name, param in model_ft.named_parameters():
    param.requires_grad = False # Transfer larning:False, Fine Tuning: True
    
num_ftrs = model_ft.classifier[6].in_features
print(num_ftrs)
model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
print("-----------------------------------------")  

print(model_ft)

for name, param in model_ft.named_parameters():
    print(name, param.requires_grad)
    
model_ft = model_ft.to(device)

'''最適化手法の定義'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.01)

'''訓練用の関数を定義'''
def train(train_loader):
    model_ft.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(train_loader) 
    return train_loss

'''評価用の関数を定義'''
def valid(val_loader):
    model_ft.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_ft(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = outputs.max(1, keepdim=True)[1]
            labels = labels.view_as(predicted)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc

'''誤差(loss)を記録する空の配列を用意'''
loss_list = []
val_loss_list = []
val_acc_list = []


'''学習'''
EPOCHS=30
import time

since1 = time.time()
for epoch in range(EPOCHS):
    
    since2 = time.time()
    loss = train(train_loader)
    val_loss, val_acc = valid(val_loader)
    time_elapsed = time.time() - since2
    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc),
          'Training_time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

print("-----------------------------------------")      
time_elapsed = time.time() - since1
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
'''学習の結果と使用したモデルを保存'''
np.save('loss_list.npy', np.array(loss_list))
np.save('val_loss_list.npy', np.array(val_loss_list))
np.save('val_acc_list.npy', np.array(val_acc_list))
torch.save(model_ft.state_dict(), 'ft.pkl')

'''結果の表示'''
plt.plot(range(EPOCHS), loss_list, 'r-', label='train_loss')
plt.plot(range(EPOCHS), val_loss_list, 'b-', label='test_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure()

plt.plot(range(EPOCHS), val_acc_list, 'g-', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')

print('正解率：',val_acc_list[-1]*100, '%')