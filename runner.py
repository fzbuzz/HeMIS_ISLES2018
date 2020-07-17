import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

unet_mm = UNet_MM()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet_mm.to(device)

modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset_mtt = ISLES2018Dataset_MTT('/home/ubuntu/projs/CS168/HeMIS_ISLES2018/ISLES2018/TRAINING', modalities=modalities)

train_set, test_set = torch.utils.data.random_split(dataset_mtt, (402, 100))

trainloader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                          shuffle=True, num_workers=2)

BCE_loss = nn.BCELoss()

dsc_loss = DiceLoss()
optimizer = optim.Adam(unet_mm.parameters(), lr=1e-5)

# unet_mtt = torch.load(PATH)
# unet_mtt.eval()

for epoch in range(325): #325
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        
        modalities, OT = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = unet_mm(modalities)
        loss = dsc_loss(output, OT)
#         print(loss)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 25 == 24:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 25))
            running_loss = 0.0
print('Finished Training')


total_loss = 0.0
for i, data in enumerate(testloader, 0):
        
        modalities, OT = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            output = unet_mm(modalities)
            loss = dsc_loss(output, OT)
        
            running_loss += loss.item()
            print('[%5d] loss: %.3f' % (i + 1, running_loss))
            running_loss = 0.0
            count = i
            total_loss += loss.item()

    
print('[%s, %5d] loss: %.3f' %
                  ('TOTAL', count + 1, total_loss / (count+1)))

