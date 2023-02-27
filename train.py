
import cv2, time,sys
from backbone.customnet import CustomNet

import numpy as np

import torch

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from custom_dataset import *
sys.path.append('./func_transforms')

#Local Imports
from torch.utils.data import random_split, DataLoader
from backbone.pplcnet import _pplcnet


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

validation_transform = transforms.Compose([
                        transforms.ToTensor(),   
                        normalize
                        ])

data_path = 'data'
hdb = CustomDataSet(data_path,transform=None)



train_size = int(0.8 * len(hdb))
validation_size = len(hdb) - train_size
train_subset, validation_subset = random_split(hdb, [train_size, validation_size])

train_dataset = DatasetFromSubset(train_subset,train_transform)
validation_dataset = DatasetFromSubset(validation_subset,validation_transform)

del hdb,train_subset,validation_subset


print('Train Dataset Length: ',len(train_dataset), flush=True)
print('Validation Dataset Length: ',len(validation_dataset), flush=True)

######## SMALL NET ############
backbone = _pplcnet(width_mult=2.0, class_num=64)
net = backbone
##############################
######## BIG NET ############
# backbone = torchvision.models.resnet34(pretrained = True)
# net = CustomNet(backbone)
##############################
##Load pretrain model
# state_dict_ = torch.load("snapshots/res34.pt", map_location=torch.device('cuda'))
# net.load_state_dict(state_dict_['state'])

#Setup dataloaders for train and validation
batch_size = 32

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True)

validation_loader = DataLoader(validation_dataset, 
                          batch_size=batch_size,
                          shuffle=False)




criterion = nn.CrossEntropyLoss().to('cuda')
learning_rate = 0.0001

optimizer = optim.AdamW(net.parameters(),
                       lr=learning_rate)



# **7. define learning rate decay scheduler:-**

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# **8. Place Model in GPU:-**

device = torch.device("cuda")
net.to(device)

history = {'train_loss' : [], 'validation_loss' : []}

#this holds best states wrt validation loss that we are saving,
#we can use these to resume from last best loss or keep the best one for inference later. 
best_states = {}


# **10. Define Training and Validation Function:-**

def train_net(n_epochs):
    global history
    
    #average test loss over epoch used to find best model parameters
    if(len(history['validation_loss'])>0):
        min_loss_idx = np.argmin(history['validation_loss'])
        best_loss = history['validation_loss'][min_loss_idx]
    else:
        best_loss = 10000
    
    #this is where we store all our losses during epoch before averaging them and storing in history
    itter_loss = []  
    t = time.time()


    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # prepare the model for training
        net.train()

        train_running_loss = []
        
        print(f'Epoch: {epoch}', flush=True)
        # train on batches of data, assumes you already have train_loader
        print("Printing Train Loss...", flush=True)
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding poses
            images,gt = data
            # put data inside gpu
            images = images.float().to(device)

            gt = gt.squeeze().to(device)

            # call model forward pass
            predicted = net(images)
            # calculate the softmax loss between predicted poses and ground truth poses
            loss = criterion(predicted, gt)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()
            
            # get loss
            loss_scalar = loss.item()

            #update loss logs
            train_running_loss.append(loss_scalar)
            
            #collect batch losses to compute epoch loss later
            itter_loss.append(loss_scalar)
            
            if batch_i % 10 == 9:    # print every 100 batches
                print(f'Batch: {batch_i+1}, Avg. Train Loss: {np.mean(train_running_loss)}', flush=True)
                train_running_loss = []
        else: 
            history['train_loss'].append(np.mean(itter_loss))
            itter_loss.clear()
            validation_running_loss = []
            net.eval()
            with torch.no_grad():
                print("Printing Validation Loss...", flush=True)
                for batch_i, data in enumerate(validation_loader):
                    # get the input images and their corresponding poses
                    images,gt = data

                    # put data inside gpu
                    images = images.float().to(device)

                    gt = gt.squeeze().to(device)

                    # call model forward pass
                    predicted = net(images)

                    # calculate the softmax loss between predicted poses and ground truth poses
                    loss = criterion(predicted, gt)
                    
                    #convert loss into a scalar using .item()
                    loss_scalar = loss.item()
                    
                    #add loss to the running_loss, use
                    validation_running_loss.append(loss_scalar)
                    
                    #collect batch losses to compute epoch loss later
                    itter_loss.append(loss_scalar)
                    
                    
                    if batch_i % 10 == 9:    # print every 10 batches
                        print(f'Batch: {batch_i+1}, Avg. Validation Loss: {np.mean(validation_running_loss)}', flush=True)
                        validation_running_loss = []
                        
            history['validation_loss'].append(np.mean(itter_loss))
            itter_loss.clear()
            
            #if current is better than previous, update state_dict and store current as best
            if(history['validation_loss'][-1] < best_loss):
                best_loss = history['validation_loss'][-1]
                #Save Model Checkpoint
                torch.save({
                            "state" : net.state_dict()
                            }, 'snapshots/rest34.pt')
                
                print(f'Model improved since last epoch! New Best Val Loss: {best_loss}', flush=True)
                
        # update lr schedular
        scheduler.step()                                     

    print('Finished Training', flush=True)

try:
    # train your network
    n_epochs = 100 # start small, and increase when you've decided on your model structure and hyperparams
    train_net(n_epochs)

except KeyboardInterrupt:
    print('Stopping Training...', flush=True)
