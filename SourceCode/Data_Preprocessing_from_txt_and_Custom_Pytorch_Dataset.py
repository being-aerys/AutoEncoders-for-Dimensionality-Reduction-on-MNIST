import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import numpy as np
import torch.utils.data as utils


num_epochs = 100
batch_size = 64
learning_rate = 1e-3
img_transform = transforms.Compose([transforms.ToTensor()])    #-------------------------------------we do not want to normalize



myfilename = "../data/data_4_10_20.txt"
f=open(myfilename,'r')
line = f.readline()
line_cnt = 1
states=[]
temp_line=[]
temp_line_for_1_state = []
updating_string = []
count = 0

#---------------------------for ninja technique, we need this demo string
demo_string = "a"
updating_string.append(demo_string)


while line:

    demo_string = demo_string + line
    if ("]" in line):

        demo_string = demo_string.replace("[","") #-----------
        demo_string = demo_string.replace("\n","")
        demo_string = demo_string.replace("]","")
        demo_string = demo_string.replace("a","")

        updating_string[count] = demo_string.split()

        count = count + 1
        demo_string = "a"
        updating_string.append(demo_string)

    line = f.readline()

del updating_string[-1]     #removing the "a" added at the end of the loop above
states = updating_string
for i in range(len(states)):

    if(len(states[i]) != 40):
        #print(len(states[i]))
        print("Not all joint states have the same number of elements. Position ", i," has inconsistent data.")


#Since states is a list of lists, convert it into a list of numpy arrays

states_list_x_of_nparrays = []
states_list_y = []
for item in range(len(states)):
    states_list_x_of_nparrays.append(np.asarray(states[item], dtype=np.float32))

for item in range(len(states)):#-------------------------------------------------works as lable, need to provide to make a data set
    states_list_y.append(np.asarray([0], dtype=np.float32))

#Pass the list of arrays here
tensor_x = torch.stack([torch.Tensor(i) for i in states_list_x_of_nparrays]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor(i) for i in states_list_y])

#custom dataset and custom dataloader
custom_dataset = utils.TensorDataset(tensor_x, tensor_y) # create your datset
custom_dataloader = utils.DataLoader(dataset=custom_dataset,batch_size = batch_size,shuffle = True) # create your dataloader
