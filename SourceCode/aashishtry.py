import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import functools
import operator
import csv
import time
import numpy as np
import pandas as pd
import torch.utils.data as utils


if not os.path.exists('./autoencoder_img'):
    os.mkdir('./autoencoder_img')


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
demo_string = "a"
updating_string.append(demo_string)


while line:

    demo_string = demo_string + line
    #print(demo_string)
    #time.sleep(5)
    if ("]" in line):
        #print("Found ]")

        demo_string = demo_string.replace("[","") #-----------
        demo_string = demo_string.replace("\n","")
        demo_string = demo_string.replace("]","")
        demo_string = demo_string.replace("a","")

        #print("inside", demo_string)

        updating_string[count] = demo_string.split()

        #print("Joint State ",count," :", updating_string[count])
        count = count + 1
        demo_string = "a"
        updating_string.append(demo_string)

        #time.sleep(5)
        #temp_line_for_1_state.append(updating_string[count])

        #flattened = [val for sublist in temp_line_for_1_state for val in sublist]


        #time.sleep(488)
        #
        # print()

    #to remove the last added a, do this

    line = f.readline()

del updating_string[-1]     #removing the "a" added at the end of the loop above
#print("(states) ",(len(updating_string[11700])))
#time.sleep(333)
for i in range(len(updating_string)):

    if(len(updating_string[i]) != 40):
        #print(len(states[i]))
        print("Not all joint states have the same number of elements. Position ", i," has inconsistent data.")


#convert a list of lists into a list of numpy arrays
# states = np.array(states)
# print(type(states[0]))

# states_array = np.asarray(states)
states_list_x_of_nparrays = []
states_list_y = []
for item in range(len(states)):
    states_list_x_of_nparrays.append(np.asarray(states[item], dtype=np.float32))

for item in range(len(states)):#-------------------------------------------------works as lable, need to provide to make a data set
    states_list_y.append(np.asarray([0], dtype=np.float32))


for i in range(len(states_list_x_of_nparrays)):
    if(len(states_list_x_of_nparrays[i]) != 40):
        print(len(states_list_x_of_nparrays[i]))


tensor_x = torch.stack([torch.Tensor(i) for i in states_list_x_of_nparrays]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor(i) for i in states_list_y])



custom_dataset = utils.TensorDataset(tensor_x, tensor_y) # create your datset
time.sleep(5)
custom_dataloader = utils.DataLoader(dataset=custom_dataset,batch_size = 1,shuffle = True) # create your dataloader




#dataloader = DataLoader(dataset=a, batch_size=1, shuffle=True)

for sample in custom_dataloader:

    print(sample)
    time.sleep(2222)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(len(states[0]), 512),
                                     nn.LeakyReLU(),
                                     nn.Linear(512, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32),
                                     nn.LeakyReLU()

                                     )

        # self.encoder_linear1 = nn.Linear(len(training_dataset[0]), 512) #in features, out features
        # self.encoder_relu1 = nn.LeakyReLU()
        # self.encoder_linear2 = nn.Linear(512, 256)
        # self.encoder_relu2 = nn.LeakyReLU()
        # self.encoder_linear3 = nn.Linear(256, 128)
        # self.encoder_relu3 = nn.LeakyReLU()
        # self.encoder_linear_4 = nn.Linear(128, 64)
        # self.encoder_relu4 = nn.LeakyReLU()
        # self.encoder_linear5 = nn.Linear(64, 32)
        # self.encoder_relu5 = nn.LeakyReLU()

        #------------------------------------------Encoder ends here

        self.decoder = nn.Sequential(nn.Linear(32, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 512),
                                     nn.LeakyReLU(),
                                     nn.Linear(512, len(states[0]))


        )

        # self.decoder_linear1 = nn.Linear(32, 64)
        # self.decoder_relu1 = nn.LeakyReLU()
        # self.decoder_linear2  = nn.Linear(64,128)
        # self.decoder_relu2 = nn.LeakyReLU()
        # self.decoder_linear3 = nn.Linear(128,256)
        # self.decoder_relu3 = nn.LeakyReLU()
        # self.decoder_linear4 = nn.Linear(256, 512)
        # self.decoder_relu4 = nn.LeakyReLU()
        # self.decoder_linear_op = nn.Linear(512, len(training_dataset[0]))


    def forward(self, x):

        #Encoder part

        x = self.encoder(x)
        # x = self.encoder_linear1(x)
        # x= self.encoder_relu1(x)
        # x = self.encoder_linear2(x)
        # x = self.encoder_relu2(x)
        # x = self.encoder_linear3(x)
        # x = self.encoder_relu3(x)
        # x= self.encoder_linear_4(x)
        # x = self.encoder_relu4(x)
        # x = self.encoder_linear5(x)
        # x = self.encoder_relu5(x)

#-----------------------------------------------------may be we can put tolerance stuff on latent dimensions
        #decoder part


        x = self.decoder(x)


        # x = self.decoder_linear1(x)
        # x = self.decoder_relu1(x)
        # x = self.decoder_linear2(x)
        # x = self.decoder_relu2(x)
        # x = self.decoder_linear3(x)
        # x = self.decoder_relu3(x)
        # x = self.decoder_linear4(x)
        # x = self.decoder_relu4(x)
        # x = self.decoder_linear_op(x)
        return x


model = AutoEncoder().cuda()
criterion = nn.L1Loss()#-----------------------------------------------Which loss to use, L1 or MSE, L1 loss makes more sense because we do not have outliers in our data
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for sample in dataloader:
        print(len(sample))
        #print(sample[39])
        time.sleep(5)
        sample = Variable(sample).cuda()
        # ----------------------------------------------forward
        output = model(sample)

        # print("original image ", sample)
        # print("output image", output)

        loss = criterion(output, sample)#------------------take care of the order of the sample and the output

        # ----------------------------------------------backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data.item()))

torch.save(model.state_dict(), './vanilla_autoencoder_for_rd.pth')