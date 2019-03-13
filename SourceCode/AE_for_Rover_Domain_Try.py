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

if not os.path.exists('./autoencoder_img'):
    os.mkdir('./autoencoder_img')


num_epochs = 100
batch_size = 64
learning_rate = 1e-3
img_transform = transforms.Compose([transforms.ToTensor()])    #-------------------------------------we do not want to normalize


#------------------------Unroll the joint state list of lists into a single list
joint_state_list_of_lists = []
joint_state_list_of_lists = functools.reduce(operator.iconcat, joint_state_list_of_lists, [])

size_of_input_dimension_of_joint_state = len(joint_state_list_of_lists)

#---------------------------get data
dataset = None #
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)





class autoencoder_custom(nn.Module):
    def __init__(self):
        super(autoencoder_custom, self).__init__()
        self.encoder_linear1 = nn.Linear(size_of_input_dimension_of_joint_state, 512) #in features, out features
        self.encoder_relu1 = nn.LeakyReLU()
        self.encoder_linear2 = nn.Linear(512, 256)
        self.encoder_relu2 = nn.LeakyReLU()
        self.encoder_linear3 = nn.Linear(256, 128)
        self.encoder_relu3 = nn.LeakyReLU()
        self.encoder_linear_4 = nn.Linear(128, 64)
        self.encoder_relu4 = nn.LeakyReLU()
        self.encoder_linear5 = nn.Linear(64, 32)
        self.encoder_relu5 = nn.LeakyReLU()

        #------------------------------------------Encoder ends here

        self.decoder_linear1 = nn.Linear(32, 64)
        self.decoder_relu1 = nn.LeakyReLU()
        self.decoder_linear2  = nn.Linear(64,128)
        self.decoder_relu2 = nn.LeakyReLU()
        self.decoder_linear3 = nn.Linear(128,256)
        self.decoder_relu3 = nn.LeakyReLU()
        self.decoder_linear4 = nn.Linear(256, 512)
        self.decoder_relu4 = nn.LeakyReLU()
        self.decoder_linear_op = nn.Linear(512, size_of_input_dimension_of_joint_state)


    def forward(self, x):

        #Encoder part
        x = self.encoder_linear1(x)
        x= self.encoder_relu1(x)
        x = self.encoder_linear2(x)
        x = self.encoder_relu2(x)
        x = self.encoder_linear3(x)
        x = self.encoder_relu3(x)
        x= self.encoder_linear_4(x)
        x = self.encoder_relu4(x)
        x = self.encoder_linear5(x)
        x = self.encoder_relu5(x)

#-----------------------------------------------------may be we can put tolerance stuff on latent dimensions
        #decoder part
        x = self.decoder_linear1(x)
        x = self.decoder_relu1(x)
        x = self.decoder_linear2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_linear3(x)
        x = self.decoder_relu3(x)
        x = self.decoder_linear4(x)
        x = self.decoder_relu4(x)
        x = self.decoder_linear_op(x)
        return x


model = autoencoder_custom().cuda()
criterion = nn.L1Loss()#-----------------------------------------------Which loss to use, L1 or MSE, L1 loss makes more sense because we do not have outliers in our data
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for sample in dataloader:
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