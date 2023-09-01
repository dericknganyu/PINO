"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
sys.path.insert(1, '/home/derick/Documents/PINO/')

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utils import *

import time
from datetime import datetime
from readData import readtoArray
import os
from colorMap import parula
import argparse

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################


print("torch version is ",torch.__version__)
ntrain0 = 1000
ntest0 = 5000
learning_rate0 = 0.001
step_size0 = 100
gamma0 = 0.5
modes0 = 12
width0 = 32
batch_size0 = 10 #100
epochs0 = 500 #500
wd0 = 1e-4

ntrain = 1
learning_rate = 0.001#25
step_size = 100
gamma = 0.5
modes = 12
width = 32

parser = argparse.ArgumentParser(description='parse batch_size, epochs and resolution')
parser.add_argument('--bs',  default=1, type = int, help='batch-size')
parser.add_argument('--ep', default=500, type = int, help='epochs')
parser.add_argument('--res', default=32, type = int, help='resolution')
parser.add_argument('--wd', default=1e-4, type = float, help='weight decay')
parser.add_argument('--al', default=10, type = float, help='alpha for operator loss')

args = parser.parse_args()

batch_size = args.bs #100
epochs = args.ep #500
res = args.res + 1#32#sys.argv[1]
wd = args.wd
alpha = args.al

print("\nUsing batchsize = %s, epochs = %s, and resolution = %s\n"%(batch_size, epochs, res))
params = {}
params["xmin"] = 0
params["ymin"] = 0
params["xmax"] = 1
params["ymax"] = 1

################################################################
# load data and data normalization
################################################################
PATH = "aUL_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
#Read Datasets
X_train, Y_train, _, _ = readtoArray(PATH, 1024, 5000, Nx = 512, Ny = 512)

print ("Converting dataset to numpy array.")
tt = time.time()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
#X_test  = np.array(X_test )
#Y_test  = np.array(Y_test )
print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Subsampling dataset to the required resolution.")
tt = time.time()
X_train = SubSample(X_train, res, res)
Y_train = SubSample(Y_train, res, res)
#X_test  = SubSample(X_test , res, res)
#Y_test  = SubSample(Y_test , res, res)
print ("    Subsampling completed after %.2f minutes"%((time.time()-tt)/60))

print ("Taking out the required train/test size.")
tt = time.time()
x_train = torch.from_numpy(X_train[:ntrain, :, :]).float()
y_train = torch.from_numpy(Y_train[:ntrain, :, :]).float()
#x_test  = torch.from_numpy(X_test[ :ntest,  :, :]).float()
#y_test  = torch.from_numpy(Y_test[ :ntest,  :, :]).float()
print ("    Taking completed after %s seconds"%(time.time()-tt))
print("...")

x_normalizer = torch.load("normalisers/fwd-%s-x_normalizer.pt"%(res-1))
y_normalizer = torch.load("normalisers/fwd-%s-y_normalizer.pt"%(res-1))
x_train = x_normalizer.encode(x_train)
#x_test = x_normalizer.encode(x_test)
#print(x_train)
#y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)


x_train = x_train.reshape(ntrain,res,res,1)
#x_test = x_test.reshape(ntest,res,res,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
#test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
model_0 = FNO2d(modes, modes, width).cuda()

print("Model has %s parameters"%(count_params(model)))

if res == 513:
    test_l2 = 0.001877
    timestamp = '20220716-131345'
if res == 257:
    test_l2 = 0.001805
    timestamp = '20220716-005009'
if res == 129:
    test_l2 = 0.001856
    timestamp = '20220715-172512'
if res == 65:
    test_l2 = 0.001724
    timestamp = '20220715-162920'
if res == 33:
    test_l2 = 0.00211
    timestamp = '20220715-160503'
if res == 17:
    test_l2 = 0.003005
    timestamp = '20220715-174507'

ModelInfos = "_%03d"%(res)+"~res_"+str(np.round(test_l2,6))+"~RelL2TestError_"+str(ntrain0)+"~ntrain_"+str(ntest0)+"~ntest_"+str(batch_size0)+"~BatchSize_"+str(learning_rate0)+\
            "~LR_"+str(wd0)+"~Reg_"+str(gamma0)+"~gamma_"+str(step_size0)+"~Step_"+str(epochs0)+"~epochs_"+timestamp
     
model.load_state_dict(torch.load("files/last_model"+ModelInfos+".pt"))
model_0.load_state_dict(torch.load("files/last_model"+ModelInfos+".pt"))
with torch.no_grad():
    model_0.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
TIMESTAMP = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
if os.path.isfile('files/lossData_'+TIMESTAMP+'.txt'):
    os.remove('files/lossData_'+TIMESTAMP+'.txt')
	
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    error = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, res, res)
        out_0 = model_0(x).reshape(batch_size, res, res)

        out = y_normalizer.decode(out)
        out_0 = y_normalizer.decode(out_0)

        y = y_normalizer.decode(y)
        pde_loss_train  = darcy_loss(out, x.reshape(batch_size, res, res))
        loss_0 = myloss(out.view(batch_size,-1), out_0.view(batch_size,-1))

        loss = pde_loss_train + alpha*loss_0
        
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        error += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    scheduler.step()

    train_l2 /= ntrain
    error /= ntrain



    t2 = default_timer()
    print("epoch: %s, completed in %.4f seconds. Loss: %.4f | Error: %4f"%(ep+1, t2-t1, train_l2, error))

    file = open('files/lossData_'+TIMESTAMP+'.txt',"a")
    file.write(str(ep+1)+" "+str(train_l2)+" "+str(error)+"\n")

ModelInfos = "-tto_%03d"%(res)+"~res_"+str(np.round(error,6))+"~RelL2TestError_"+str(alpha)+"~alpha_"+str(ntrain)+"~ntrain_"+str(batch_size)+"~BatchSize_"+str(learning_rate)+\
            "~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs_"+time.strftime("%Y%m%d-%H%M%S")  
                            
torch.save(model.state_dict(), "files/last_model"+ModelInfos+".pt")
os.rename('files/lossData_'+TIMESTAMP+'.txt', 'files/lossData'+ModelInfos+'.txt')

dataTrain = np.loadtxt('files/lossData'+ModelInfos+'.txt')
steps     = dataTrain[:,0]            
lossTrain = dataTrain[:,1]            
errorTrain = dataTrain[:,2]  

print("Ploting Loss VS training step...")
fig = plt.figure(figsize=(15, 10))
plt.yscale('log')
plt.plot(steps, lossTrain, label = 'Training Loss')
plt.plot(steps, errorTrain, label = 'Training Error')
plt.xlabel('epochs')#, fontsize=16, labelpad=15)
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.title("lr = %s test error = %s"%(learning_rate, str(np.round(train_l2,6))))
plt.savefig("figures/Error_VS_TrainingStep"+ModelInfos+".png", dpi=500)




#def use_model():#(params, model,device,nSample,params):

model.load_state_dict(torch.load("files/last_model"+ModelInfos+".pt"))
model.eval()


print()
print()

#Just a file containing data sampled in same way as the training and test dataset
fileName = "aUL_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
F_train, U_train, F_test, U_test= readtoArray(fileName, 1, 1, 512, 512)

F_train = SubSample(np.array(F_train), res, res)

print("Starting the Verification with Sampled Example")
tt = time.time()
U_FDM = SubSample(np.array(U_train), res, res)[0]

print("      Doing PINO on Example...")
tt = time.time()
ff = torch.from_numpy(F_train).float()
ff = x_normalizer.encode(ff)
ff = ff.reshape(1,res,res,1).cuda()#torch.cat([ff.reshape(1,res,res,1), grid.repeat(1,1,1,1)], dim=3).cuda()

U_PINO = y_normalizer.decode(model(ff).reshape(1, res, res)).detach().cpu().numpy()
U_PINO = U_PINO[0] 
print("            PINO completed after %s"%(time.time()-tt))

myLoss = LpLoss(size_average=False)
print()
print("Ploting comparism of FDM and PINO Simulation results")
fig = plt.figure(figsize=((5+2)*4, 5))

fig.suptitle("Plot of $- \Delta u = f(x, y)$ on $\Omega = ]0,1[ x ]0,1[$ with $u|_{\partial \Omega}  = 0.$")

colourMap = parula() #plt.cm.jet #plt.cm.coolwarm

plt.subplot(1, 4, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Input")
plt.imshow(F_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM")
plt.imshow(U_FDM, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 3)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("PINO")
plt.imshow(U_PINO, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 4)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM-PINO, RelL2Err = "+str(round(myLoss.rel_single(U_PINO, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_PINO), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.savefig('figures/compare'+ModelInfos+'.png',dpi=500)

plt.show()