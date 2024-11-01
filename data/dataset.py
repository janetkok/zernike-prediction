from PIL import Image
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
import random
import math
import torch.nn.functional as F
from pathlib import Path
import os


class Aberration():
    def __init__(self,img_size,device,precision=torch.half,zRange=1.0,bias_z=4, zernike=[3,5,6,7],bias_val=[-1,0,1],npts=97):
        self.zRange_start = 0
        self.zRange_end = 200 #0-200 ->-1.0 to 1.0
        if zRange==0.5:
            self.zRange_start = 50
            self.zRange_end = 150 
        self.zRange = zRange
        self.tRange = int(zRange*100)
        self.img_size = img_size
        self.device = device

        self.precisionFloat = precision
        self.precisionInt = torch.int
        if self.precisionFloat==torch.half:
            self.precisionInt = torch.short
        self.bias_z = bias_z

        self.num_channel = len(bias_val)
        self.bias_val = bias_val
        self.zernike = zernike
        self.npts=npts #125
        self.npad=2001 #2001
        self.nhpad=math.ceil(self.npad/2)-1
        self.nrange =64
        self.nex=int(round(self.npad-self.npts)/2)

        self.znorm=0
        self.phaseapp=self.znorm*math.pi/2
        self.z_sub = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6]
        self.defocus = [0.0]

        self.obs = [self.npts/self.npad,1,1]
        self.pup = self.annulus()
        self.circ_res = self.circ(self.npts)
        self.rad= self.circ_res*self.circrad(self.npts)
        self.rad=F.pad(self.rad,(self.nex,self.nex,self.nex,self.nex))
        self.pupPhase = self.pup*torch.exp(1j*self.phaseapp*self.rad*self.rad)
        self.zernike_cartesian_matrix()


    def annulus(self):
        if self.obs[2]>1:
            print('error outer ring out of aperture')
            return

        mattemp = self.circrad(self.npad)

        pup=torch.zeros(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        temp1=self.obs[0]*torch.ones(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        temp2=self.obs[1]*torch.ones(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        temp3=self.obs[2]*torch.ones(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        pup=(mattemp<=temp1)|((mattemp>temp2)&(mattemp<=temp3))

        nh=math.ceil(self.npad/2)-1
        if self.obs[0]==0:
            pup[nh,nh]=0 # just blocks central pixel

        return pup

    def circrad(self,npts):

        xtemp = torch.linspace(-1,1,steps=npts,device=self.device,dtype=self.precisionFloat)
        ytemp=xtemp

        mattemp1 = xtemp.view(len(xtemp),1)@torch.ones(1,npts,device=self.device,dtype=self.precisionFloat)
        mattemp2 = 1j*torch.ones(npts,1,device=self.device)*ytemp
        mattemp = mattemp1+mattemp2
        mattemp = abs(mattemp)

        return mattemp

    def circ(self,npts):
        mattemp = self.circrad(npts)
        return (mattemp<=torch.ones(npts,npts,device=self.device))
   
    def zernike_cartesian_matrix(self):
        # my simplified version of zernike polynomial function 
        # Gives Zernikes in Cartesians over a unit circle.
        # param: subscript superscript azimuthal angle
        x=torch.linspace(-1,1,steps=self.npts,device=self.device,dtype=self.precisionFloat) #Unit radius
        x=torch.flip(x,dims=(0,))
        x = x.repeat(self.npts,1)
        y=x.t()
        #dx=2/(self.npts-1)
        Z = torch.zeros(28,self.npts,self.npts,dtype=self.precisionFloat,device=self.device)
        Z[0,:,:] = 1*torch.ones(self.npts,self.npts,dtype=self.precisionFloat,device=self.device)
        Z[1,:,:] = 2*x #tilt
        Z[2,:,:] = 2*y
        Z[3,:,:] = math.sqrt(6)*(2*x*y) #astigmatism
        Z[4,:,:] = math.sqrt(3)*((2*x*x)+(2*y*y)-1) #defocus
        Z[5,:,:] = math.sqrt(6)*((-x*x)+(y*y)) #astigmatism
        Z[6,:,:] = 2*math.sqrt(2)*((-x**3)+(3*x*y*y)) #trefoil
        Z[7,:,:] = 2*math.sqrt(2)*((-2*x)+(3*x**3)+(3*x*y*y))#coma
        Z[8,:,:] = 2*math.sqrt(2)*((-2*y)+(3*y**3)+(3*y*x*x)) #coma
        Z[9,:,:] = 2*math.sqrt(2)*((-y**3)+(3*y*x*x)) #trefoil
        Z[10,:,:] = math.sqrt(10)*((-4*x**3*y)+(4*x*y**3))
        Z[11,:,:] = math.sqrt(10)*((-6*x*y)+(8*y*x**3)+(8*x*y**3))
        Z[12,:,:] = math.sqrt(5)*(1-(6*x*x)-(6*y*y)+(6*x**4)+(12*x*x*y*y)+(6*y**4))
        Z[13,:,:] = math.sqrt(10)*((3*x*x)-(3*y*y)-(4*x**4)+(4*y**4))
        Z[14,:,:] = math.sqrt(10)*((x**4)-(6*y**2*x**2)+(y**4))
        Z[15,:,:] = 2*math.sqrt(3)*((x**5)-(10*x**3*y**2)+(5*x*y**4))
        Z[16,:,:] = 2*math.sqrt(3)*((4*x**3)-(12*y**2*x)-(5*x**5)+(10*y**2*x**3)+(15*y**4*x))
        Z[17,:,:] = 2*math.sqrt(3)*((3*x)-(12*x**3)-(12*x*y**2)+(10*x**5)+(20*x**3*y**2)+(10*x*y**4))
        Z[18,:,:] = 2*math.sqrt(3)*((3*y)-(12*y**3)-(12*y*x**2)+(10*y**5)+(20*x**2*y**3)+(10*y*x**4))
        Z[19,:,:] = 2*math.sqrt(3)*(-(4*y**3)+(12*x**2*y)+(5*y**5)-(10*x**2*y**3)-(15*x**4*y))
        Z[20,:,:] = 2*math.sqrt(3)*((y**5)-(10*y**3*x**2)+(5*y*x**4))
        Z[21,:,:] = math.sqrt(14)*((6*x**5*y)-(20*x**3*y**3)+(6*x*y**5))
        Z[22,:,:] = math.sqrt(14)*((20*x**3*y)-(20*x*y**3)-(24*x**5*y)+(24*x*y**5))
        Z[23,:,:] = math.sqrt(14)*((12*x*y)-(40*x**3*y)-(40*x*y**3)+(30*x**5*y)+(60*x**3*y**3)+(30*x*y**5))
        Z[24,:,:] = math.sqrt(7)*(-1+(12*x*x)+(12*y*y)-(30*x**4)-(60*x**2*y**2)-(30*y**4)+(20*x**6)+(60*x**4*y**2)+(60*x**2*y**4)+(20*y**6))
        Z[25,:,:] = math.sqrt(14)*(-(6*x**2)+(6*y**2)+(20*x**4)-(20*y**4)-(15*x**6)-(15*x**4*y**2)+(15*x**2*y**4)+(15*y**6))
        Z[26,:,:] = math.sqrt(14)*(-(5*x**4)+(30*x**2*y**2)-(5*y**4)+(6*x**6)-(30*x**4*y**2)-(30*y**4*x**2)+(6*y**6))
        Z[27,:,:] = math.sqrt(14)*(-(x**6)+(15*x**4*y**2)-(15*y**4*x**2)+(y**6))

        self.Z = Z*self.circ_res

    def gen(self,C=None):
        aberration = torch.empty(size=(self.num_channel,self.img_size,self.img_size),device=self.device)
        
        Zsum_noDef=torch.zeros(self.npts,self.npts,device=self.device,dtype=self.precisionFloat)
        Zsum=torch.zeros(self.npts,self.npts,device=self.device,dtype=self.precisionFloat)

        for j,m1 in enumerate(self.zernike): #might want to change to z4-z9
            if C[j]!=0:
                Zsum_noDef = Zsum_noDef + (C[j]*self.Z[m1,:,:])
        for j,k in enumerate(self.bias_val):    
            if k!=0:
                Zsum =Zsum_noDef + (k*self.Z[self.bias_z,:,:])
            else:
                Zsum = Zsum_noDef 

            Zpadsum=F.pad(Zsum,(self.nex,self.nex,self.nex,self.nex)) 

            Zpadsum=self.pupPhase*torch.exp(1j*Zpadsum) #includes phase from physical defocus

            Zpadsum=torch.fft.ifftshift(Zpadsum)
            out=torch.fft.fft2(Zpadsum)
            # del Zpadsum
            out=torch.fft.fftshift(out)
            out=abs(out)
            
            outsmall=out[self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1),self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1)]

            outsmall=outsmall/(torch.max(outsmall))

            aberration[j,:,:]=outsmall
        # del outsmall,out
        return aberration, C
    
 

def gen_gaussian_kernel(kernel_size=128,sigma=10):
    # create a 1D Gaussian kernel
    kernel_1d = torch.tensor([torch.exp(-(x - kernel_size//2)**2/torch.tensor(2*sigma**2)) for x in range(kernel_size)],device='cuda')
    gaussian_kernel = torch.outer(kernel_1d, kernel_1d)
    return gaussian_kernel


def add_poisson_noise(image, noise_level):
    """Add Poisson noise to an image."""
    noisy_image = torch.poisson(image * noise_level) / noise_level
    return noisy_image


def add_gaussian_noise(image, mean=0, std=0.01):
    """Add Gaussian noise to an image."""
    noise = torch.randn(image.size(),device='cuda') * std + mean
    noisy_image = torch.clamp(image + noise, min=0, max=1)
    return noisy_image

class AberrationStrehl(Aberration):
    def __init__(self,img_size,device,precision=torch.half,zRange=1.0,bias_z=4, zernike=[3,5,6,7],bias_val=[-1,0,1],npts=97):
       super().__init__(img_size,device,precision,zRange,bias_z, zernike,bias_val,npts)


    def gen(self,C=None):
        aberration = torch.empty(size=(self.num_channel,self.img_size,self.img_size),device=self.device)
        
        Zsum_noDef=torch.zeros(self.npts,self.npts,device=self.device,dtype=self.precisionFloat)
        Zsum=torch.zeros(self.npts,self.npts,device=self.device,dtype=self.precisionFloat)

        for j,m1 in enumerate(self.zernike): #might want to change to z4-z9
            if C[j]!=0:
                Zsum_noDef = Zsum_noDef + (C[j]*self.Z[m1,:,:])
        for j,k in enumerate(self.bias_val):    
            if k!=0:
                Zsum =Zsum_noDef + (k*self.Z[self.bias_z,:,:])
            else:
                Zsum = Zsum_noDef 

            Zpadsum=F.pad(Zsum,(self.nex,self.nex,self.nex,self.nex)) 

            Zpadsum=self.pupPhase*torch.exp(1j*Zpadsum) #includes phase from physical defocus

            Zpadsum=torch.fft.ifftshift(Zpadsum)
            out=torch.fft.fft2(Zpadsum)
            # del Zpadsum
            out=torch.fft.fftshift(out)
            out=abs(out)
            
            outsmall=out[self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1),self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1)]

            aberration[j,:,:]=outsmall*outsmall

        return aberration, C
    
class AberrationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_size,num_zernike,precision,bias_z,val_test_size, zernike,bias_val,npts):
        self.dataset_size = dataset_size
        self.zRange_start = 0
        self.zRange_end = 200 #0-200 ->-1.0 to 1.0
        self.val_test_size = val_test_size
        self.num_zernike = num_zernike
        self.C_val_test = torch.randint(self.zRange_start, self.zRange_end ,size=(val_test_size,self.num_zernike))
        self.C_val_test = (self.C_val_test-100)*0.01

        self.gen_aberration = Aberration(128,device='cuda',precision=precision,bias_z=bias_z, zernike=zernike,bias_val=bias_val,npts=npts)

    def __getitem__(self,idx):
        if idx >= self.val_test_size:
          gen_tr = True
          while gen_tr:
            C = torch.randint(self.zRange_start, self.zRange_end ,size=(self.num_zernike,))
            C = (C-100)*0.01
            is_equal = torch.eq(self.C_val_test, C).all(dim=1)
            # Execute code if any set of numbers is equal
            if not is_equal.any():
              gen_tr = False
          aberration, coeffs = self.gen_aberration.gen(C=C)
        else:
          aberration, coeffs = self.gen_aberration.gen(C=self.C_val_test[idx])

        return aberration, coeffs

    def __len__(self):
        return self.dataset_size


class NoisyDataset(torch.utils.data.Dataset):
    """custom dataset: input is mat file
    Arguments:
        ds: dataset containing path to images and their respective labels
        split_idx: indices of samples to be used
        aug: perform augmentation
        transforms: basic transformation
        transformsAug: transformation with augmentation
    """
    def __init__(self, dataset_size,num_zernike,precision,bias_z,val_test_size, zernike,bias_val,npts,z_range):
        self.dataset_size = dataset_size
        self.zRange_start = 0
        self.zRange_end = 200 #0-200 ->-1.0 to 1.0
        if z_range != 1.0:
            self.zRange_start = int((1.0-z_range)*100)
            self.zRange_end = int(100 + (z_range*100))
        C_val_test = torch.randint(self.zRange_start, self.zRange_end ,size=(val_test_size,num_zernike))
        self.C_val_test = (C_val_test-100)*0.01
        self.val_test_size = val_test_size
        self.num_zernike = num_zernike
        self.channel_len = len(bias_val)
        self.gen_aberration = 	Aberration(128,device='cuda',precision=precision,bias_z=bias_z,zernike=zernike,bias_val=bias_val,npts=npts)
        self.gaussian_kernel = gen_gaussian_kernel(kernel_size=128,sigma=30) #previous sigma is 20

    def __getitem__(self,idx):
        C=torch.zeros(self.num_zernike,)
        if idx >= self.val_test_size:
          gen_tr = True
          while gen_tr:
            C = torch.randint(self.zRange_start, self.zRange_end ,size=(self.num_zernike,))
            C = (C-100)*0.01
            is_equal = torch.eq(self.C_val_test, C).all(dim=1)
            # Execute code if any set of numbers is equal
            if not is_equal.any():
              gen_tr = False
        else:
          C = self.C_val_test[idx]
        
        aberration, coeffs = self.gen_aberration.gen(C=C)

        for i in range(self.channel_len):
            aberration[i] = aberration[i] * self.gaussian_kernel
            aberration[i] /= aberration[i].max() 
            aberration[i] = add_poisson_noise(aberration[i], noise_level=2000)
            aberration[i] = add_gaussian_noise(aberration[i], mean=0, std=0.02)

        return aberration, C

    def __len__(self):
        return self.dataset_size

def generate_array(num_zernike):
  array = torch.rand(num_zernike) * 0.4 - 0.2
  zeros = (array == 0).sum().item()

  # If there are less than 5 zeros, replace random elements with zeros
  while zeros < 5:
    index = torch.randint(0, num_zernike, (1,))
    array[index] = 0
    zeros = (array == 0).sum().item()

  return array


    
class PSF_RealDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, exts = ['jpg', 'jpeg', 'png','PNG'],img_size=128,channel=[0,1,2]):

        self.ds = pd.read_csv(data_path) 
        self.dir = data_path.rsplit('/', 1)[0]
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            ])
        self.img_size = img_size
        self.channel = channel

    def __getitem__(self,idx):
        d = self.ds.iloc[idx]
        coeffs = torch.tensor(d[1:].tolist())
        aberration = torch.zeros(len(self.channel),self.img_size,self.img_size)
        for i,j in enumerate(self.channel):
          im = Image.open(self.dir+'/'+str(int(d[0]))+'_'+str(j)+'.png')
          red, _, _ = im.split() #red channel contains the most info
          red = self.transform(red)
          red = torch.from_numpy(np.asarray(red))
          aberration[i,:,:] = red/torch.max(red)
        return aberration, coeffs
    def __len__(self):
        return len(self.ds)


    
