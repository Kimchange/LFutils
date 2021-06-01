import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time

def mpi2stack(img,lfSize):
    # u,v,h,w = lfSize[0], lfSize[1], lfSize[2], lfSize[3]
    if img.ndimension() == 4:
        img = img.view(img.shape[0], img.shape[1], lfSize[2],lfSize[0],lfSize[3],lfSize[1]) \
              .permute(0,1,3,5,2,4) \
              .contiguous() \
              .view(img.shape[0], img.shape[1], lfSize[0] * lfSize[1], lfSize[2], lfSize[3])

    return img

def stack2mpi(img, lfSize):
    # u,v,h,w = lfSize[0], lfSize[1], lfSize[2], lfSize[3]
    if img.ndimension() == 5:
        img = img.view(img.shape[0], img.shape[1], lfSize[0], lfSize[1], lfSize[2], lfSize[3]) \
              .permute(0, 1, 4, 2, 5, 3) \
              .contiguous() \
              .view(img.shape[0], img.shape[1], lfSize[0]*lfSize[2], lfSize[1] * lfSize[3])
    return img

def stack2sai_img(img, lfSize):
    # u,v,h,w = lfSize[0], lfSize[1], lfSize[2], lfSize[3]
    if img.ndimension() == 5:
        img = img.view(img.shape[0], img.shape[1], lfSize[0], lfSize[1], lfSize[2], lfSize[3]) \
              .permute(0,1,2,4,3,5) \
              .contiguous() \
              .view(img.shape[0], img.shape[1], lfSize[0]*lfSize[2], lfSize[1] * lfSize[3])
    return img
    

def imread3dtiff(imgPath, ):
    img = Image.open(imgPath)
    while True:
        try:
            img.seek(img.tell()+1)
        except:
            frames = img.tell()+1
            break
    npimg = np.zeros((frames,)+img.size)
    for i in range(frames):
        img.seek(i)
        npimg[i,:,:] = np.array(img)
    return npimg
def imwrite3dtiff(npimg, name):
    # the frames == npimg[0]
    npimg = np.array(npimg,dtype='uint16')
    frames = [Image.fromarray(frame) for frame in npimg]
    # for
    frames[0].save("test.tif", compression="tiff_deflate", save_all=True,
                   append_images=frames[1:])

    return
##img = transforms.ToTensor()(Image.open('LR2_mpi.tif')).squeeze(0)
### (2015,2015) from (155,155,169)
##img = img.view(155,13,155,13).permute(1,3,0,2).contiguous()
# img = transforms.ToTensor()(Image.open('LR2_sai_img.tif')).squeeze(0)
lfSize = [13,13,155,155]
img = torch.tensor(np.array(Image.open('./imresize/LR2_sai_img.tif'))*1.0)
img2 = torch.tensor(imread3dtiff('./imresize/LR2.tif')*1.0) # 
img2 = stack2sai_img(img2.unsqueeze(0).unsqueeze(0),lfSize).squeeze(0).squeeze(0)
print(torch.sum(abs(img-img2)))

# img = Image.open('./imresize/LR2_sai_img.tif')
# img = Image.open('./imresize/LR2.tif')
