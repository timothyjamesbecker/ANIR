from PIL import Image
from numpy import *
import os

def get_imlist(path):
    L = []
    for f in os.listdir(path):
        if not f.startswith('.') and f.endswith('.jpg'):
            L.append(path+f)
    return L

def imflip(im,a):
    return im.rotate(a)

def imresize(im,sz):
    #resize an image array using the PIL
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def histeq(im, nbr_bins=256):
    #histogram equalization of a grayscale image
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum()     #cumulative dist funct
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

def compute_average(imlist):
    #compute the average of a list of images
    averageim = array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print imname + '...skipped'
    averageim /= len(imlist)
    #return average as uint8
    return array(averageim, 'uint8')

