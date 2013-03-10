from PIL import Image
from pylab import *
from numpy import *
from scipy import *
from imtools import *
import os

proj = '/home/opencv/Desktop/ANIR_Track/'
pict = 'Prelim_SQ_C_Scene_Direct/'
infile = 'D2_0_IR720_TEST1_2x2_SQ.jpg'
outfile = 'Test1.jpg'
# test out PIL
#pil_im = Image.open(proj+pict+infile).convert('L')
#pil_im.crop((100,100,400,400))
#pil_im.save(proj+outfile)

#test out mayplotlib
im = array(Image.open(proj+pict+infile).convert('L'),'f')
im2 = 255 - im #invert grayscale
pil_im = Image.fromarray(im)   #rotates array...
pil_im = pil_im.rotate(180)    #rotate back
imshow(pil_im)
title('Ploting: '+infile) #overlay graphics
print im.shape, im.dtype #diagnostics
print int(im.min()), int(im.max())
show()


