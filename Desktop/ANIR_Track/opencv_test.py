from PIL import Image
from pylab import *
from numpy import *
from scipy import *
import os
import cv
import imtools

fullpath = '/home/opencv/Desktop/ANIR_Track/'
outfile = 'OpenCV_Test.jpg'
pics = 'Prelim_SQ_C_Scene_Direct/'
video = 'Direct_IR950_2x4_TRI.mov'
#dig through the full path direct and find jpegs
imlist = imtools.get_imlist(fullpath+pics)
for i in imlist[1:]: print i #print out names to verify

#put up a window
cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)
#load a jpeg
#image=cv.LoadImage(imlist[1], cv.CV_LOAD_IMAGE_COLOR)
g_capture = cv.CreateFileCapture(fullpath+video)
image = cv.QueryFrame(g_capture)
font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX,1,1,0,3,8)
x = 500
y = 400
cv.PutText(image,'Hellow World!!!', (x,y),font,255)
cv.ShowImage('a_window',image)
cv.SaveImage(fullpath+outfile,image)
s = str(x)
print s
