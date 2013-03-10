import numpy
import cv,cv2
    
def gs(im):
    #return an 8-bit greyscale matrix
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def color(im):
    #return an 8*3-bit color matrix from a 8-bit greyscale
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

def white(im):
    #return a 8-bit white greyscale matrix
    return numpy.zeros(im.shape,numpy.uint8)+255

def invert(im):
    return cv2.bitwise_not(im)

def thresh(im,t,m):
    #t = threshold, m = max for datatype
    ret,im = cv2.threshold(im,t,m,cv2.THRESH_BINARY)
    return im    

def athresh(im,t,n,o):
    #t = threshold, n = neighbors, o = offset
    return cv2.adaptiveThreshold(im,t,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,n,o)
def blur(im,n):
    return cv2.medianBlur(im,n)
