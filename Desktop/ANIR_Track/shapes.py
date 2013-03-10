import sys
import numpy
import cv,cv2

def contours(im):
    #use binary thresholding first
    #items you want should be white with background in black
    #im = image to find contours
    #returns a list of contours (lists)
    return cv2.findContours(im,cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
def approx_poly(cnt,e):
    #cnt = an input contour
    #e = percentage value of arc length used to define approximation
    #returns an approximation of cnt
    return cv2.approxPolyDP(cnt,e*arc_len(cnt),True)

def draw_contours(im,cont,c,t):
    #im = the output image to be drawn onto
    #cont = a set of contours, can print out specific ones
    #c = color
    #t = line thickness
    cv2.drawContours(im,cont,-1,c,t)

def draw_contour(im,cnt,c,t):
    #im = output image to be drawn to
    #cnt = the contour to be drawn onto im
    #c = the color
    #t = the thinkness of the line to draw
    mask = numpy.zeros(im.shape,numpy.uint8)
    cv2.drawContours(im,[cnt],0,c,t)

def draw_point(im,(x,y),c,t):
    #im = output image to be drawn to
    #(x,y) = point in 2D space
    #c = color
    #t = line thickness
    cv2.circle(im,(int(x),int(y)),1,c,t)

def mrect(cnt):
    #cnt =
    return cv2.minAreaRect(cnt)

def draw_mrect(im,rect,c,t):
    box = cv2.cv.BoxPoints(rect)
    box = numpy.int0(box)
    cv2.drawContours(im,[box],0,c,t)
    
def brect(cnt):
    return cv2.boundingRect(cnt)

def draw_brect(im,rect,c,t):
    x,y,w,h = rect
    cv2.rectangle(im,(x,y),(x+w,y+h),c,t)

def bcircle(cnt):
    (x,y),r = cv2.minEnclosingCircle(cnt)
    return (x,y,r)

def draw_bcircle(im,cir,c,t):
    x,y,r = int(cir[0]),int(cir[1]),int(cir[2])
    cv2.circle(im,(x,y),r,c,t)

def bellipse(cnt):
    #cnt = one contour, need 5 or more points
    return cv2.fitEllipse(cnt)

def draw_bellipse(im,ell,c,t):
    #im = image to draw ell onto
    #ell= the elipse to draw (x,y,axes,O,startO,EndO)
    #c  = the color to draw
    #t  = thickness of the line to draw
    cv2.ellipse(im, ell, c, t)

def chull(cnt):
    return cv2.convexHull(cnt)

#def cdefects(hull):

def aspect(rect):
    (x,y),(w,h),o = rect
    if h <= 0: return sys.float_info.max #saturated div by 0
    else: return w/float(h)

def similiar_aspect(r1,r2,a):
    if aspect_diff(r1,r2) < a : return True
    else: return False

def aspect_diff(r1,r2):
    d1 = abs(r1-float(r2))
    d2 = abs(r1-(1/float(r2)))
    return min(d1,d2)

def aspect_dist(r1,r2):
    D1 = numpy.sqrt(pow(r1-float(r2)))
    D2 = numpy.sqrt(pow(r1-(1/float(r2)),2))
    return min(D1,D2)

def extent(area,rect):
    (x,y),(w,h),o = rect
    if h <= 0: return sys.float_info.max #saturated div by 0
    else: return area/float(w*h)

def solidity(area,hull):
    if hull <= 0: return sys.float_info.max #saturated div by 0
    else:
        hull_area = cv2.contourArea(hull)
        return area/float(hull_area)

def eql_diameter(area):
    return numpy.sqrt(4*area/numpy.pi)

def orientation(rect):
    (x,y),(w,h),o = rect
    return o

def moments(cnt):
    #cnt = the input countour
    #returns the dictionary of image moments
    return cv2.moments(cnt)

def hu_moments(M):
    return cv2.HuMoments(M)

def area(cnt):
    #cnt - an input contour
    #returns the pixel area
    return cv2.contourArea(cnt)

def arc_len(cnt):
    #cnt - an input contour
    #returns rgw arc length of the contour
    return cv2.arcLength(cnt,True)

def centroid(M):
    #take the contour moments M and return the center point (x,y)
    #prevents a divide by zero by drawing the point at (-1,-1)
    #which will be off the image space and not viewable
    d = M['m00']
    if(d > 0):
        return M['m10']/d,M['m01']/d
    else:
        return -1,-1

def match(cnt1, cnt2, mode=3):
    if mode==1:
        return cv2.matchShapes(cnt1,cnt2,cv.CV_CONTOURS_MATCH_I1,0) 
    elif mode==2:
        return cv2.matchShapes(cnt1,cnt2,cv.CV_CONTOURS_MATCH_I2,0)
    else:
        return cv2.matchShapes(cnt1,cnt2,cv.CV_CONTOURS_MATCH_I3,0)

def filter_by_area(cont,thr):
    x = []
    for i in range(0,len(cont)):
        a = area(cont[i])
        if area > thr: x.append(i)
    return map(lambda z:cont[z],x)

def filter_by_apsect(cont,t_asp,thr):
    x = []
    for i in range(0,len(cont)):
        asp = 


    
    




