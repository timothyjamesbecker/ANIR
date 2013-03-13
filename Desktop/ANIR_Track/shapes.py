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
    #cnt =the contour to find the minimum area bounding rectangle
    #this calculation incorporates a rational angle: theta
    return cv2.minAreaRect(cnt)

def draw_mrect(im,rect,c,t):
    #im - image to draw rectangle on
    #rect (x,y),(w,h),o rectangle with angle
    #c = color to draw
    #t = line thickness
    box = cv2.cv.BoxPoints(rect)
    box = numpy.int0(box)
    cv2.drawContours(im,[box],0,c,t)
    
def brect(cnt):
    #cnt = contour to bound
    #returns a right bound rectangle
    return cv2.boundingRect(cnt)

def draw_brect(im,rect,c,t):
    #im = image to draw ell onto
    #rect= the bounded rectangle
    #c = color to draw
    #t  = thickness of the line to draw
    #side effect: draws on the window
    x,y,w,h = rect
    cv2.rectangle(im,(x,y),(x+w,y+h),c,t)

def bcircle(cnt):
    #cnt = contour
    #returns the tuple (x,y) r which is a
    #point, radius circle form
    (x,y),r = cv2.minEnclosingCircle(cnt)
    return (x,y,r)

def draw_bcircle(im,cir,c,t):
    #im = image to draw ell onto
    #cir= the circle to draw(x,y),r
    #c = color to draw
    #t  = thickness of the line to draw
    #side effect: draws on the window
    x,y,r = int(cir[0]),int(cir[1]),int(cir[2])
    cv2.circle(im,(x,y),r,c,t)

def bellipse(cnt):
    #cnt = one contour, need 5 or more points
    #returns the rectangle that the ellipse fits to (x,y),(w,h),o
    return cv2.fitEllipse(cnt)

def draw_bellipse(im,ell,c,t):
    #im = image to draw ell onto
    #ell= the elipse to draw (x,y,axes,O,startO,EndO)
    #c  = the color to draw
    #t  = thickness of the line to draw
    #side effect: draws on the window
    cv2.ellipse(im, ell, c, t)

def chull(cnt):
    #cnt = one contour
    #computes the convex hull of the contour
    return cv2.convexHull(cnt)

#def cdefects(hull):
    #to do...

def min_max_pts(cnt):
    #cnt = a contour
    #returns the minimal and maximal points L,R,U,D
    L = tuple(cnt[cnt[:,:,0].argmin()][0])
    R = tuple(cnt[cnt[:,:,0].argmax()][0])
    U = tuple(cnt[cnt[:,:,1].argmin()][0])
    D = tuple(cnt[cnt[:,:,1].argmax()][0])
    return L,R,U,D
    

def aspect(rect):
    #rect = a minimum area rectangle
    #returns the apect ratio of the rect, saturating
    #under a divide by zero condition
    (x,y),(w,h),o = rect
    if h <= 0: return sys.float_info.max #saturated div by 0
    else: return w/float(h)

def similiar_aspect(r1,r2,a):
    #r1 = aspect ratio #1 should be the template or control
    #r2 = aspect ratio #2 should be the test
    #returns a boolean condition, used as a binary aspect filter
    if aspect_diff(r1,r2) < a : return True
    else: return False

def aspect_diff(r1,r2):
    #r1 = aspect ratio #1 should be the template or control
    #r2 = aspect ratio #2 should be the test
    #returns the minimum absolute difference bewteen the standard
    #ratios and the 90 degree (w h) swapped set of ratios
    d1 = abs(r1-float(r2))
    d2 = abs(r1-(1/float(r2)))
    return min(d1,d2)

def aspect_dist(r1,r2):
    #r1 = aspect ratio #1 should be the template or control
    #r2 = aspect ratio #2 should be the test
    #returns the minimum geometric distance between the ratios
    #and a 90 degree rotated (w h swapped) set of ratios
    D1 = numpy.sqrt(pow(r1-float(r2)))
    D2 = numpy.sqrt(pow(r1-(1/float(r2)),2))
    return min(D1,D2)

def extent(area,rect):
    #area = contour area of the shape
    #rect = a minimum area rectangle
    #returns the ratio of the contour area over
    #the minimum rect area (so it measures rectangularity)
    (x,y),(w,h),o = rect
    if h <= 0: return sys.float_info.max #saturated div by 0
    else: return area/float(w*h)

def solidity(area,hull):
    #area = contour area of a shape
    #hull = the convex hull calculation from chull
    #returns a ratio that basically describes how solid an
    #object apears, a fork versus a spoon
    if hull <= 0: return sys.float_info.max #saturated div by 0
    else:
        hull_area = cv2.contourArea(hull)
        return area/float(hull_area)

def eql_diameter(area):
    #returns the diameter of a circle that
    #is equal to the contour area of a shape
    return numpy.sqrt(4*area/numpy.pi)

def orientation(rect):
    #rect is a minimum area rectangle
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
    #prevents a divide by zero by drawing the point at the max float
    #which will be off the image space and not viewable
    d = M['m00']
    if(d > 0):
        return M['m10']/d,M['m01']/d
    else: 
        return sys.float_info.max,sys.float_info.max

def match(cnt1, cnt2, mode=3):
    #threshold distance based matching via Humoments
    #cnt1 = the train countour set you want to match to
    #cnt2 = the test contour to match to cnt1
    #mode = the various modes employ alternate algorithms if desired
    if mode==1:
        return cv2.matchShapes(cnt1,cnt2,cv.CV_CONTOURS_MATCH_I1,0) 
    elif mode==2:
        return cv2.matchShapes(cnt1,cnt2,cv.CV_CONTOURS_MATCH_I2,0)
    else:
        return cv2.matchShapes(cnt1,cnt2,cv.CV_CONTOURS_MATCH_I3,0)

def filter_by_area(cont,thr):
    #cont = a set of contours (list)
    #thr = a value used for minimum pixel area
    #returns only the contours that exeed the thr value
    x = []
    for i in range(0,len(cont)):
        a = area(cont[i])
        if area > thr: x.append(i)
    return map(lambda z:cont[z],x)

def filter_by_apsect(cont,t_asp,thr):
    #cont = a set of contours (list)
    #t_asp = the minimum rectangle aspect ratio from the test shape
    #thr = the proximity to t_asp that the contours in the list must
    #satisfy. This provides a small measure of robustness to handle
    #shapes that are slightly occluded
    #returns only those shapes whose minimum rectangle aspect ratios
    #are within the thr value (also tests the rated version to allow
    #for aspect ratio tests where L & W are swapped)
    x = []
    for i in range(0,len(cont)):
        rect = mrect(cont[i])
        asp = aspect(rect)
        if apect_diff(asp,t_asp) < thr:
            x.append(i)
    return map(lambda z:cont[z],x)


    
    




