import numpy as np
import cv2
import cv
import sys

"""
  Function Name:  getContours
  Purpose:  extracts grayscale contours from any provided image ( color, grayscale, or binary )
  Returns:  a list containing all contours in the provided image
  Params:
            img - An image to search for contours
"""
def getContours( img ):

    img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    
    img_ret  , img_thresh           = cv2.threshold( img_gray, 200, 255, 0 )
    img_cont , img_hierarchy        = cv2.findContours( img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

    return img_cont

"""
  Funtion Name: filterByContours
  Purpose:  This function is intended to take a dual symbol template and a list of contours to filter
            based on the objects in the template.
  Returns:  A list for each symbol containing the contours that matched regardless of rotation
            or scale.
  Params: 
           templates - A template img containing two symbols
           contoursToSrch - Any list of contours to search for the template symbols
"""

def filterByContours( templates, contoursToSrch, accuracy ):

    if len( templates ) != 2 :
      raise Exception, "Template must contain exactly two symbols"

    if len( contoursToSrch ) <= 0 :
        raise Exception, "Search space is empty"

    if not isinstance( accuracy, float ):
        raise Exception, "accuracy must be of type 'float'."

    if accuracy <= 0.0 or accuracy >= 1.0 :
        raise Exception, "accuracy must be in range [0.0, 1.0], non-inclusive."
    
    leftSymMatches = list()
    rightSymMatches = list()

    for cont in contoursToSrch:
        if( cv2.matchShapes( templates[0], cont, cv.CV_CONTOURS_MATCH_I3, 0 ) < accuracy ):

            leftSymMatches.append( cont )
            contoursToSrch.remove( cont )

            """  # To view visualization of object1 selection uncomment this section
            cv2.drawContours( colorSearch, [cont], -1, (0,255,0), 4)
            cv2.imshow( 'contours', colorSearch )
            cv2.waitKey(0)
            """

    for cont in contoursToSrch:
        if( cv2.matchShapes( templates[1], cont, cv.CV_CONTOURS_MATCH_I3, 0 ) < accuracy ):

            rightSymMatches.append( cont )
        
            """  # To view visualization of object1 selection uncomment this section
            cv2.drawContours( colorSearch, [cont], -1, (0,255,0), 4)
            cv2.imshow( 'contours', colorSearch )
            cv2.waitKey(0)
            """

    return leftSymMatches, rightSymMatches


"""
  Function Name:  findSymbolApprox
  Purpose:  given a contour and an expected accuracy the function returns a minimal approximation
            of the given object.
  Returns:  Set of points representing the minimal approximation of the object given the accuracy constraint.
  Params:
          contour  - The contour to be simplified
          accuracy - The level of desired accuracy.  Range is [0.0, 1.0], non-inclusive. 
"""

def findSymbolApprox( contour, accuracy ):
    if not isinstance( accuracy, float )  :
        raise Exception, "accuracy must be a float"
    if accuracy >= 1.0 or accuracy <= 0.0:
        raise Exception, "accuracy must be in range [0.0, 1.0], non-inclusive."

    perimeter = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour, accuracy*perimeter, True)
    
    return approx

    """ The following code could be used to extract the approximation of many contours
        and visualize the results.

    for cont in contours:
        perimeter = cv2.arcLength(cont,True)
        approx = cv2.approxPolyDP(cont, accuracy*perimeter, True)
        for pnt in approx:
            [x] = pnt
            cv2.circle( colorSearch, (x[0],x[1]), 2, (255,0,0) )
            cv2.imshow( 'contours', colorSearch )
            cv2.waitKey(0)
    """

# The below code will allow for direct testing of the functions contained in this file
# this should be replaced with a unit test.

"""
if len(sys.argv) < 3:
    print 'usage %s template imgToSearch' % sys.argv[0]
    sys.exit(1)

template_path    = sys.argv[1]
imgToSearch      = sys.argv[2]

colorTemplate    = cv2.imread( template_path )
colorSearch      = cv2.imread( imgToSearch )

templates = getContours( colorTemplate )
srchSpace = getContours( colorSearch )

left, right = filterByContours( templates, srchSpace, 0.1 )
leftApprox  = findSymbolApprox( left[0], 0.1 )
rightApprox = findSymbolApprox( right[0], 0.1 )

print len( leftApprox )
print len( findSymbolApprox( templates[0], 0.1 ) )

temp = findSymbolApprox( templates[0], 0.1 ) 

print cv2.estimateRigidTransform( temp, leftApprox, False )
"""
