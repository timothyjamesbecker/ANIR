import numpy
import cv,cv2
import filters

class Sources:
    w_pos = (0,0) #window position
    f_type = cv2.FONT_HERSHEY_SIMPLEX #font type
    capture = 0 #holds the input device video stream pointer
    frames = 0  #max frames for video input
    mode = 0    #0 = webcam, 1 = video/picture
    
    def win_start(self,name):
        #setup a linux window fork and new window and move it
        cv2.startWindowThread()
        cv2.namedWindow(name,cv2.CV_WINDOW_AUTOSIZE)
        cv.MoveWindow(name,self.w_pos[0],self.w_pos[1])
    
    def win_print(self,name,im):
        cv2.imshow(name,im) #display output
    
    def win_diag(self,im,f,p,o):
        #f = frame#, p = CPU in msec, o = object#
        cv2.putText(im,str(f),(10,30),self.f_type,1,0,1,1)
        cv2.putText(im,str(p),(10,50),self.f_type,0.5,0,1,1)
        cv2.putText(im,str(o),(10,70),self.f_type,0.5,0,1,1)

    def win_message(self,im,t):
        #output a temp message
        cv2.putText(im,t,(10,90),self.f_type,0.5,0,1,1)

    def win_stop(self,name,hold):
        #closes the capture device, stops threads and then window
        del(self.capture) #release camera/input video stream
        cv2.waitKey(hold*1000) #pause for a sec
        cv2.destroyWindow(name) #take down the window
        cv2.destroyAllWindows() #finsh last threads

    def zeros(self,im):
        #convert to gray scale first
        return numpy.zeros(im.shape,numpy.unint8)

    def vin(self,width=640,height=480,path=''):
        #main video input dabstraction
        #width = pixels of input width
        #height= pixels of input height
        #path = string to a video input file or picture
        #if left = '' will use the webcam or (0) device
        if path == '':
            self.capture = cv2.VideoCapture(0)
            self.mode = 0
        else:
            self.capture = cv2.VideoCapture(path)
            self.frames  = self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT)/3
            self.mode = 1
        self.capture.set(cv.CV_CAP_PROP_FRAME_WIDTH,width)
        self.capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT,height)
            
    def read(self):
        #reads in the next frame from a webcam or video file
        ret,im = self.capture.read()
        return im

    def load(self,path):
        #reads in a picture as a image
        im = cv2.imread(path)
        img = filters.gs(im)
        return filters.thresh(img,200,255)
        
    
    def write(self,im,path):
        #used to save images as picture files
        cv2.imwrite(path,im)
    


    
