#Timothy Becker CSE5095 ANIR VPU
#this is a template that opens video streams
#and allows testing of Video Processing and Descriptors
#this can serve as a starting place for both the train/test apps
#A new video viewer opens in the upper left hand screen
#and runs until the end of the video file, or the user hits 'esc'

#uses the cv2 bindings (exept where noted)
import cv2
import numpy
import os

#ANIR Project Sources
import sources #windowing and video utils
import filters #modifiers
import shapes  #shape processing/masking
import tracker #tracking and smoothing
import perform #performance measures

#base colors
red,green,blue,black  = (0,0,255),(0,255,0),(255,0,0),(255,255,255)
#RT control keys:
#esc key exits the RT loop (with a variable pause in seconds)
#number 1 key will write the current frame into the shape1.jpg
#effectivly allowing the RT system to key new shapes
KEY_ESC,KEY_1,KEY_2,KEY_3 = 27,49,50,51

#set video file input and output paths ####################
path  = '/home/opencv/Desktop/ANIR_Track/' #project path
train = path+'Train/' 
out_f = 'OpenCV_Test.mov' #output stream to make demos with
vin_f = 'Direct_IR950_2x4_TRI.mov' #training video, will update
#this section will be changed to work with args############

#new video source object (makes windows, opens videos, camera,ect)
source = sources.Sources()
train_i = source.load(train+'shape1.jpg') #load as GS image
train_c = shapes.contours(train_i)#get the image contours

#new performance CPU measurement object
p = perform.CPU()

#new window to view video streams
source.win_start('video_input')

#captures video files or default webcam = 0
source.vin(640,480,'')  #leaving the path = '' sets to webcam
n, m = 1,10 #n=number of max frames, m= a divsor for output
RT,frame,values,asp = True,0,0,0

#kalman filter on (x,y) point ****************************
target = tracker.Tracker(0.0,0.0) #initialize to point (0,0)
#kalman filter on (x,y) point ****************************

while RT and (frame < n): #Real-Time Loop=> 'esc' key turns off
    p.start()#----------------performance start---CPU

    #compute loop:::::::::::::::::::::::::::::::::::::::::::
    im = source.read() #get one frame
    #[1]-----gray scale conversion--------------[1]
    im  = filters.gs(im)
    imc = filters.color(im) #used to colors 
    #[1]-----thresholding-----------------------[1]
    im2 = filters.thresh(im,220,255)
    #[1]-----smoothing--------------------------[1]
    im2 = filters.blur(im2,5)
    #[2]-----shape collection-------------------[2]
    im4 = im2.copy() #make a deep copy=>cont is destructive
    cont = shapes.contours(im2) #im2 destroyed now use im4
    im3 = filters.white(filters.color(im)) #white color const
    #[2]-----contour filtering-------------------[2]
    cont = shapes.filter_by_area(cont,1000) #remove small shapes
    #[3]-----find best shapes--------------------[3]
    s_f = shapes.features(train_c,cont,0.1);
    shapes.draw_contours(im3,cont,green,2) #all contours
    #[4]-----kalman-filter-----------------------[4]
    target.set_KF(s_f[2][0],s_f[2][1])
    p_1 = target.predict_KF()
    px,py = int(p_1[0]),int(p_1[0])
    target.correct_KF()
    target.update_KF(s_f[2][0],s_f[2][1])
    #[4]-----kalman-filter-----------------------[4]
    shapes.draw_point(im3,(px,py),red,2) #centroids of matched
    shapes.draw_point(im3,s_f[3],red,2) #centroids of matched
    values = s_f[0],s_f[1]
    #compute loop:::::::::::::::::::::::::::::::::::::::::::

    p.stop()#--------------------performance end---CPU
    #[n]-----diagnostic-text--------------------[n]
    im3 = filters.flip(im3,1) #mirror image
    source.win_diag(im3,frame,p.diff(),len(cont))
    source.win_message(im3,str(values))

    #Sentel Loop Control Logic============================
    k = cv2.waitKey(30)           #wait period
    if k == KEY_ESC: break        #'esc' key exit
    if k == KEY_1:                #'1' key save to shape1
        source.write(im4,train+'shape1.jpg')
        source.win_message(im3,'Shape 1 Written')
    if k == KEY_2:                #'2' key save to shape1
        source.write(im4,train+'shape2.jpg')
        source.win_message(im3,'Shape 2 Written')
    if k == KEY_3:                #'3' key save to shape1
        source.write(im4,train+'shape2.jpg')
        source.win_message(im3,'Shape 3 Written')
        
    source.win_print('video_input',im3) #display output
    frame+=1 #update the frame counter (for video only)
    if(source.mode == 0): n+=1    #take out frame + n for video
    #Sentel Loop Control Logic============================
    
source.win_stop('video_input',1) #close up window thread, pause 1 sec
