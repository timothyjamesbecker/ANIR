#Timothy Becker CSE5095 ANIR VPU
#this is a template that opens video streams
#and allows testing of Video Processing and Descriptors
#this can serve as a starting place for both the train/test apps
#A new video viewer opens in the upper left hand screen
#and runs until the end of the video file, or the user hits 'esc'

#uses the cv2 bindings (exept where noted)
import cv2
import numpy as np
import os

#ANIR Project Sources
import sources #windowing and video utils
import filters #modifiers
import shapes  #shape processing/masking
import tracker #tracking and smoothing
import perform #performance measures

#base colors
red,green,blue,black     = (0,0,255),(0,255,0),(255,0,0),(255,255,255)
yellow,purple,gray,white = (255,255,0),(200,0,200),(100,100,100),(0,0,0)
#RT control keys:
#esc key exits the RT loop (with a variable pause in seconds)
#number 1 key will write the current frame into the shape1.jpg
#effectivly allowing the RT system to key new shapes
KEY_ESC,KEY_W,KEY_1,KEY_2,KEY_3 = 27,119,49,50,51

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
#KF based tracker ****************************************
#has a circular buffer set to last 15 frames
target = tracker.Tracker(train_c) #initialize to point (0,0)
#KF based tracker ****************************************

#new performance CPU measurement object
p = perform.CPU()

#new window to view video streams
source.win_start('video_input')

#captures video files or default webcam = 0
source.vin(640,480,'')  #leaving the path = '' sets to webcam
n, m = 1,10 #n=number of max frames, m= a divsor for output
RT,IM,frame,diag = True,True,0,white
while RT and (frame < n): #Real-Time Loop=> 'esc' key turns off
    p.start()#-----------performance:msec-------CPU

    #compute loop:::::::::::::::::::::::::::::::::::::::::::
    im = source.read() #get one frame
    #[1]-----gray scale conversion--------------[1]
    im  = filters.gs(im)
    imc = filters.color(im) #used for colors 
    #[1]-----thresholding-----------------------[1]
    im2 = filters.thresh(im,220,255)
    #[1]-----smoothing--------------------------[1]
    im2 = filters.blur(im2,5)
    #[2]-----shape collection-------------------[2]
    imw = im2.copy() #make a deep copy=>cont is destructive
    cont = shapes.contours(im2) #im2 destroyed, now use im4
    ims = filters.white(filters.color(im)) #white color const
    #[2]-----contour filtering-------------------[2]
    cont = shapes.filter_by_area(cont,1000) #remove small shapes fast
    #[3]-----find best shapes--------------------[3]
    s_f = shapes.match_features(train_c,cont,0.1) #len(s_f) <= 2
    #[4]-----kalman-filter-----------------------[4]
    v = target.follow(s_f,cont) #returns smoothed c1,c2,scale,angle
    #[4]-----kalman-filter-----------------------[4]
    px,py = np.int32(v[0]),np.int32(v[1])
    qx,qy = np.int32(v[2]),np.int32(v[3])
    values = (v[4],v[5]) #these are the smoothed scale,angle
    #compute loop:::::::::::::::::::::::::::::::::::::::::::

    p.stop()#-----------performance:msec-------CPU
    #[5]------drawing----------------------------[5]
    if(IM): draw,diag = imc,black #toggles viewing mode
    else:   draw,diag = ims,white #between input+overlay && overlay
    shapes.draw_contours(draw,cont,green,2)  #Filtered Contours
    shapes.draw_point(draw,(px,py),purple,4) #KF Left symbol
    shapes.draw_point(draw,(qx,qy),purple,4) #KF Right symbol
    shapes.draw_point(draw,s_f[0],red,16)    #Left centroid
    shapes.draw_point(draw,s_f[1],red,16)    #Right centroid
    shapes.draw_line(draw,s_f[1],s_f[0],red,1)
    shapes.draw_line(draw,(qx,qy),(px,py),purple,1)
    #[6]-----diagnostic-text---------------------[6]
    draw = filters.flip(draw,1) #mirror image, comment to turn off
    source.win_diag(draw,frame,p.diff(),len(cont),diag)
    source.win_message(draw,str(values),diag)
    #Sentel Loop Control Logic============================
    k = cv2.waitKey(30)            #wait period
    if k == KEY_ESC: break        #'esc' key exit
    if k == KEY_W: IM = not IM    #toggle viewimg mode
    if k == KEY_1:                #'1' key save to shape1
        source.write(imw,train+'shape1.jpg')
        source.win_message(draw,'Shape 1 Written',diag)
    if k == KEY_2:                #'2' key save to shape1
        source.write(imw,train+'shape2.jpg')
        source.win_message(draw,'Shape 2 Written',diag)
    if k == KEY_3:                #'3' key save to shape1
        source.write(imw,train+'shape2.jpg')
        source.win_message(draw,'Shape 3 Written',diag)
        
    source.win_print('video_input',draw) #display output
    frame = (frame+1)%31          #update the frame counter
    if(source.mode == 0): n+=1    #take out frame + n for video
    #Sentel Loop Control Logic============================
    
source.win_stop('video_input',1) #close up window thread, pause 1 sec
