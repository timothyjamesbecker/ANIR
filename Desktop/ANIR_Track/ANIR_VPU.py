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
vin_f = 'Direct_IR950_2x4_TRI.mov' #training video
#this section will be changed to work with args############

#new video source object (makes windows, opens videos, camera,ect)
source = sources.Sources()
train_s = train+'shape1.jpg'   #shape you want to match
train_i = source.load(train_s) #load it as an image
train_c,train_h = shapes.contours(train_i) #get the image contours

#new performance CPU measurement object
p = perform.CPU()

#new window to view video streams
source.win_start('video_input')

#captures video files or default webcam = 0
#source.vin()   #open camera
source.vin(640,480,'')  #leaving the path = '' sets to webcam
n, m = 1,10 #n=number of max frames, m= a divsor for output
RT,frame,value,asp = True,0,0,0
while RT and (frame < n): #Real-Time Loop=> 'esc' key turns off
    p.start()#----------------performance start---CPU

    #compute loop:::::::::::::::::::::::::::::::::::::::::::
    im = source.read() #get one frame
    #[1]-----gray scale conversion--------------[1]
    im  = filters.gs(im)
    imc = filters.color(im)
    #[2]-----thresholding-----------------------[2]
    im2 = filters.thresh(im,220,255)
    #[3]-----smoothing--------------------------[3]
    im2 = filters.blur(im2,5)
    #[4]-----shape collection-------------------[4]
    im4 = im2.copy() #make a deep copy=>cont is destructive
    cont,hier = shapes.contours(im2) #im2 is the main input
    im3 = filters.white(filters.color(im)) #white color const
    cont = shapes.filter_by_area(cont,20)  #remove small area shapes
    #process each contour set in the list cont:
    for i in range(0,len(cont)):
        #[5]-----shape_processing---------------[5]
        rect = shapes.mrect(cont[i]) #get a mrect
        asp  = shapes.aspect(rect)   #get aspect of mrect
        M = shapes.moments(cont[i])  #contour moments
        #CH = shapes.chull(cont[0])  #convex hull
        #H = shapes.hu_moments(M)    #Hu moments
        x,y = shapes.centroid(M)     #centroid calculation
        value = shapes.match(cont[i],cont[i])
        #[7]-----draw-shapes--------------------[7]
        shapes.draw_mrect(im3,rect,blue,2)
        shapes.draw_point(im3,(x,y),red,5)
        
    shapes.draw_contours(im3,cont,green,2)
    #compute loop:::::::::::::::::::::::::::::::::::::::::::

    p.stop()#--------------------performance end---CPU
    #[n]-----diagnostic-text--------------------[n]
    source.win_diag(im3,frame,p.diff(),len(cont))
    source.win_message(im3,str(asp))

    #Sentel Loop Control Logic
    k = cv2.waitKey(30)           #wait period
    if k == KEY_ESC: break        #'esc' key exit
    if k == KEY_1:                #'1' key save to shape1
        source.write(im4,test_s)
        source.win_message(im3,'Shape 1 Written')
    if k == KEY_2:                #'2' key save to shape1
        source.write(im4,test_s)
        source.win_message(im3,'Shape 2 Written')
    if k == KEY_3:                #'3' key save to shape1
        source.write(im4,test_s)
        source.win_message(im3,'Shape 3 Written')
        
    source.win_print('video_input',im3) #display output
    frame+=1 #update the frame counter (for video only)
    if(source.mode == 0): n+=1    #take out frame + n for video
    
source.win_stop('video_input',1) #close up window thread pause 1 sec
