#Timothy Becker CSE5095 ANIR VPU
#this is a template that opens video streams
#and allows testing of Video Processing and Descriptors
#this can serve as a starting place for both the train/test apps
#A new video viewer opens in the upper right hand screen
#and runs until the end of the video file, or the user hit 'esc'

#uses the cv2 bindings (exept where noted)
import cv2
import numpy
import os

#ANIR Project Sources
import sources #windowing and video utils
import filters #modifiers
import shapes  #shape processing/masking
import perform #performance measures

#set video file input and output paths
path  = '/home/opencv/Desktop/ANIR_Track/'
out_f = 'OpenCV_Test.mov'
vin_f = 'Direct_IR950_2x4_TRI.mov'
pin_f = 'Prelim_SQ_C_Scene_Direct/D2_0_1A_TEST1_2x2_SQ.jpg'

#base colors
red   = (0,0,255)
green = (0,255,0)
blue  = (255,0,0)

#new video source object (makes windows, opens videos, camera,ect)
source = sources.Sources()
#new performance CPU measurement object
p = perform.CPU()

#new window to view video streams
source.win_start('video_input')

#captures video files or default webcam = 0
#source.vin()   #open camera
source.vin(640,480,'')  #video
n, m = 1,10

frame = 0
RT = True
while RT and (frame < n): #Real-Time Loop=> 'esc' key turns off: c==27
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
    cont,hier = shapes.contours(im2) #im2 is the main input
    im3 = filters.white(filters.color(im)) #white color const
    if len(cont) > 0:#need at least one contour to draw
        #[5]-----shape_processing---------------[5]
        rect = shapes.brect(cont[0]) #bounding rectangle
        M = shapes.moments(cont[0])  #contour moments
        CH = shapes.chull(cont[0]) #convex hull
        #H = shapes.hu_moments(M)     #Hu moments
        x,y = shapes.centroid(M)     #centroid calculation
        #[7]-----draw-shapes--------------------[7]
        shapes.draw_brect(imc,rect,blue,2)
        shapes.draw_point(imc,(x,y),red,5)
        
    shapes.draw_contours(imc,cont,green,2)
    #compute loop:::::::::::::::::::::::::::::::::::::::::::

    p.stop()#--------------------performance end---CPU
    #[n]-----diagnostic-text--------------------[n]
    source.win_diag(imc,frame,p.diff(),len(cont))
    source.win_print('video_input',imc) #display output
    
    c = cv2.waitKey(30)           #wait period
    if c == 27: break             #'esc' key exit
    if c == 49:
    if c == 50:
    if c == 51:
    frame+=1
    if(source.mode == 0): n+=1    #take out fram + n for video
    
source.win_stop('video_input',5)
