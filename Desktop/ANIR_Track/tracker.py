import numpy
import cv,cv2

class Tracker:
    #train and test contour sets (should each have len=2)
    train_c,test_c = [],[]
    #train and test features: [d,o,(c1),(c2)]
    train_f = [0,0,(0,0),(0,0),0,0]
    test_f  = [0,0,(0,0),(0,0),0,0]
    diff_d,diff_o = 0.0,0.0
    
    def __init__(self,train,test):
        self.train_c = train
        self.test_c  = test

    
    
