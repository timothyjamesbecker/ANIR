import numpy as np
import cv,cv2

class Tracker:
    #train and test contour sets (should each have len=2)
    #train_c,test_c = [],[]
    #train and test features: [d,o,(c1),(c2),a1,a2]
    state = np.float32([0.0,0.0]) #this is the unbuffed
    pred  = np.float32([0.0,0.0]) #this is the buffed

    #one kalman filter
    kf         = cv.CreateKalman(4, 2, 0)
    kf_st      = cv.CreateMat(4, 1, cv.CV_32FC1)
    kf_noise   = cv.CreateMat(4, 1, cv.CV_32FC1)
    kf_measure = cv.CreateMat(2, 1, cv.CV_32FC1)

    def __init__(self,x=0.0,y=0.0):
        self.state = np.float32([x,y])
        self.pred  = np.float32([0.0,0.0])
    
    def set_KF(self,x,y):
        # set previous state for prediction
        self.kf.state_pre[0,0]  = np.float32(x)
        self.kf.state_pre[1,0]  = np.float32(y)
        self.kf.state_pre[2,0]  = self.pred[0]
        self.kf.state_pre[3,0]  = self.pred[1]
        # set kalman transition matrix
        #self.kf.transition_matrix = trans
        #self.kf.transition_matrix[0,0] = 1
        #self.kf.transition_matrix[0,1] = 0
        self.kf.transition_matrix[0,2] = 1
        #self.kf.transition_matrix[0,3] = 0
        #self.kf.transition_matrix[1,0] = 0
        #self.kf.transition_matrix[1,1] = 1
        #self.kf.transition_matrix[1,2] = 0
        self.kf.transition_matrix[1,3] = 1
        #self.kf.transition_matrix[2,0] = 0
        #self.kf.transition_matrix[2,1] = 0
        #self.kf.transition_matrix[2,2] = 1
        #self.kf.transition_matrix[2,3] = 0
        #self.kf.transition_matrix[3,0] = 0
        #self.kf.transition_matrix[3,1] = 0
        #self.kf.transition_matrix[3,2] = 0
        #self.kf.transition_matrix[3,3] = 1
        # set Kalman Filter
        cv.SetIdentity(self.kf.measurement_matrix,
                       cv.RealScalar(1))
        cv.SetIdentity(self.kf.process_noise_cov,
                       cv.RealScalar(1e-1))
        cv.SetIdentity(self.kf.measurement_noise_cov,
                       cv.RealScalar(1e-5))
        cv.SetIdentity(self.kf.error_cov_post,
                       cv.RealScalar(1e-3))
     
    def predict_KF(self):
        #predict new points
        kf_pred = cv.KalmanPredict(self.kf)
        self.pred  = (kf_pred[0,0],kf_pred[1,0])
        return self.pred
        
    def correct_KF(self):
        #Kalman Correction
        kf_est = cv.KalmanCorrect(self.kf,self.kf_measure)
        self.state = (kf_est[0,0],kf_est[1,0])
        return self.state

    def update_KF(self,x,y):
        #Changing Kalman Measurement
        self.kf_measure[0, 0] = x
        self.kf_measure[1, 0] = y

    def run_KF(self,x,y):
        self.set_KF(x,y)
        self.predict_KF()
        z = self.correct_KF()
        self.update_KF(x,y)
        return z
        
    
    
    
