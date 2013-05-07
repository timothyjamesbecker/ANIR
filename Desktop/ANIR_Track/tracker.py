import numpy as np
import cv,cv2

class Tracker:
    #train and test contour sets (should each have len=2)
    #train_c,test_c = [],[]
    #train and test features: [d,o,(c1),(c2),a1,a2]
    state = np.float32([0.0,0.0,0.0,0.0]) #this is the unbuffed
    pred  = np.float32([0.0,0.0,0.0,0.0]) #this is the buffed

    #kalman filter
    kf         = cv.CreateKalman(8, 4, 0)
    kf_st      = cv.CreateMat(8, 1, cv.CV_32FC1)
    kf_noise   = cv.CreateMat(8, 1, cv.CV_32FC1)
    kf_measure = cv.CreateMat(4, 1, cv.CV_32FC1)
    kf_const   = cv.fromarray(np.float32([[0,0,1,0,1,0,1,0],
                                          [0,0,0,1,0,1,0,1],
                                          [0,0,0,0,1,0,1,0],
                                          [0,0,0,0,0,1,0,1],
                                          [0,0,0,0,0,0,1,0],
                                          [0,0,0,0,0,0,0,1],
                                          [0,0,0,0,0,0,0,0],
                                          [0,0,0,0,0,0,0,0]]))
    def __init__(self,x1=0.0,y1=0.0,x2=0.0,y2=0.0):
        self.state = np.float32([x1,y1,x2,y2])
        self.pred  = np.float32([0.0,0.0,0.0,0.0])
    
    def set_KF(self,x1,y1,x2,y2):
        # set previous state for prediction
        self.kf.state_pre[0,0]  = np.float32(x1)
        self.kf.state_pre[1,0]  = np.float32(y1)
        self.kf.state_pre[2,0]  = np.float32(x2)
        self.kf.state_pre[3,0]  = np.float32(y2)
        self.kf.state_pre[4,0]  = self.pred[0]
        self.kf.state_pre[5,0]  = self.pred[1]
        self.kf.state_pre[6,0]  = self.pred[2]
        self.kf.state_pre[7,0]  = self.pred[3]
        # set kalman transition matrix
        cv.SetIdentity(self.kf.transition_matrix,
                       cv.RealScalar(1))
        cv.Add(self.kf.transition_matrix, self.kf_const,
               self.kf.transition_matrix)
        # set Kalman Filter
        cv.SetIdentity(self.kf.measurement_matrix,
                       cv.RealScalar(1))
        cv.SetIdentity(self.kf.process_noise_cov,
                       cv.RealScalar(1e-1))
        cv.SetIdentity(self.kf.measurement_noise_cov,
                       cv.RealScalar(1e-5))
        cv.SetIdentity(self.kf.error_cov_post,
                       cv.RealScalar(1))
     
    def predict_KF(self):
        #predict new points
        kf_pred = cv.KalmanPredict(self.kf)
        self.pred  = (kf_pred[0,0],kf_pred[1,0],
                      kf_pred[2,0],kf_pred[3,0])
        
    def correct_KF(self):
        #Kalman Correction
        kf_est = cv.KalmanCorrect(self.kf,self.kf_measure)
        self.state = (kf_est[0,0],kf_est[1,0],
                      kf_est[2,0],kf_est[3,0])

    def update_KF(self,x1,y1,x2,y2):
        #Changing Kalman Measurement
        self.kf_measure[0,0] = x1
        self.kf_measure[1,0] = y1
        self.kf_measure[2,0] = x2
        self.kf_measure[3,0] = y2

    def get_state(self):
        v1 = np.int16(self.state[0])
        v2 = np.int16(self.state[1])
        v3 = np.int16(self.state[2])
        v4 = np.int16(self.state[3])
        return (v1,v2,v3,v4)
    
    def get_pred(self):    
        v1 = np.int16(self.pred[0])
        v2 = np.int16(self.pred[1])
        v3 = np.int16(self.pred[2])
        v4 = np.int16(self.pred[3])
        return (v1,v2,v3,v4)
    
    def run_KF(self,x1,y1,x2,y2):
        self.set_KF(x1,y1,x2,y2)
        self.predict_KF()
        self.correct_KF()
        self.update_KF(x1,y1,x2,y2)
        return self.get_state()
        
    
    
    
    
    
    
