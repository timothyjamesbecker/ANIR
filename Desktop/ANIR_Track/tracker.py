import numpy
import cv,cv2

class Tracker:
    #train and test contour sets (should each have len=2)
    #train_c,test_c = [],[]
    #train and test features: [d,o,(c1),(c2),a1,a2]
    state_pt   = numpy.float32([0.0,0.0]) #this is the unbuffed
    predict_pt = numpy.float32([0.0,0.0]) #this is the buffed

    #one kalman filter
    kalman = cv.CreateKalman(4, 2, 0)
    kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
    kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
    kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

    def __init__(self,x,y):
        #basic setup here
        self.state_pt[0] = numpy.float32(x)
        self.state_pt[1] = numpy.float32(y)
        self.predict_pt[0] = 0.0
        self.predict_pt[1] = 0.0
    
    def set_KF(self,x,y):
        # set previous state for prediction
        self.kalman.state_pre[0,0]  = numpy.float32(x)
        self.kalman.state_pre[1,0]  = numpy.float32(y)
        self.kalman.state_pre[2,0]  = self.predict_pt[0]
        self.kalman.state_pre[3,0]  = self.predict_pt[1]
 
        # set kalman transition matrix
        self.kalman.transition_matrix[0,0] = 1
        self.kalman.transition_matrix[0,1] = 0
        self.kalman.transition_matrix[0,2] = 1
        self.kalman.transition_matrix[0,3] = 0
        self.kalman.transition_matrix[1,0] = 0
        self.kalman.transition_matrix[1,1] = 1
        self.kalman.transition_matrix[1,2] = 0
        self.kalman.transition_matrix[1,3] = 1
        self.kalman.transition_matrix[2,0] = 0
        self.kalman.transition_matrix[2,1] = 0
        self.kalman.transition_matrix[2,2] = 1
        self.kalman.transition_matrix[2,3] = 0
        self.kalman.transition_matrix[3,0] = 0
        self.kalman.transition_matrix[3,1] = 0
        self.kalman.transition_matrix[3,2] = 0
        self.kalman.transition_matrix[3,3] = 1

        # set Kalman Filter
        cv.SetIdentity(self.kalman.measurement_matrix,
                       cv.RealScalar(1))
        cv.SetIdentity(self.kalman.process_noise_cov,
                       cv.RealScalar(1e-1))
        cv.SetIdentity(self.kalman.measurement_noise_cov,
                       cv.RealScalar(1e-5))
        cv.SetIdentity(self.kalman.error_cov_post,
                       cv.RealScalar(1))
     
    def predict_KF(self):
        #predict new points
        self.kalman_prediction = cv.KalmanPredict(self.kalman)
        self.predict_pt  = (self.kalman_prediction[0,0],
                            self.kalman_prediction[1,0])
        return self.predict_pt
        
    def correct_KF(self):
        #Kalman Correction
        self.kalman_estimated = cv.KalmanCorrect(self.kalman,
                                                 self.kalman_measurement)
        self.state_pt = (self.kalman_estimated[0,0],
                         self.kalman_estimated[1,0])
        return self.state_pt

    def update_KF(self,x,y):
        #Changing Kalman Measurement
        self.kalman_measurement[0, 0] = x
        self.kalman_measurement[1, 0] = y
        
        
    
    
    
