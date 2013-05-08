import numpy as np
import cv,cv2
import shapes

class Tracker:
    #train and test features: [(c1),(c2),a1,a2,b]
    #c1 = left symbol centroid
    #c2 = right symbol centroid
    #a1 = left contour area
    #a2 = right contour area
    #b  = Boolean flag for a match
    
    #train and test contour sets (should each have len=2)
    ref_c, targ_c = [],[]   #contours of reference and target
             #c1, c2, a1, a2 of reference
    ref_f  = [(0.0,0.0),(0.0,0.0),0.0,0.0,False]
             #c1, c2, a1, a2 of target
    targ_f = [(0.0,0.0),(0.0,0.0),0.0,0.0,False]
    ref_v    = np.float32([0.0,0.0,0.0]) #mag and unit vector
    targ_v   = np.float32([0.0,0.0,0.0]) #mag and unit vector
    #history slots to use as memory
    hist = [False]*8        #8 frame history ~0.25 sec
    h_len = len(hist)       #fixed history buffer size
    h_w  = 0                #writing index into circular buffer
    h_r  = 1                #reading index into circular buffer
    
    #kalman filter variables
    state = np.float32([0.0,0.0,0.0,0.0]) #this is the unbuffed
    pred  = np.float32([0.0,0.0,0.0,0.0]) #this is the buffed
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
    
    def __init__(self,train_c):
        #setup the reference variables---------
        #load reference contours
        self.ref_c = train_c
        #use two copies to get the fetaure pair
        self.ref_f = shapes.match_features(train_c,train_c,0.1)
        #get ref distance and angle
        self.ref_v = shapes.unit_vect(self.ref_f[0][0],
                                      self.ref_f[0][1],
                                      self.ref_f[1][0],
                                      self.ref_f[1][1])
        #setup the reference variables---------
        #set KF<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.state = np.float32([self.ref_f[0][0],
                                 self.ref_f[0][1],
                                 self.ref_f[1][0],
                                 self.ref_f[1][1]])
        self.pred  = np.float32([0.0,0.0,0.0,0.0])

    def get_ref_v(self):
        return ref_v
        
    def hist_write(self,e):
        #updates the circluating history buffer
        self.h_w = (self.h_w+1)%self.h_len
        self.h_r = (self.h_r+1)%self.h_len
        #insert e, into the circlating buffer at h_i
        self.hist[self.h_w] = e

    def hist_read(self):
        #always draw from read index
        return self.hist[self.h_r]

    def get_closest_centroids(self,ref_cent,cont,alpha):
        #return the closest centroids from cont to the ref_cent pair
        #get features from the cont list
        l = shapes.features(cont)
        #get closest matches for the pair
        sim1,sim2 = alpha,alpha #implicit threshold values
        i1,i2 = -1,-1 #indecies of the best matches
        for i in range(0,len(l)):
            c1,c2 = l[i][0], l[i][0]
            d1 = shapes.distance(ref_cent[0][0],ref_cent[0][1],
                                 c1[0],c1[1])
            d2 = shapes.distance(ref_cent[1][0],ref_cent[1][1],
                                 c2[0],c2[1])
            if(d1 < sim1): sim1,i1 = d1,i
            if(d2 < sim2): sim2,i2 = d2,i
        if i1!=-1 and i2!=-1:
            return [l[i1][0], l[i2][0]]
        else:
            return [(0.0,0.0),(0.0,0.0)]
        
    def targ_update(self):
        self.targ_v = shapes.unit_vect(self.state[0],
                                       self.state[1],
                                       self.state[2],
                                       self.state[3])    
    
    def targ_scale(self):
        return self.targ_v[0]/self.ref_v[0]
        
    def targ_angle(self):
        return shapes.angle(self.ref_v[1],self.ref_v[2],
                            self.targ_v[1],self.targ_v[2])
    
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
        v1 = np.int32(self.state[0])
        v2 = np.int32(self.state[1])
        v3 = np.int32(self.state[2])
        v4 = np.int32(self.state[3])
        return (v1,v2,v3,v4)
    
    def get_pred(self):    
        v1 = np.int32(self.pred[0])
        v2 = np.int32(self.pred[1])
        v3 = np.int32(self.pred[2])
        v4 = np.int32(self.pred[3])
        return (v1,v2,v3,v4)
    
    def follow(self,s_f,cont):
        #s_f = [(0,0),(0,0),0,0,True|False]
        #s_f[4]==T => you have a match
        if(s_f[4]): #use value for KF and insert into hist
            self.set_KF(s_f[0][0],s_f[0][1],
                        s_f[1][0],s_f[1][1])
            self.predict_KF()
            self.correct_KF()
            self.update_KF(s_f[0][0],s_f[0][1],
                           s_f[1][0],s_f[1][1])
            self.hist_write(True) #this will have a True flag
                           
        else: #use hist values to update buffer (until run out)
            x1 = self.state[0] #last predicted values
            y1 = self.state[1]
            x2 = self.state[2]
            y2 = self.state[3]
            hist_r = self.hist_read()
            #resort to using non matching blobs
            c = self.get_closest_centroids([(x1,y1),(x2,y2)],
                                           cont,100)
            if(c[0]!=(0.0,0.0) and c[1]!=(0.0,0.0)): #track blobs
                self.set_KF(c[0][0],c[0][1],c[1][0],c[1][1])
                self.predict_KF()
                self.correct_KF()
                self.update_KF(c[0][0],c[0][1],c[1][0],c[1][1])
                self.hist_write(False) #this has a False flag
            
            elif hist_r: #use hist buffer
                #x1 = hist_r[0][0]
                #y1 = hist_r[0][1]
                #x2 = hist_r[1][0]
                #y2 = hist_r[1][1]
                self.set_KF(x1,y1,x2,y2) #use old values
                self.predict_KF()
                self.correct_KF()
                self.update_KF(x1,y1,x2,y2)
                self.hist_write(False) #update using KF
            elif not all(self.hist):
                self.targ_update()
                return np.float32([-16,-16,-16,-16,0,0])
        #return the state
        self.targ_update()
        return np.float32([self.state[0],self.state[1],
                           self.state[2],self.state[3],
                           self.targ_scale(),self.targ_angle()])
        
    
    
    
    
    
    
