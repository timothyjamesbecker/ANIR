import cv2

class CPU:
    t1,t2 = 0,0
    def start(self):
        self.t1 = cv2.getTickCount()
        
    def stop(self):
        self.t2 = cv2.getTickCount()
        
    def diff(self):
        return (self.t2-self.t1)/cv2.getTickFrequency()

