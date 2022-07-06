
import numpy as np
import cv2 as cv

def estimateF(pts1,pts2):
        return cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
def estimateEandComputeF(pts1,pts2,K):
        E, mask1 = cv.findEssentialMat(pts1,pts2,K,method=cv.FM_LMEDS)
        #R1,R2,t = cv.decomposeEssentialMat(E)
        retval, R, t, mask2 = cv.recoverPose(E,pts1[mask1.ravel() == 1],pts2[mask1.ravel() == 1],K)
        tcross = np.cross(t.reshape((3,)),np.identity(3)*-1)
        print(f"tcross={tcross}")

        E = tcross.dot(R)
        Kinv = np.linalg.inv(K)
        F = Kinv.transpose() @ E @ Kinv
        return F, mask1,R,t
        
