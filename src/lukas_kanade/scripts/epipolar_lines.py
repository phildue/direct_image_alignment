
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def estimate_E_compute_F(pts1,pts2,K):
        E, mask1 = cv.findEssentialMat(pts1,pts2,K,method=cv.FM_LMEDS)
        retval, R, t, mask2 = cv.recoverPose(E,pts1[mask1.ravel() == 1],pts2[mask1.ravel() == 1],K)
        tcross = np.cross(t.reshape((3,)),np.identity(3)*-1)

        E = tcross.dot(R)
        Kinv = np.linalg.inv(K)
        F = Kinv.T @ E @ Kinv
        return F, mask1,R,t

def extract_matches(img1,img2):
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        matches = sorted(matches, key=lambda val: val[1].distance)

        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                        pts2.append(kp2[m.trainIdx].pt)
                        pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        return pts1,pts2
def drawlines(img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
                lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
        img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
                color = tuple(np.random.randint(0,255,3).tolist())
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
                img1 = cv.circle(img1,tuple(pt1),5,color,-1)
                img2 = cv.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2
        
if __name__ == "__main__":
        img1 = cv.imread('/media/data/dataset/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk/rgb/1311868164.399026.png',cv.IMREAD_GRAYSCALE)
        img2 = cv.imread('/media/data/dataset/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk/rgb/1311868165.599188.png',cv.IMREAD_GRAYSCALE)
        K = np.array([
                [525.0,     0, 319.5],
                [    0, 525.0, 239.5],
                [    0,     0, 1]])
        
        pts1,pts2 = extract_matches(img1,img2)
        F,mask,R,t = estimate_E_compute_F(pts1,pts2,K)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]


        x2 = np.row_stack([pts2[:,0],pts2[:,1],np.ones((pts2.shape[0],))])
        x1 = np.row_stack([pts1[:,0],pts1[:,1],np.ones((pts1.shape[0],))])
        lines1 = F.T @ x2
        error = (lines1.T * x1.T).sum(-1)# equivalent of diag(lines1.T @ x1) 
        print (f"error = {error.mean():.2f} +- {error.std():.2f}")
        lines2 = F @ x1
        plt.figure()
        img5,img6 = drawlines(img1,img2,lines1.T,pts1,pts2)
        img3,img4 = drawlines(img2,img1,lines2.T,pts2,pts1)
        plt.subplot(121),plt.imshow(img5)
        plt.subplot(122),plt.imshow(img3)
        plt.show()