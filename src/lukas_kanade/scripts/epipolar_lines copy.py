
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from epipolar_lines import estimateEandComputeF,estimateF
img1 = cv.imread('/media/data/dataset/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk/rgb/1311868164.399026.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('/media/data/dataset/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk/rgb/1311868165.599188.png',cv.IMREAD_GRAYSCALE)
K = np.array([
        [525.0,     0, 319.5],
        [    0, 525.0, 239.5],
        [    0,     0, 1]])
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
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

#F,mask = estimateF(pts1,pts2)
F,mask,R,t = estimateEandComputeF(pts1,pts2,K)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
img1_rgb = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
img2_rgb = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
for i, ft in enumerate(pts1):
        u,v = ft
        ft_target_0 = K.dot((R.dot(np.linalg.inv(K).dot(np.array([u,v,1]).reshape((3,))))) + t.reshape((3,)))
        ft_target_1 = K.dot(R.dot(np.linalg.inv(K).dot(np.array([u,v,100]).reshape((3,)))) + t.reshape((3,)))
        ft_target_0 /= ft_target_0[2]
        ft_target_1 /= ft_target_1[2]
        ft_target_0 = ft_target_0[:2].astype(int)
        ft_target_1 = ft_target_1[:2].astype(int)
        color = tuple(np.random.randint(0,255,3).tolist())

        img1_rgb = cv.circle(img1_rgb, (int(u),int(v)), 5, color, -1)
        img2_rgb = cv.circle(img2_rgb, pts2[i], 5, color, -1)
        img2_rgb = cv.line(img2_rgb, ft_target_0, ft_target_1, color, 1)

img_stack = np.hstack([img1_rgb,img2_rgb])
cv.imshow("Out",img_stack)
cv.waitKey(0)

"""
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
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
"""