import numpy as np
import cv2
from scipy.misc import imread, imsave, imresize
from PIL import Image

testfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/test/20140928-03.gif"
confile = "confidence.npy"
confidence = np.load(confile)
finalshape = confidence[0]*12

rawimg = imread(testfile,mode='RGB')
print confidence.shape
print rawimg.shape
rawimg = imresize(rawimg,0.01)

for idx in np.argwhere(confidence < 0.0001):
    if idx[0] != 81:
        continue
    bsize = 12
    xpos = idx[1]*4 + 6
    ypos = idx[2]*4 + 6
    print(xpos,ypos)
    print rawimg.shape
    cv2.rectangle(rawimg,(ypos-bsize,xpos-bsize),(ypos+bsize,xpos+bsize),(0,255,255))
Image.fromarray(rawimg).show()
"""
maxval = np.max(confidence)
minval = np.min(confidence)
if i != 5:
    continue
image = np.asarray(confidence[i])
image = Image.fromarray(image)
image.show()
cv2.imshow("confidence",image)
k = cv2.waitKey(30) & 0xff
if k == 27:
break
"""