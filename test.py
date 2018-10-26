from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from scipy.misc import imread, imsave
from skimage.transform import pyramid_gaussian
from net12_model import  calib12, detect12
from config import *

def NMS(boxes):

        for i, box in enumerate(boxes[:-1]):
                def IOU(boxA):
                        boxB =  box[1:]
                        ## Borrowed from Pyimagesearch
                        #determine the (x, y)-coordinates of the intersection rectangle
                        xA = max(boxA[0], boxB[0])
                        yA = max(boxA[1], boxB[1])
                        xB = min(boxA[2], boxB[2])
                        yB = min(boxA[3], boxB[3])
                        print (xA,yA,xB,yB)
                
                        # compute the area of intersection rectangle
                        interArea = max(0, xB - xA ) * max(0, yB - yA )
                        print interArea
                
                        # compute the area of both the prediction and ground-truth
                        # rectangles
                        boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
                        boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )
                        print boxAArea, boxBArea
                
                        # compute the intersection over union by taking the intersection
                        # area and dividing it by the sum of prediction + ground-truth
                        # areas - the interesection area
                        iou = interArea / float(boxAArea + boxBArea - interArea)
                
                        # return the intersection over union value
                        return iou
                ious = np.apply_along_axis(IOU,1,boxes[i+1:,1:])
                print "CHECK HEAR"
                print box[1:]
                print boxes[i+1:,1:]
                print ious
        return boxes
NMS(np.asarray([[0.1,1,1,3,3],[0.1,2,0,4,2]]))
sys
def testcalib12():
        c12 = calib12()
        testfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/adj12/train/face/1.jpg"
        validfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/adj12/train/tag/1.txt"
        model = load_model('calib12.h5')
        rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
        rawimg = rawimg[np.newaxis,...]
        predictions =  model.predict(rawimg)

        for prediction in predictions:
                totS = 0
                totY = 0
                totX = 0
                Z = np.sum(prediction > calib12Tresh)
                for pred,aclass in zip(prediction,adjclass):
                        if pred > calib12Tresh:
                                calib = adjclassV[adjclass.index(aclass)]
                                totS += calib[0]
                                totX += calib[1]
                                totY += calib[2]
                totS /= Z
                totX /= Z
                totY /= Z

        print "ACTUAL", adjclassV[tag]
        print "Predic", totS, totX, totY
def testdetect12():
        d12 = detect12()
        testfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/detect12/train/notface/8.jpg"
        model = load_model('net12.h5')
        rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
        rawimg = rawimg[np.newaxis,...]
        predictions =  model.predict(rawimg)


        print predictions
testcalib12()
