from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from scipy.misc import imread, imsave
from skimage.transform import pyramid_gaussian
from net12_model import  calib12, detect12
from config import *

# BB is defined (conf,min_x,min_y,max_x,maxy)
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
                
                        # compute the area of intersection rectangle
                        interArea = max(0, xB - xA ) * max(0, yB - yA )
                
                        # compute the area of both the prediction and ground-truth
                        # rectangles
                        boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
                        boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )
                        union = float(boxAArea + boxBArea - interArea)
                        if interArea > union:
                                return 0 
                
                        # compute the intersection over union by taking the intersection
                        # area and dividing it by the sum of prediction + ground-truth
                        # areas - the interesection area
                        iou = interArea / union
                
                        # return the intersection over union value
                        return iou
                ious = np.apply_along_axis(IOU,1,boxes[i+1:,1:])
                print ious
        return boxes
def adjBB(BB,shift):
        actW = int(shift[0]*(BB[3] - BB[1]))
        actH = int(shift[0]*(BB[4] - BB[2]))
        actX = BB[1] + int(shift[1]*actW/shift[0])
        actY = BB[2] + int(shift[2]*actH/shift[0])
        return np.asarray((BB[0],actX + actH/2, actY + actH/2 ,actX + actW,actY + actH),dtype = np.uint32)
# returns a compiled model
# identical to the previous one
def runframe(model,frame,wsize):
        maxX = frame.shape[1]
        maxY = frame.shape[2]
        data = np.ones((frame.shape[0],int((maxX-wsize)/step) + 1,int((maxY-wsize)/step) + 1))
        for xi,x in enumerate(range(wsize/2,maxX - wsize/2,step)):
                for yi,y in enumerate(range(wsize/2,maxY - wsize/2,step)):
                        lowwindx = x - wsize/2
                        lowwindy = y - wsize/2
                        maxwindx = x + wsize/2
                        maxwindy = y + wsize/2
                        data[:,xi,yi] = model.predict(frame[:,lowwindx:maxwindx,lowwindy:maxwindy,:])
        return data

def predToShift(predictions):
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
        return totS, totX, totY
if __name__ == "__main__":
        testfile = "/home/cephalopodoverlord/Downloads/gettyimages-493747754-612x612.jpg"
        detect12model = load_model('net12.h5')
        calib12model = load_model('calib12.h5')
        rawimg = imread(testfile,mode='RGB')
        window = np.reshape(rawimg,(1,rawimg.shape[0],rawimg.shape[1],3))
        X = rawimg.shape[0]
        Y = rawimg.shape[1]
        scalestep = 1.2
        imgpyr = tuple(pyramid_gaussian(rawimg,downscale =scalestep))
        confidences = []
        for iscale,frame in enumerate(imgpyr):
                confidence = {
                        "confmap":None,
                        "scale":None,
                        "frame":rawimg,
                        "boxes":[]
                }
                frame = Image.fromarray(np.uint8(frame*255)).resize((int(frame.shape[1] * L1Size/float(minface)), int(frame.shape[0]*L1Size/float(minface))))
                frame = np.asarray(frame).astype(np.float32)/255
                if frame.shape[0] < L1Size or frame.shape[1] < L1Size:
                        break
                frame = np.reshape(frame,(1,frame.shape[0],frame.shape[1],3))
                confidence["confmap"] =  runframe(detect12model,frame,L1Size)
                confidence["scale"] =  scalestep**iscale*(float(minface)/L1Size)
                confToPos = lambda x: (L1Size/2 + x*step)

                threshedboxes = (confidence["confmap"] <detect12Thresh).nonzero()
                confidence["boxes"] = np.zeros((len(threshedboxes[0]),5),dtype=np.uint32)
                for i, box in enumerate(threshedboxes[0]):
                        xidx = threshedboxes[2][i]
                        yidx = threshedboxes[1][i]
                        conf = int(confidence['confmap'][0][yidx][xidx]*65536)
                        X = confToPos(xidx)
                        Y = confToPos(yidx)
                        confidence["boxes"][i] = np.asarray([conf,int(X-L1Size/2),int(Y-L1Size/2),int(X+L1Size/2),int(Y+L1Size/2)])
                confidence["boxes"]  = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                for idx, box in enumerate(confidence["boxes"]):
                        shift = calib12model.predict(window[:,box[2]:box[4],box[1]:box[3],:])
                        confidence["boxes"][idx] = adjBB(box,predToShift(shift))
                confidence["boxes"] = NMS(confidence["boxes"])

                confidences.append(confidence)

        final = rawimg.copy()
        for confidence in confidences:
                imageatscale = rawimg.copy()
                for box in confidence["boxes"]:
                        scale = confidence["scale"]
                        X1 = int(box[1]*scale)
                        X2 = int(box[3]*scale)
                        Y1 = int(box[2]*scale)
                        Y2 = int(box[4]*scale)
                        imageatscale = cv2.rectangle(imageatscale, (X1,Y1), (X2,Y2), (255,0,0))
                        final = cv2.rectangle(final, (X1,Y1), (X2,Y2), (255,0,0))
                cv2.imshow("Scale{}".format(scale),imageatscale)
        cv2.imshow("final",final)
        cv2.waitKey(0)