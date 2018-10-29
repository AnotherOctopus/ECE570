from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from scipy.misc import imread, imsave
from skimage.transform import pyramid_gaussian
from net12_model import  calib12, detect12, detect24
from config import *
import sys

# BB is defined (conf,min_x,min_y,max_x,maxy)
def NMS(boxes):
        numboxesremoved = 0
        for i, box in enumerate(boxes[:-1]):
                def IOU(boxA):
                        if boxA[0] == 65536 or box[0] == 65536:
                                return 0 
                        boxA =  boxA[1:]
                        boxB =  box[1:]
                        ## Borrowed from Pyimagesearch
                        #determine the (x, y)-coordinates of the intersection rectangle
                        xA = int(max(boxA[0], boxB[0]))
                        yA = int(max(boxA[1], boxB[1]))
                        xB = int(min(boxA[2], boxB[2]))
                        yB = int(min(boxA[3], boxB[3]))
                
                        # compute the area of intersection rectangle
                        interArea = max(0, xB - xA ) * max(0, yB - yA )
                
                        # compute the area of both the prediction and ground-truth
                        # rectangles
                        boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
                        boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )
                        union = float(boxAArea + boxBArea - interArea)
                        if interArea > union or union == 0:
                                return 0 
                
                        # compute the intersection over union by taking the intersection
                        # area and dividing it by the sum of prediction + ground-truth
                        # areas - the interesection area
                        iou = interArea / union
                
                        # return the intersection over union value
                        return iou
                ious = np.apply_along_axis(IOU,1,boxes[i+1:,:])
                for iouidx, iou in enumerate(ious):
                        if iou > iouThresh:
                                boxes[i+iouidx] = np.asarray([65536,0,0,0,0])
                                numboxesremoved += 1
        if numboxesremoved == 0:
                return boxes
        boxes  = boxes[boxes[:,0].argsort()]
        return boxes[:-numboxesremoved]
def adjBB(BB,shift):
        actW = int(shift[0]*(BB[3] - BB[1]))
        actH = int(shift[0]*(BB[4] - BB[2]))
        actX = BB[1] + int(shift[1]*actW/shift[0])
        actY = BB[2] + int(shift[2]*actH/shift[0])
        return np.asarray((BB[0],actX , actY ,actX + actW,actY + actH),dtype = np.uint32)
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

def predToShift(prediction,thresh=calib12Thresh):
        totS = 0
        totY = 0
        totX = 0
        Z = np.sum(prediction > thresh)
        if Z == 0:
                return 1, 0 , 0
        for pred,aclass in zip(prediction,adjclass):
                if pred > thresh:
                        calib = adjclassV[adjclass.index(aclass)]
                        totS += calib[0]
                        totX += calib[1]
                        totY += calib[2]
        totS /= Z
        totX /= Z
        totY /= Z
        return totS, totX, totY

def rectifypoints(point,windowsize,frameshape):
        newX1 = max(point[0], windowsize/2)
        newY1 = max(point[1], windowsize/2)
        newX2 = min(point[2], frameshape[0] - windowsize/2)
        newY2 = min(point[3], frameshape[1] - windowsize/2)
        return newX1, newY1, newX2, newY2
def drawfinal(name,rawimg,boxes):
        final = rawimg.copy()
        for box in boxes:
                X1 = int(box[1])
                Y1 = int(box[2])
                X2 = int(box[3])
                Y2 = int(box[4])
                final = cv2.rectangle(final, (X1,Y1), (X2,Y2), (255,0,0))
        cv2.imshow(name,final)
def drawall(name,rawimg,confidences,drawscale = False,scaleup = True):
        final = rawimg.copy()
        for confidence in confidences:
                imageatscale = rawimg.copy()
                scale = confidence["scale"]
                for box in confidence["boxes"]:
                        if scaleup:
                                X1 = int(box[1]*scale)
                                Y1 = int(box[2]*scale)
                                X2 = int(box[3]*scale)
                                Y2 = int(box[4]*scale)
                        else:
                                X1 = int(box[1])
                                Y1 = int(box[2])
                                X2 = int(box[3])
                                Y2 = int(box[4])
                        imageatscale = cv2.rectangle(imageatscale, (X1,Y1), (X2,Y2), (255,0,0))
                        final = cv2.rectangle(final, (X1,Y1), (X2,Y2), (255,0,0))
                if drawscale:
                        cv2.imshow("{} - Scale{}".format(name,scale),imageatscale)
        cv2.imshow("{} - final".format(name),final)
def resizetoshape(img,wind):
        img = Image.fromarray(np.uint8(img*255)).resize((wind[0],wind[1]))
        img = np.asarray(img).astype(np.float32)/255
        img = np.reshape(img,(1,img.shape[0],img.shape[1],3))
        return img

if __name__ == "__main__":
        testfile = "/home/cephalopodoverlord/Downloads/20140928-03.gif"
        detect12model = load_model('net12.h5')
        detect24model = load_model('detect24.h5')
        detect48model = load_model('detect48.h5')
        calib12model = load_model('calib12.h5')
        calib24model = load_model('calib24.h5')
        calib48model = load_model('calib48.h5')
        rawimg = imread(testfile,mode='RGB')
        window = np.reshape(rawimg,(1,rawimg.shape[0],rawimg.shape[1],3))
        scalestep = 1.5
        imgpyr = tuple(pyramid_gaussian(rawimg,downscale =scalestep))
        confidences = []
        for iscale,frame in enumerate(imgpyr):
                confidence = {
                        "confmap":None,
                        "scale":None,
                        "frame":rawimg,
                        "boxes":[]
                }
                frame = resizetoshape(frame,(int(frame.shape[1] * L1Size/float(minface)), int(frame.shape[0]*L1Size/float(minface))))
                if frame.shape[1] < L1Size or frame.shape[2] < L1Size:
                        break
                confidence["confmap"] =  runframe(detect12model,frame,L1Size)
                confidence["scale"] =  float(rawimg.shape[0])/frame.shape[1]
                confToPos = lambda x: (L1Size/2 + x*step)

                threshedboxes = (confidence["confmap"] < detect12Thresh).nonzero()
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
                        shift = calib12model.predict(window[:,box[2]:box[4],box[1]:box[3],:])[0]
                        confidence["boxes"][idx] = adjBB(box,predToShift(shift, thresh = calib12Thresh))
                confidence["boxes"] = NMS(confidence["boxes"])

                confidences.append(confidence)

        drawall("Post12 - postshift, postnms",rawimg,confidences)
        for confidence in confidences:
                scale = confidence["scale"]
                numremoved = 0
                for idx, box in enumerate(confidence["boxes"]):
                        X1 = int(box[1]*scale)
                        Y1 = int(box[2]*scale)
                        X2 = int(box[3]*scale)
                        Y2 = int(box[4]*scale)
                        wind12 = resizetoshape(rawimg[Y1:Y2,X1:X2,:],(L1Size,L1Size))
                        wind24 = resizetoshape(rawimg[Y1:Y2,X1:X2,:],(L2Size,L2Size))
                        conf = detect24model.predict([wind24,wind12])[0][0]
                        if conf > detect24Thresh:
                                confidence["boxes"][idx][0] = conf*65536
                        else:
                                confidence["boxes"][idx][0] = 65536
                                numremoved += 1
                if numremoved == 0:
                        continue
                confidence["boxes"] = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                confidence["boxes"] = confidence["boxes"][:-numremoved]
        drawall("Post24 - preshift, prenms",rawimg,confidences)
        for confidence in confidences:
                scale = confidence["scale"]
                for idx, box in enumerate(confidence["boxes"]):
                        X1 = int(box[1]*scale)
                        Y1 = int(box[2]*scale)
                        X2 = int(box[3]*scale)
                        Y2 = int(box[4]*scale)
                        wind = resizetoshape(rawimg[Y1:Y2,X1:X2,:],(L2Size,L2Size))
                        shift = calib24model.predict(wind)[0]
                        confidence["boxes"][idx] = adjBB(box,predToShift(shift,thresh = calib24Thresh))
        drawall("Post24 - postshift, prenms",rawimg,confidences)
        for confidence in confidences:
                confidence["boxes"] = NMS(confidence["boxes"])
        drawall("Post24 - postshift, postnms",rawimg,confidences)
        for confidence in confidences:
                scale = confidence["scale"]
                numremoved = 0
                for idx, box in enumerate(confidence["boxes"]):
                        X1 = int(box[1]*scale)
                        Y1 = int(box[2]*scale)
                        X2 = int(box[3]*scale)
                        Y2 = int(box[4]*scale)
                        wind12 = resizetoshape(rawimg[Y1:Y2,X1:X2,:],(L1Size,L1Size))
                        wind24 = resizetoshape(rawimg[Y1:Y2,X1:X2,:],(L2Size,L2Size))
                        wind48 = resizetoshape(rawimg[Y1:Y2,X1:X2,:],(L3Size,L3Size))
                        conf = detect48model.predict([wind48,wind24,wind12])[0][0]
                        if conf > detect48Thresh:
                                confidence["boxes"][idx][0] = conf*65536
                                confidence["boxes"][idx][1] = X1
                                confidence["boxes"][idx][2] = Y1
                                confidence["boxes"][idx][3] = X2
                                confidence["boxes"][idx][4] = Y2
                        else:
                                confidence["boxes"][idx][0] = 65536
                                numremoved += 1
                if numremoved == 0:
                        continue
                confidence["boxes"] = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                confidence["boxes"] = confidence["boxes"][:-numremoved]
        drawall("Post48 - Preshift, pre combined",rawimg,confidences,scaleup=False)
        combinedBoxes = np.concatenate([b["boxes"] for b in confidences])
        drawfinal("Post48 - Preshift, Pre nms",rawimg,combinedBoxes)
        combinedBoxes = NMS(combinedBoxes)
        drawfinal("Post48 - Preshift, post nms",rawimg,combinedBoxes)

        for idx, box in enumerate(combinedBoxes):
                wind = resizetoshape(rawimg[box[2]:box[4],box[1]:box[3],:],(L3Size,L3Size))
                shift = calib48model.predict(wind)[0]
                combinedBoxes[idx] = adjBB(box,predToShift(shift,thresh = calib48Thresh))
        drawfinal("Final",rawimg,combinedBoxes)
        cv2.waitKey(0)
