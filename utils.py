import os
from config import *
import numpy as np
import cv2
from skimage.transform import pyramid_reduce,pyramid_expand
from scipy.misc import imread
from keras.utils import to_categorical
from datetime import datetime
import time
from keras.models import load_model

def collecttagsfromdir(datadir,tag = "face",shape = 12):
    numsamples = len(os.listdir(os.path.join(datadir,tag)))
    imgs = np.empty((numsamples,shape,shape,3),dtype=np.float32)
    labels = []
    for idx, img, tagn in zip(range(numsamples), os.listdir(os.path.join(datadir,tag)),os.listdir(datadir+"/tag")):
        imgs[idx,:,:,:] = imread(os.path.join(datadir,tag,img))
        with open(os.path.join(datadir,"tag",tagn),"r") as fh:
            labels.append(adjclass.index(fh.read()))
    return imgs, labels
# BB is defined (conf,min_x,min_y,max_x,maxy)
def NMS(boxes):
        numboxesremoved = 0
        idx = 0
        while idx < len(boxes) -1 :
                if boxes[idx][0] == MAXCONF:
                        idx += 1
                        continue
                def IOU(boxA):
                        if boxA[0] == MAXCONF:
                                return 0 
                        boxA =  boxA[1:]
                        boxB =  boxes[idx][1:]
                        aX1 = int(boxA[0])
                        aX2 = int(boxA[2])
                        aY1 = int(boxA[1])
                        aY2 = int(boxA[3])
                        bX1 = int(boxB[0])
                        bX2 = int(boxB[2])
                        bY1 = int(boxB[1])
                        bY2 = int(boxB[3])
                        boxAArea = (aX2 - aX1 ) * (aY2 - aY1)
                        boxBArea = (bX2 - bX1 ) * (bY2 - bY1)

                        x_over = max(0,min(aX2,bX2) - max(aX1,bX1))
                        y_over = max(0,min(aY2,bY2) - max(aY1,bY1))
                        interArea = x_over*y_over
                
                        # compute the area of both the prediction and ground-truth
                        # rectangles
                        union = float(boxAArea + boxBArea - interArea)
                        if union == 0:
                                return 0 
                
                        # compute the intersection over union by taking the intersection
                        # area and dividing it by the sum of prediction + ground-truth
                        # areas - the interesection area
                        iou = interArea / union
                        # return the intersection over union value
                        return iou

                ious = np.apply_along_axis(IOU,1,boxes[idx+1:,:])
                for iouidx, iou in enumerate(ious):
                        if iou > IOUTHRESH:
                                boxes[idx+iouidx+1] = np.asarray([MAXCONF,0,0,0,0])
                                numboxesremoved += 1
                idx += 1
        boxes  = boxes[boxes[:,0].argsort()]
        if numboxesremoved == 0:
                return boxes
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
        sw = Stopwatch()
        sw.start("start")
        maxX = frame.shape[1]
        maxY = frame.shape[2]
        data = np.ones((int((maxX-wsize)/STEP) + 1,int((maxY-wsize)/STEP) + 1))
        framebuffer = np.ones(((data.shape[0]) * (data.shape[1] ),wsize,wsize,frame.shape[3]))
        yistep = data.shape[0]
        sw.lap("genwindows")
        for yi,y in enumerate(range(0,maxY - wsize + 1,STEP)):
                for xi,x in enumerate(range(0,maxX - wsize + 1,STEP)):
                        lowwindx = x
                        lowwindy = y
                        maxwindx = x + wsize
                        maxwindy = y + wsize
                        framebuffer[yi*yistep + xi] = frame[:,lowwindx:maxwindx,lowwindy:maxwindy,:]
        sw.lap("runmodel")
        out = model.predict(framebuffer)
        sw.lap("loop")
        for idx,conf in enumerate(out):
                data[idx%yistep, int(idx/yistep)] = conf
        sw.stop()
        return data

def predToShift(prediction,thresh=CALIB12THRESH):
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
def drawfinal(name,rawimg,boxes,color=(255,0,0)):
        if not DISPLAY:
                return
        final = rawimg.copy()
        for box in boxes:
                X1 = int(box[1])
                Y1 = int(box[2])
                X2 = int(box[3])
                Y2 = int(box[4])
                try:
                        final = cv2.rectangle(final, (Y1,X1), (Y2,X2), color )
                except:
                        pass
        final = pyramid_reduce(final,downscale=4)
        cv2.imshow(name,final)
def drawall(name,rawimg,confidences,drawscale = False,scaleup = True,color = (255,0,0)):
        if not DISPLAY:
                return

        font = cv2.FONT_HERSHEY_SIMPLEX
        final = rawimg.copy()
        textshift = 10
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
                        #cv2.putText(imageatscale, str(box[0]), (X1 + textshift, Y1 + textshift), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        #cv2.putText(final, str(box[0]), (X1 + textshift, Y1 + textshift), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        imageatscale = cv2.rectangle(imageatscale, (Y1,X1), (Y2,X2),color)
                        final = cv2.rectangle(final, (Y1,X1), (Y2,X2), color)
                if drawscale:
                        imageatscale = pyramid_reduce(imageatscale,downscale=4)
                        cv2.imshow("{} - Scale{}".format(name,scale),imageatscale)
        final = pyramid_reduce(final,downscale=4)
        cv2.imshow("{} - final".format(name),final)
def resizetoshape(img,wind):
        wind = wind[0]
        if img.shape[1] > img.shape[0]:
                scale = float(img.shape[0])/wind
        else:
                scale = float(img.shape[1])/wind

        if scale < 1.0:
                img = pyramid_expand(img,upscale=1/scale)
        elif scale > 1.0:
                img = pyramid_reduce(img,downscale=scale)
        img = img[:wind,:wind,:]
        img = img[np.newaxis,:]
        return img
class Stopwatch():
        def __init__(self,disable=False):
                self.starttime = None
                self.endtime = None
                self.live = None
                self.times = {}
                self.disable = disable
        def start(self, name):
                if self.disable:
                        return 
                self.starttime = datetime.now()
                self.live = name
                self.times[name] = datetime.now()
        def lap(self, name):
                if self.disable:
                        return 
                if self.starttime == None:
                        raise Exception("You haven't started the stopwatch")
                self.times[self.live] = datetime.now() - self.times[self.live]
                self.live = name
                self.times[name] = datetime.now()
        def stop(self):
                if self.disable:
                        return 
                if self.starttime == None:
                        raise Exception("You haven't started the stopwatch")
                self.endtime = datetime.now()
                self.times[self.live] = datetime.now() - self.times[self.live]
                self.live = None
        def log(self):
                if self.disable:
                        return  "stopwatch disabled"
                ret = ""
                if self.starttime == None:
                        raise Exception("You haven't started the stopwatch")
                for k,v  in self.times.items():
                        ret += "------------{}-----------\n".format(k)
                        ret += "Duration: {}\n".format(v)
                        ret += "-------------------------\n"
                ret += "Total Runtime: {}\n".format(self.endtime - self.starttime)
                return ret
if __name__ == "__main__":
        detect48model = load_model('detect48.h5')
        testfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/datasets/myfaceinback.JPG"
        rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
        img = pyramid_reduce(rawimg,downscale=4)
        img = img[np.newaxis,:]
        runframe(detect48model,img,48)
        testboxes = np.array([[1,5,3,3,5], [0,0,2,2,0]])
        print NMS(testboxes)
