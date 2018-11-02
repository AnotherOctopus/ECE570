from keras.models import load_model
import numpy as np
import cv2
from scipy.misc import imread, imsave
from skimage.transform import pyramid_gaussian, pyramid_reduce
from config import *
import sys
from utils import *

# Run the whole layer net
if __name__ == "__main__":
        sw = Stopwatch()
        # File we are running on
        testfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/datasets/myfaceinback.JPG"

        # Instantiate all the models
        detect12model = load_model('net12.h5')
        detect24model = load_model('detect24.h5')
        detect48model = load_model('detect48.h5')
        calib12model  = load_model('calib12.h5')
        calib24model  = load_model('calib24.h5')
        calib48model  = load_model('calib48.h5')

        sw.start("load image and scale")
        # Load image. Raw and batch variant
        rawimg = imread(testfile,mode='RGB').astype(np.float32)/255
        rawimg = pyramid_reduce(rawimg,downscale=float(MINFACE)/L1SIZE)

        # Generate a Downscaled variant of the image, down to 12x12 img
        scalestep = 1.2
        imgpyr = tuple(pyramid_gaussian(rawimg,downscale =scalestep))

        sw.lap("12net detect")
        # Generate the first batch of confidences
        confidences = []
        for iscale,frame in enumerate(imgpyr):
                brawimg = np.reshape(frame,(1,frame.shape[0],frame.shape[1],3))
                confidence = {
                        "confmap":None, # Map of confidences on the map space
                        "scale":1, # how much to upcale the image to reach original shape
                        "frame":brawimg, # The downscaled image
                        "boxes":[] # Boxes. (confidence,minX,minY,maxX,maxY)
                }

                if frame.shape[0] < L1SIZE or frame.shape[1] < L1SIZE:
                        break
                confidence["confmap"] =  runframe(detect12model,brawimg,L1SIZE)
                confidence["scale"] =  float(rawimg.shape[0])/frame.shape[0]
                confToPos = lambda x: (x*STEP)

                threshedboxes = (confidence["confmap"] < DETECT12THRESH).nonzero()
                confidence["boxes"] = np.zeros((len(threshedboxes[0]),5),dtype=np.uint32)
                for i, box in enumerate(threshedboxes[0]):
                        xidx = threshedboxes[0][i]
                        yidx = threshedboxes[1][i]
                        conf = int(confidence['confmap'][xidx][yidx]*MAXCONF)
                        X = int(confToPos(xidx))
                        Y = int(confToPos(yidx))
                        confidence["boxes"][i] = np.asarray([conf,X,Y,X+L1SIZE,Y+L1SIZE])
                confidence["boxes"]  = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                confidences.append(confidence)
        sw.lap("12net calib")
        for confidence in confidences:
                for idx, box in enumerate(confidence["boxes"]):
                        shift = calib12model.predict(confidence["frame"][:,box[1]:box[3],box[2]:box[4],:])[0]
                        confidence["boxes"][idx] = adjBB(box,predToShift(shift, thresh = CALIB12THRESH))
        sw.lap("12net nms")
        for confidence in confidences:
                confidence["boxes"] = NMS(confidence["boxes"])
        drawall("12net",rawimg,confidences)

        sw.lap("24net detect")
        for confidence in confidences:
                scale = confidence["scale"]
                numremoved = 0
                for idx, box in enumerate(confidence["boxes"]):
                        X1 = int(box[1]*scale)
                        Y1 = int(box[2]*scale)
                        X2 = int(box[3]*scale)
                        Y2 = int(box[4]*scale)
                        wind12 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L1SIZE,L1SIZE))
                        wind24 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L2SIZE,L2SIZE))
                        conf = detect24model.predict([wind24,wind12])[0][0]
                        if conf > DETECT24THRESH:
                                confidence["boxes"][idx][0] = (1-conf)*MAXCONF
                        else:
                                confidence["boxes"][idx][0] = MAXCONF
                                numremoved += 1
                if numremoved == 0:
                        continue
                confidence["boxes"] = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                confidence["boxes"] = confidence["boxes"][:-numremoved]
        sw.lap("24net calib")
        for confidence in confidences:
                scale = confidence["scale"]
                for idx, box in enumerate(confidence["boxes"]):
                        X1 = int(box[1]*scale)
                        Y1 = int(box[2]*scale)
                        X2 = int(box[3]*scale)
                        Y2 = int(box[4]*scale)
                        wind = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L2SIZE,L2SIZE))
                        shift = calib24model.predict(wind)[0]
                        confidence["boxes"][idx] = adjBB(box,predToShift(shift,thresh = CALIB24THRESH))
        sw.lap("24net nms")
        for confidence in confidences:
                confidence["boxes"] = NMS(confidence["boxes"])
        sw.lap("48net detect")
        for confidence in confidences:
                scale = confidence["scale"]
                numremoved = 0
                for idx, box in enumerate(confidence["boxes"]):
                        X1 = int(box[1]*scale)
                        Y1 = int(box[2]*scale)
                        X2 = int(box[3]*scale)
                        Y2 = int(box[4]*scale)
                        wind12 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L1SIZE,L1SIZE))
                        wind24 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L2SIZE,L2SIZE))
                        wind48 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L3SIZE,L3SIZE))
                        conf = detect48model.predict([wind48,wind24,wind12])[0][0]
                        if conf > DETECT48THRESH:
                                confidence["boxes"][idx][0] = (1-conf)*MAXCONF
                                confidence["boxes"][idx][1] = X1
                                confidence["boxes"][idx][2] = Y1
                                confidence["boxes"][idx][3] = X2
                                confidence["boxes"][idx][4] = Y2
                        else:
                                confidence["boxes"][idx][0] = MAXCONF
                                numremoved += 1
                if numremoved == 0:
                        continue
                confidence["boxes"] = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                confidence["boxes"] = confidence["boxes"][:-numremoved]
        combinedBoxes = np.concatenate([b["boxes"] for b in confidences])
        combinedBoxes = combinedBoxes[combinedBoxes[:,0].argsort()]
        sw.lap("48net NMS")
        combinedBoxes = NMS(combinedBoxes)

        sw.lap("48net calib")
        for idx, box in enumerate(combinedBoxes):
                wind = resizetoshape(rawimg[box[1]:box[3],box[2]:box[4],:],(L3SIZE,L3SIZE))
                shift = calib48model.predict(wind)[0]
                combinedBoxes[idx] = adjBB(box,predToShift(shift,thresh = CALIB48THRESH))
        sw.stop()
        drawfinal("Final",rawimg,combinedBoxes)
        print combinedBoxes
        print sw.log()
        cv2.waitKey(0)
