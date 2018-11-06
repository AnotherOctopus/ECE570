from keras.models import load_model
import numpy as np
import cv2
from imageio import imread, imsave
from skimage.transform import pyramid_gaussian, pyramid_reduce
from config import *
import sys
from utils import *
import time

# Run the whole layer net
if __name__ == "__main__":
        sw = Stopwatch(disable=True)
        # File we are running on

        # Instantiate all the models
        detect12model = load_model('facedetect12.h5')
        detect24model = load_model('facedetect24.h5')
        detect48model = load_model('facedetect48.h5')
        calib12model  = load_model('facecalib12.h5')
        calib24model  = load_model('facecalib24.h5')
        calib48model  = load_model('facecalib48.h5')


        handdetect12model = load_model('handdetect12.h5')
        handdetect24model = load_model('handdetect24.h5')
        handdetect48model = load_model('handdetect48.h5')
        handcalib12model  = load_model('handcalib12.h5')
        handcalib24model  = load_model('handcalib24.h5')
        handcalib48model  = load_model('handcalib48.h5')

        sw.start("load image and scale")
        # Load image. Raw and batch variant
        while True:
                if False:
                        cap = cv2.VideoCapture("rtsp://192.168.0.124:554/mpeg4?username=admin&password=123456")
                        ret, inpimg = cap.read() 
                        while not ret :
                                time.sleep(1)
                                ret, inpimg = cap.read() 
                        cap.release()

                else:
                        testfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/datasets/myfaceinback.JPG"
                        inpimg = imread(testfile,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                frame = rawimg

                sw.lap("12net detect")
                # Generate the first batch of confidences
                scale = 1.0001
                scalestep = 0.05
                confidences = []
                handconfidences = []
                while True: 
                        frame = pyramid_reduce(rawimg,downscale=scale*float(MINFACE)/L1SIZE)
                        if frame.shape[0] < 3*L1SIZE and frame.shape[1] < 3*L1SIZE:
                                break
                        print frame.shape, scale
                        brawimg = np.reshape(frame,(1,frame.shape[0],frame.shape[1],3))
                        confidence = {
                                "confmap":None, # Map of confidences on the map space
                                "scale":1, # how much to upcale the image to reach original shape
                                "frame":brawimg, # The downscaled image
                                "boxes":[] # Boxes. (confidence,minX,minY,maxX,maxY)
                        }
                        handconfidence = {
                                "confmap":None, # Map of confidences on the map space
                                "scale":1, # how much to upcale the image to reach original shape
                                "frame":brawimg, # The downscaled image
                                "boxes":[] # Boxes. (confidence,minX,minY,maxX,maxY)
                        }


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

                        handconfidence["confmap"] =  runframe(handdetect12model,brawimg,L1SIZE)
                        handconfidence["scale"] =  float(rawimg.shape[0])/frame.shape[0]
                        confToPos = lambda x: (x*STEP)

                        threshedboxes = (handconfidence["confmap"] < DETECT12THRESH).nonzero()
                        handconfidence["boxes"] = np.zeros((len(threshedboxes[0]),5),dtype=np.uint32)
                        for i, box in enumerate(threshedboxes[0]):
                                xidx = threshedboxes[0][i]
                                yidx = threshedboxes[1][i]
                                conf = int(handconfidence['confmap'][xidx][yidx]*MAXCONF)
                                X = int(confToPos(xidx))
                                Y = int(confToPos(yidx))
                                handconfidence["boxes"][i] = np.asarray([conf,X,Y,X+L1SIZE,Y+L1SIZE])
                        handconfidence["boxes"]  = handconfidence["boxes"][handconfidence["boxes"][:,0].argsort()]

                        handconfidences.append(handconfidence)
                        confidences.append(confidence)
                        scale += scale*scale*scalestep

                sw.lap("12net calib")
                for confidence in confidences:
                        for idx, box in enumerate(confidence["boxes"]):
                                shift = calib12model.predict(confidence["frame"][:,box[1]:box[3],box[2]:box[4],:])[0]
                                confidence["boxes"][idx] = adjBB(box,predToShift(shift, thresh = CALIB12THRESH))

                for confidence in handconfidences:
                        for idx, box in enumerate(confidence["boxes"]):
                                shift = handcalib12model.predict(confidence["frame"][:,box[1]:box[3],box[2]:box[4],:])[0]
                                confidence["boxes"][idx] = adjBB(box,predToShift(shift, thresh = CALIB12THRESH))
                sw.lap("12net nms")
                for confidence in confidences:
                        confidence["boxes"] = NMS(confidence["boxes"])

                for confidence in handconfidences:
                        confidence["boxes"] = NMS(confidence["boxes"])
                drawall("12net postcalib",rawimg,confidences)

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

                for confidence in handconfidences:
                        scale = confidence["scale"]
                        numremoved = 0
                        for idx, box in enumerate(confidence["boxes"]):
                                X1 = int(box[1]*scale)
                                Y1 = int(box[2]*scale)
                                X2 = int(box[3]*scale)
                                Y2 = int(box[4]*scale)
                                wind12 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L1SIZE,L1SIZE))
                                wind24 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L2SIZE,L2SIZE))
                                conf = handdetect24model.predict([wind24,wind12])[0][0]
                                if conf > DETECT24THRESH:
                                        confidence["boxes"][idx][0] = (1-conf)*MAXCONF
                                else:
                                        confidence["boxes"][idx][0] = MAXCONF
                                        numremoved += 1
                        if numremoved == 0:
                                continue
                        confidence["boxes"] = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                        confidence["boxes"] = confidence["boxes"][:-numremoved]

                drawall("24net",rawimg,confidences)
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

                for confidence in handconfidences:
                        scale = confidence["scale"]
                        for idx, box in enumerate(confidence["boxes"]):
                                X1 = int(box[1]*scale)
                                Y1 = int(box[2]*scale)
                                X2 = int(box[3]*scale)
                                Y2 = int(box[4]*scale)
                                wind = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L2SIZE,L2SIZE))
                                shift = handcalib24model.predict(wind)[0]
                                confidence["boxes"][idx] = adjBB(box,predToShift(shift,thresh = CALIB24THRESH))
                sw.lap("24net nms")
                for confidence in confidences:
                        confidence["boxes"] = NMS(confidence["boxes"])

                for confidence in handconfidences:
                        confidence["boxes"] = NMS(confidence["boxes"])

                drawall("24net",rawimg,confidences)
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
                for confidence in handconfidences:
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
                                conf = handdetect48model.predict([wind48,wind24,wind12])[0][0]
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

                handcombinedBoxes = np.concatenate([b["boxes"] for b in confidences])
                handcombinedBoxes = handcombinedBoxes[combinedBoxes[:,0].argsort()]

                sw.lap("48net NMS")
                combinedBoxes = NMS(combinedBoxes)
                handcombinedBoxes = NMS(handcombinedBoxes)

                sw.lap("48net calib")
                for idx, box in enumerate(combinedBoxes):
                        wind = resizetoshape(rawimg[box[1]:box[3],box[2]:box[4],:],(L3SIZE,L3SIZE))
                        shift = calib48model.predict(wind)[0]
                        combinedBoxes[idx] = adjBB(box,predToShift(shift,thresh = CALIB48THRESH))

                for idx, box in enumerate(handcombinedBoxes):
                        wind = resizetoshape(rawimg[box[1]:box[3],box[2]:box[4],:],(L3SIZE,L3SIZE))
                        shift = handcalib48model.predict(wind)[0]
                        combinedBoxes[idx] = adjBB(box,predToShift(shift,thresh = CALIB48THRESH))

                print combinedBoxes
                drawfinal("Final",rawimg,combinedBoxes)
                drawfinal("Final Hand",rawimg,combinedBoxes)
                for box in combinedBoxes:
                        wind = rawimg[box[1]:box[3],box[2]:box[4],:]
                        print box[0]
                sw.stop()
                print sw.log()
                cv2.waitKey(30)
