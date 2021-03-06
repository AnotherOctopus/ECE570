import csv
import random
import numpy as np
import os
import cv2
from scipy.misc import imread,imsave
from detect12model import detect12
from skimage.transform import pyramid_reduce
from config import *
from utils import *
from scipy.misc import imread, imsave

datasetdir = "datasets/"
#calibration perterbations
import time
timestamp = int(time.time())
random.seed(timestamp)
def umdcsvtobb(ident,datapoint):
        ret = {
                "id":0,
                "filename":"",
                "xpos":0,
                "ypos":0,
                "w":0,
                "h":0
              }
        ret["id"] = ident
        ret["filename"] = os.path.join(datasetdir,"umdfaces_batch1",datapoint[1])
        ret["xpos"] = float(datapoint[4])
        ret["ypos"] = float(datapoint[5])
        ret["w"] = float(datapoint[6])
        ret["h"] = float(datapoint[7])
        return ret
def randombb(imagesize,minwind,maxwind):
        ret = {
                "id":0,
                "filename":"",
                "xpos":0,
                "ypos":0,
                "w":0,
                "h":0
              }
        ret["id"] = -1
        ret["filename"] = "NULL"
        ret["xpos"] = random.uniform(0,imagesize[0] - maxwind)
        ret["ypos"] = random.uniform(0,imagesize[1] - maxwind)
        ret["w"] = random.uniform(minwind,maxwind)
        ret["h"] = ret["w"]
        return ret
def displaybb(boundbox):
        image = cv2.imread(os.path.join(datasetdir,"umdfaces_batch1",boundbox["filename"]))
        if not image:
                print("image not found")
                return
        cv2.imshow(image)
        cv2.waitKey(30)

def gethandBB(frame):
        ret = {
                "id":0,
                "filename":"",
                "xpos":0,
                "ypos":0,
                "w":0,
                "h":0
              }
        whitethresh = 250
        grayimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        threshedboxes = grayimg < whitethresh
        minX = 0
        maxX = threshedboxes.shape[0]
        minY = 0
        maxY = threshedboxes.shape[1]
        for i in range(threshedboxes.shape[0]):
                if any(threshedboxes[i,:]):
                       minX = i 
                       break
        for i in range(threshedboxes.shape[1]):
                if any(threshedboxes[:,i]):
                       minY = i 
                       break
        for i in range(threshedboxes.shape[0]-1,-1,-1):
                if any(threshedboxes[i,:]):
                       maxX = i 
                       break
        for i in range(threshedboxes.shape[1]-1,-1,-1):
                if any(threshedboxes[:,i]):
                       maxY = i 
                       break
        ret["xpos"] = minX
        ret["ypos"] = minY
        ret["w"] = maxX - minX
        ret["h"] = maxY - minY
        return ret
def prepdetecthand(numface,datadir,trainvalidratio=10,size = 96,startid=1):
        traindir = datadir + "train/"
        validdir = datadir + "valid/"
        tagcsv = datasetdir + "Hands/HandInfo.csv"
        handdata = datasetdir + "Hands"
        imcnt = 0
        with open(tagcsv,"r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                        if "palmar" in row["aspectOfHand"]:
                                img = imread(os.path.join(handdata,row["imageName"]))
                                img = cv2.flip(img,0)
                                hand =  gethandBB(img)
                                hand["id"] = imcnt
                                area = (int(hand['xpos']),
                                        int(hand['ypos']), 
                                        int(hand['xpos']+hand['w']),
                                        int(hand['ypos']+hand['h']))
                                img = img[area[0]:area[2],area[1]:area[3],:]
                                if hand['w'] > hand['h']:
                                        img = pyramid_reduce(img,downscale=hand['h']/float(size))
                                        img = img[(img.shape[0]-size)/2:(img.shape[0]+size)/2 ,:,:]
                                else:
                                        img = pyramid_reduce(img,downscale=hand['w']/float(size))
                                        img = img[:,(img.shape[1]-size)/2:(img.shape[1]+size)/2,:]
                                img = img[:size,:size,:]
                                if hand['id']%trainvalidratio == 0:
                                        hand['filename'] = os.path.join(validdir,"hand","{}.jpg".format(hand['id']))
                                else:
                                        hand['filename'] = os.path.join(traindir,"hand","{}.jpg".format(hand['id']))
                                if img.shape[0] != size or img.shape[1] != size:
                                        raise Exception("BLAH")
                                imsave(hand['filename'],img)
                                print(size,"detect-hand",hand)
                                imcnt += 1
def prepdetect(numface,datadir,trainvalidratio=10,size = 12,startid=1):
        traindir = datadir + "train/"
        validdir = datadir + "validation/"
        tagcsv = datasetdir + "umdfaces_batch1/umdfaces_batch1_ultraface.csv"
        with open(tagcsv, 'r') as csvfile:
                data = list(csv.reader(csvfile))    

        lengthofcsv = len(data)
        for i in range(startid,numface + startid):
                position = random.choice(data[1:])
                face = umdcsvtobb(i,position)
                img = imread(face['filename'])
                area = (int(face['xpos']),
                        int(face['ypos']), 
                        int(face['xpos']+face['w']),
                        int(face['ypos']+face['h']))

                if face['w'] < size or face['h'] < size:
                        continue
                img = img[area[1]:area[3],area[0]:area[2],:]
                if face['w'] > face['h']:
                        img = pyramid_reduce(img,downscale=face['h']/float(size))
                        img = img[(img.shape[0]-size)/2:(img.shape[0]+size)/2 ,:,:]
                else:
                        img = pyramid_reduce(img,downscale=face['w']/float(size))
                        img = img[:,(img.shape[1]-size)/2:(img.shape[1]+size)/2,:]

                img = img[:size,:size,:]
                if face['id']%trainvalidratio == 0:
                        face['filename'] = os.path.join(validdir,"face","{}.jpg".format(face['id']))
                else:
                        face['filename'] = os.path.join(traindir,"face","{}.jpg".format(face['id']))
                if img.shape[0] != size or img.shape[1] != size:
                        raise Exception("BLAH")
                imsave(face['filename'],img)
                print(size,lengthofcsv,"detect-face",face)

def prepbackground(numback,datadir,trainvalidratio=10,size = 12,startid = 0):
        traindir = datadir + "train/"
        validdir = datadir + "validation/"
        backgroundfeeddir = datasetdir + "classroombackround/"
        bkgroundimgs = os.listdir(backgroundfeeddir)
        for i in range(startid,numback + startid):
                backimg = bkgroundimgs[i%len(bkgroundimgs)]
                img = imread(os.path.join(backgroundfeeddir,backimg))
                if size == 12:
                        notface = randombb(img.shape,12,14)
                        area = (int(notface['xpos']),
                                int(notface['ypos']), 
                                int(notface['xpos']+12),
                                int(notface['ypos']+12))
                        img = img[area[0]:area[2],area[1]:area[3],:]
                else:
                        notface = randombb(img.shape,48,96)
                        area = (int(notface['xpos']),
                                int(notface['ypos']), 
                                int(notface['xpos']+notface['w']),
                                int(notface['ypos']+notface['h']))
                        img = img[area[0]:area[2],area[1]:area[3],:]
                        img = pyramid_reduce(img,downscale=notface['h']/float(size))
                        img = img[:size,:size,:]

                notface['id'] = i
                notface['filename'] = os.path.join(backgroundfeeddir,backimg)
                if notface['id']%trainvalidratio == 0:
                        notface['filename'] = os.path.join(validdir,"notface","{}.jpg".format(notface['id']))
                else:
                        notface['filename'] = os.path.join(traindir,"notface","{}.jpg".format(notface['id']))

                imsave(notface['filename'],img)
                print(size,"detect-notface",notface)
def perterb(BB):
        si = random.choice(sn)
        xi = random.choice(xn)
        yi = random.choice(yn)
        newBB = {}
        newBB["id"] = BB["id"]
        newBB["filename"] = BB["filename"]
        newBB["xpos"] = BB["xpos"] - xi*BB["w"]/si
        newBB["ypos"] = BB["ypos"] - yi*BB["h"]/si
        newBB["w"] = BB["w"]/si
        newBB["h"] = BB["h"]/si
        return newBB,(si,xi,yi)
def bubbleup(detect12,detect24,detect48):
        traindir12 = detect12 + "train/"
        validdir12 = detect12 + "validation/"
        traindir24 = detect24 + "train/"
        validdir24 = detect24 + "validation/"
        traindir48 = detect48 + "train/"
        validdir48 = detect48 + "validation/"
        id24 = len(os.listdir( traindir24 + "face")) + \
                    len(os.listdir( traindir24 + "notface")) + \
                    len(os.listdir( validdir24 + "face")) + \
                    len(os.listdir( validdir24 + "notface"))
        id48 = len(os.listdir( traindir48 + "face")) + \
                    len(os.listdir( traindir48 + "notface")) + \
                    len(os.listdir( validdir48 + "face")) + \
                    len(os.listdir( validdir48 + "notface"))
        for d in os.listdir(traindir12 +"face"):
                item = os.path.join(traindir12,"face",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind24 = resizetoshape(rawimg,(L2SIZE,L2SIZE))
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(traindir24,"face","{}.jpg".format(str(id24))),wind24[0,:,:,:])
                id24 += 1
                imsave(os.path.join(traindir48,"face","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
        for d in os.listdir(validdir12 +"face"):
                item = os.path.join(validdir12,"face",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind24 = resizetoshape(rawimg,(L2SIZE,L2SIZE))
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(validdir24,"face","{}.jpg".format(str(id24))),wind24[0,:,:,:])
                id24 += 1
                imsave(os.path.join(validdir48,"face","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
        for d in os.listdir(traindir24 +"face"):
                item = os.path.join(traindir24,"face",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(traindir48,"face","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
        for d in os.listdir(validdir24 +"face"):
                item = os.path.join(validdir24,"face",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(validdir48,"face","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
        for d in os.listdir(traindir12 +"notface"):
                item = os.path.join(traindir12,"notface",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind24 = resizetoshape(rawimg,(L2SIZE,L2SIZE))
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(traindir24,"notface","{}.jpg".format(str(id24))),wind24[0,:,:,:])
                id24 += 1
                imsave(os.path.join(traindir48,"notface","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
        for d in os.listdir(validdir12 +"notface"):
                item = os.path.join(validdir12,"notface",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind24 = resizetoshape(rawimg,(L2SIZE,L2SIZE))
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(validdir24,"notface","{}.jpg".format(str(id24))),wind24[0,:,:,:])
                id24 += 1
                imsave(os.path.join(validdir48,"notface","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
        for d in os.listdir(traindir24 +"notface"):
                item = os.path.join(traindir24,"notface",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(traindir48,"notface","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
        for d in os.listdir(validdir24 +"notface"):
                item = os.path.join(validdir24,"notface",d)
                print item
                inpimg = imread(item,mode='RGB')
                rawimg = inpimg.astype(np.float32)/255
                wind48 = resizetoshape(rawimg,(L3SIZE,L3SIZE))
                imsave(os.path.join(validdir48,"notface","{}.jpg".format(str(id48))),wind48[0,:,:,:])
                id48 += 1
def prephandcalib(datadir, trainvalidratio=10,size=12,startid = 0):
        traindir = datadir + "train/"
        validdir = datadir + "valid/"
        tagcsv = datasetdir + "Hands/HandInfo.csv"
        handdata = datasetdir + "Hands"
        imcnt = 0
        with open(tagcsv,"r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                        if "palmar" in row["aspectOfHand"]:
                                img = imread(os.path.join(handdata,row["imageName"]))
                                img = cv2.flip(img,0)
                                hand = gethandBB(img)
                                hand["id"] = imcnt
                                hand,pert = perterb(hand)
                                area = (int(hand['xpos']),
                                        int(hand['ypos']), 
                                        int(hand['xpos']+hand['w']),
                                        int(hand['ypos']+hand['h']))

                                if area[3] > img.shape[1] or area[2] > img.shape[0] or area[0] < 0 or area[1] < 0:
                                        continue

                                img = img[area[0]:area[2],area[1]:area[3],:]
                                if hand['w'] > hand['h']:
                                        img = pyramid_reduce(img,downscale=hand['h']/float(size))
                                        img = img[(img.shape[0]-size)/2:(img.shape[0]+size)/2 ,:,:]
                                else:
                                        img = pyramid_reduce(img,downscale=hand['w']/float(size))
                                        img = img[:,(img.shape[1]-size)/2:(img.shape[1]+size)/2,:]
                                img = img[:size,:size,:]
                                if hand['id']%trainvalidratio == 0:
                                        savedir = validdir
                                else:
                                        savedir = traindir
                                if img.shape[0] != size or img.shape[1] != size:
                                        print img.shape
                                        print hand
                                        print pert
                                        raise Exception("blah")
                                hand['filename'] = os.path.join(savedir,"hand","{}.jpg".format(hand['id']))
                                with open( os.path.join(savedir,"tag","{}.txt".format(hand['id'])),"w+") as fh:
                                        fh.write(",".join(map(str,pert))) 
                                imsave(hand['filename'],img)
                                print(size,"calib: ",hand)
                                imcnt += 1
def prepbackground24(numface, datadir, trainvalidratio=10,startid = 0):
        traindir = datadir + "train/"
        validdir = datadir + "validation/"
        backgroundfeeddir = datasetdir + "classroombackround/"
        bkgroundimgs = os.listdir(backgroundfeeddir)
        detect12model = load_model('facedetect12.h5')
        calib12model  = load_model('facecalib12.h5')
        imgnum = 0
        for i in range(len(bkgroundimgs)):
                backimg = bkgroundimgs[i]
                rawimg = imread(os.path.join(backgroundfeeddir,backimg))
                # Generate the first batch of confidences
                scale = 1.0001
                scalestep = 0.10
                confidences = []
                print backimg
                while True: 
                        frame = pyramid_reduce(rawimg,downscale=scale*float(MINFACE)/L1SIZE)
                        print frame.shape
                        if frame.shape[0] < L1SIZE and frame.shape[1] < L1SIZE:
                                break
                        brawimg = np.reshape(frame,(1,frame.shape[0],frame.shape[1],3))
                        confidence = {
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

                        confidences.append(confidence)
                        scale += scale*scale*scalestep

                drawall("12net prenms",rawimg,confidences)
                for confidence in confidences:
                        for idx, box in enumerate(confidence["boxes"]):
                                shift = calib12model.predict(confidence["frame"][:,box[1]:box[3],box[2]:box[4],:])[0]
                                shift = predToShift(shift, thresh = CALIB12THRESH)
                                confidence["boxes"][idx] = adjBB(box,shift)

                for confidence in confidences:
                        confidence["boxes"]  = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                        confidence["boxes"] = NMS(confidence["boxes"])
                for confidence in confidences:
                        scale = confidence["scale"]
                        numremoved = 0
                        for idx, box in enumerate(confidence["boxes"]):
                                X1 = int(box[1]*scale)
                                Y1 = int(box[2]*scale)
                                X2 = int(box[3]*scale)
                                Y2 = int(box[4]*scale)
                                notface = {}
                                wind12 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L1SIZE,L1SIZE))
                                wind24 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L2SIZE,L2SIZE))
                                notface['id'] = imgnum
                                imgnum += 1
                                notface['filename'] = os.path.join(backgroundfeeddir,backimg)
                                if notface['id']%trainvalidratio == 0:
                                        notface['filename'] = os.path.join(validdir,"notface","24-{}.jpg".format(notface['id']))
                                else:
                                        notface['filename'] = os.path.join(traindir,"notface","24-{}.jpg".format(notface['id']))

                                imsave(notface['filename'],wind24[0,:,:,:])
                                if notface['id']%trainvalidratio == 0:
                                        notface['filename'] = os.path.join(validdir,"notface","12-{}.jpg".format(notface['id']))
                                else:
                                        notface['filename'] = os.path.join(traindir,"notface","12-{}.jpg".format(notface['id']))

                                imsave(notface['filename'],wind12[0,:,:,:])
                                print(24,"detect-notface",notface)

def prepbackground48(numface, datadir, trainvalidratio=10,startid = 0):
        traindir = datadir + "train/"
        validdir = datadir + "validation/"
        backgroundfeeddir = datasetdir + "classroombackround/"
        bkgroundimgs = os.listdir(backgroundfeeddir)
        detect12model = load_model('facedetect12.h5')
        detect24model = load_model('facedetect24.h5')
        calib12model  = load_model('facecalib12.h5')
        calib24model  = load_model('facecalib24.h5')
        imgnum = 0
        for i in range(len(bkgroundimgs)):
                backimg = bkgroundimgs[i]
                rawimg = imread(os.path.join(backgroundfeeddir,backimg))
                # Generate the first batch of confidences
                scale = 1.0001
                scalestep = 0.10
                confidences = []
                print backimg
                while True: 
                        frame = pyramid_reduce(rawimg,downscale=scale*float(MINFACE)/L1SIZE)
                        print frame.shape
                        if frame.shape[0] < L1SIZE and frame.shape[1] < L1SIZE:
                                break
                        brawimg = np.reshape(frame,(1,frame.shape[0],frame.shape[1],3))
                        confidence = {
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

                        confidences.append(confidence)
                        scale += scale*scale*scalestep

                drawall("12net prenms",rawimg,confidences)
                for confidence in confidences:
                        for idx, box in enumerate(confidence["boxes"]):
                                shift = calib12model.predict(confidence["frame"][:,box[1]:box[3],box[2]:box[4],:])[0]
                                shift = predToShift(shift, thresh = CALIB12THRESH)
                                confidence["boxes"][idx] = adjBB(box,shift)

                for confidence in confidences:
                        confidence["boxes"]  = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                        confidence["boxes"] = NMS(confidence["boxes"])
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
                                if conf < DETECT24THRESH:
                                        confidence["boxes"][idx][0] = (1-conf)*MAXCONF
                                else:
                                        confidence["boxes"][idx][0] = MAXCONF
                                        numremoved += 1
                        if numremoved == 0:
                                continue
                        confidence["boxes"] = confidence["boxes"][confidence["boxes"][:,0].argsort()]
                        confidence["boxes"] = confidence["boxes"][:-numremoved]

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

                for confidence in confidences:
                        confidence["boxes"] = NMS(confidence["boxes"])

                for confidence in confidences:
                        scale = confidence["scale"]
                        numremoved = 0
                        for idx, box in enumerate(confidence["boxes"]):
                                X1 = int(box[1]*scale)
                                Y1 = int(box[2]*scale)
                                X2 = int(box[3]*scale)
                                Y2 = int(box[4]*scale)
                                notface = {}
                                wind12 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L1SIZE,L1SIZE))
                                wind24 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L2SIZE,L2SIZE))
                                wind48 = resizetoshape(rawimg[X1:X2,Y1:Y2,:],(L3SIZE,L3SIZE))
                                notface['id'] = imgnum
                                imgnum += 1
                                notface['filename'] = os.path.join(backgroundfeeddir,backimg)
                                if notface['id']%trainvalidratio == 0:
                                        notface['filename'] = os.path.join(validdir,"notface","48-{}.jpg".format(notface['id']))
                                else:
                                        notface['filename'] = os.path.join(traindir,"notface","48-{}.jpg".format(notface['id']))

                                imsave(notface['filename'],wind48[0,:,:,:])

                                if notface['id']%trainvalidratio == 0:
                                        notface['filename'] = os.path.join(validdir,"notface","24-{}.jpg".format(notface['id']))
                                else:
                                        notface['filename'] = os.path.join(traindir,"notface","24-{}.jpg".format(notface['id']))

                                imsave(notface['filename'],wind24[0,:,:,:])

                                if notface['id']%trainvalidratio == 0:
                                        notface['filename'] = os.path.join(validdir,"notface","12-{}.jpg".format(notface['id']))
                                else:
                                        notface['filename'] = os.path.join(traindir,"notface","12-{}.jpg".format(notface['id']))

                                imsave(notface['filename'],wind12[0,:,:,:])
                                print(48,"detect-notface",notface)

def prepcalib(numface, datadir, trainvalidratio=10,size=12,startid = 0):
        traindir = datadir + "train/"
        validdir = datadir + "validation/"
        tagcsv = datasetdir + "umdfaces_batch1/umdfaces_batch1_ultraface.csv"
        with open(tagcsv, 'r') as csvfile:
                data = list(csv.reader(csvfile))    

        lengthofcsv = len(data)
        i = startid
        while i < numface + startid:
                position = random.randrange(1, lengthofcsv)
                face = umdcsvtobb(i,data[position])
                face,pert = perterb(face)
                img = imread(face['filename'])
                area = (int(face['xpos']),
                        int(face['ypos']), 
                        int(face['xpos']+face['w']),
                        int(face['ypos']+face['h']))

                if face['w'] < size or face['h'] < size:
                        continue

                if area[3] > img.shape[0] or area[2] > img.shape[1] or area[0] < 0 or area[1] < 0:
                        continue

                img = img[area[1]:area[3],area[0]:area[2],:]
                if face['w'] > face['h']:
                        img = pyramid_reduce(img,downscale=face['h']/float(size))
                        img = img[(img.shape[0]-size)/2:(img.shape[0]+size)/2 ,:,:]
                else:
                        img = pyramid_reduce(img,downscale=face['w']/float(size))
                        img = img[:,(img.shape[1]-size)/2:(img.shape[1]+size)/2,:]
                img = img[:size,:size,:]
                if face['id']%trainvalidratio == 0:
                        savedir = validdir
                else:
                        savedir = traindir
                if img.shape[0] != size or img.shape[1] != size:
                        raise Exception("blah")
                face['filename'] = os.path.join(savedir,"face","{}.jpg".format(face['id']))
                with open( os.path.join(savedir,"tag","{}.txt".format(face['id'])),"w+") as fh:
                        fh.write(",".join(map(str,pert))) 
                imsave(face['filename'],img)
                print(size,"calib: ",face)
                i += 1
if __name__ == "__main__":
        #prepdetect(4000,"data/faces/detect48/",trainvalidratio = 4,size = 48,startid=0)
        #prepdetect(4000,"data/faces/detect24/",trainvalidratio = 4,size = 24)
        #prepdetect(4000,"data/faces/detect12/",trainvalidratio = 4,size = 12,startid=4000)
        #prepbackground(10000,"data/faces/detect48/",trainvalidratio = 4,size = 48)
        #prepbackground24(10, "data/faces/detect24/", trainvalidratio=10,startid = 0)
        prepbackground48(10, "data/faces/detect48/", trainvalidratio=10,startid = 0)
        #prepbackground(3000,"data/faces/detect24/",trainvalidratio = 4,size = 24)
        #prepbackground(10000,"data/faces/detect12/",trainvalidratio = 4,size = 12,startid=3000)
        #prepcalib(4000,"data/faces/adj48/",trainvalidratio = 4,size=48)
        #prepcalib(4000,"data/faces/adj24/",trainvalidratio = 4,size=24)
        #prepcalib(4000,"data/faces/adj12/",trainvalidratio = 4,size=12,startid=4000)
        #bubbleup("data/faces/detect12/", "data/faces/detect24/", "data/faces/detect48/")
        #prepdetecthand(500,"data/hands/detect12/",trainvalidratio = 4,size = 12)
        #prepdetecthand(500,"data/hands/detect24/",trainvalidratio = 4,size = 24)
        #prepdetecthand(500,"data/hands/detect48/",trainvalidratio = 4,size = 48)
        #prephandcalib("data/hands/adj12/",trainvalidratio=4,size=12)
        #prephandcalib("data/hands/adj24/",trainvalidratio=4,size=24)
        #prephandcalib("data/hands/adj48/",trainvalidratio=4,size=48)
