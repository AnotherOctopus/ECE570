import csv
import random
import os
import cv2
from scipy.misc import imread,imsave
from skimage.transform import pyramid_reduce
from config import *

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
def randombb(imagesize,window):
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
        ret["xpos"] = random.uniform(0,imagesize[0])
        ret["ypos"] = random.uniform(0,imagesize[1])
        while ret["xpos"] + window > imagesize[0] or ret["ypos"] + window > imagesize[1]:
                ret["xpos"] = random.uniform(0,imagesize[0])
                ret["ypos"] = random.uniform(0,imagesize[1])
        size = random.uniform(window, imagesize[0] - max(ret["xpos"],ret["ypos"]))
        ret["w"] = size
        ret["h"] = size
        return ret
def displaybb(boundbox):
        image = cv2.imread(os.path.join(datasetdir,"umdfaces_batch1",boundbox["filename"]))
        if not image:
                print("image not found")
                return
        cv2.imshow(image)
        cv2.waitKey(30)

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

                try:
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
                except:
                        pass

def prepbackground(numback,datadir,trainvalidratio=10,size = 12,startid = 0):
        traindir = datadir + "train/"
        validdir = datadir + "validation/"
        backgroundfeeddir = datasetdir + "classroombackround/"
        bkgroundimgs = os.listdir(backgroundfeeddir)
        for i in range(startid,numback + startid):
                backimg = bkgroundimgs[i%len(bkgroundimgs)]
                img = imread(os.path.join(backgroundfeeddir,backimg))
                notface = randombb(img.shape,size)
                notface['id'] = i
                notface['filename'] = os.path.join(backgroundfeeddir,backimg)
                try:
                        if size == 12:
                                area = (int(notface['xpos']),
                                        int(notface['ypos']), 
                                        int(notface['xpos']+12),
                                        int(notface['ypos']+12))
                                img = img[area[1]:area[3],area[0]:area[2],:]
                        else:
                                area = (int(notface['xpos']),
                                        int(notface['ypos']), 
                                        int(notface['xpos']+notface['w']),
                                        int(notface['ypos']+notface['h']))
                                img = img[area[1]:area[3],area[0]:area[2],:]
                                img = pyramid_reduce(img,downscale=notface['h']/float(size))
                                img = img[:size,:size,:]

                        if notface['id']%trainvalidratio == 0:
                                notface['filename'] = os.path.join(validdir,"notface","{}.jpg".format(notface['id']))
                        else:
                                notface['filename'] = os.path.join(traindir,"notface","{}.jpg".format(notface['id']))

                        print img.shape
                        imsave(notface['filename'],img)
                        print(size,"detect-notface",notface)
                except:
                        pass
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

def prepcalib(numface, datadir, trainvalidratio=10,size=12,startid = 0):
        traindir = datadir + "train/"
        validdir = datadir + "validation/"
        tagcsv = datasetdir + "umdfaces_batch1/umdfaces_batch1_ultraface.csv"
        with open(tagcsv, 'r') as csvfile:
                data = list(csv.reader(csvfile))    

        lengthofcsv = len(data)
        for i in range(startid,numface + startid):
                try:
                        position = random.randrange(1, lengthofcsv)
                        face = umdcsvtobb(i,data[position])
                        face,pert = perterb(face)
                        img = imread(face['filename'])
                        area = (int(face['xpos']),
                                int(face['ypos']), 
                                int(face['xpos']+face['w']),
                                int(face['ypos']+face['h']))
                        img = img[area[1]:area[3],area[0]:area[2],:]
                        if face['w'] > face['h']:
                                img = pyramid_reduce(img,downscale=face['h']/float(size))
                                img = img[(img.shape[0]-size)/2:(img.shape[0]+size)/2 ,:,:]
                        else:
                                img = pyramid_reduce(img,downscale=face['w']/float(size))
                                img = img[:,(img.shape[0]-size)/2:(img.shape[0]+size)/2,:]
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
                except:
                        pass
                
if __name__ == "__main__":
        prepdetect(100000,"data/detect48/",trainvalidratio = 4,size = 48)
        prepdetect(100000,"data/detect24/",trainvalidratio = 4,size = 24)
        prepdetect(100000,"data/detect12/",trainvalidratio = 4,size = 12,startid=20000)
        prepbackground(150000,"data/detect48/",trainvalidratio = 4,size = 48)
        prepbackground(150000,"data/detect24/",trainvalidratio = 4,size = 24)
        prepbackground(150000,"data/detect12/",trainvalidratio = 4,size = 12,startid=15000)
        prepcalib(100000,"data/adj48/",trainvalidratio = 4,size=48)
        prepcalib(100000,"data/adj24/",trainvalidratio = 4,size=24)
        prepcalib(100000,"data/adj12/",trainvalidratio = 4,size=12)
