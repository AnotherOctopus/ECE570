import csv
from PIL import Image
import random
import os
import cv2
datadir = "data/"
datasetdir = "datasets/"
#calibration perterbations
sn = [0.83,0.91,1.0,1.1,1.21]
xn = [-0.17,0,0.17]
yn = [-0.17,0,0.17]
def umdcsvtobb(datapoint):
        ret = {
                "id":0,
                "filename":"",
                "xpos":0,
                "ypos":0,
                "w":0,
                "h":0
              }
        ret["id"] = int(datapoint[0])
        ret["filename"] = os.path.join(datasetdir,"umdfaces_batch1",datapoint[1])
        ret["xpos"] = float(datapoint[4])
        ret["ypos"] = float(datapoint[5])
        ret["w"] = float(datapoint[6])
        ret["h"] = float(datapoint[7])
        return ret
def randombb(imagesize):
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
        ret["w"] = random.uniform(0, imagesize[0] - ret["xpos"])
        ret["h"] = random.uniform(0, imagesize[1] - ret["ypos"])
        return ret
def displaybb(boundbox):
        image = cv2.imread(os.path.join(datasetdir,"umdfaces_batch1",boundbox["filename"]))
        if not image:
                print("image not found")
                return
        cv2.imshow(image)
        cv2.waitKey(30)

def prepdetect12(numface,numback,trainvalidratio=10):
        traindir = datadir + "detect12/train/"
        validdir = datadir + "detect12/validation/"
        tagcsv = datasetdir + "umdfaces_batch1/umdfaces_batch1_ultraface.csv"
        with open(tagcsv, 'r') as csvfile:
                data = list(csv.reader(csvfile))    

        lengthofcsv = len(data)
        for i in range(numface):
                position = random.randrange(0, lengthofcsv)
                face = umdcsvtobb(data[position])
                img = Image.open(face['filename'])
                area = (face['xpos'],
                        face['ypos'], 
                        face['xpos']+face['w'],
                        face['ypos']+face['h'])
                img = img.crop(area)
                img = img.resize((12,12))
                if face['id']%trainvalidratio == 0:
                        face['filename'] = os.path.join(validdir,"face","{}.jpg".format(face['id']))
                else:
                        face['filename'] = os.path.join(traindir,"face","{}.jpg".format(face['id']))
                img.save(face['filename'])
                print("12detect-face",face)

        backgroundfeeddir = datasetdir + "classroombackround/"
        bkgroundimgs = os.listdir(backgroundfeeddir)
        picnum = 0
        for i in range(numback):
                backimg = bkgroundimgs[picnum%len(bkgroundimgs)]
                img = Image.open(os.path.join(backgroundfeeddir,backimg))
                notface = randombb(img.size)
                notface['id'] = picnum
                notface['filename'] = os.path.join(backgroundfeeddir,backimg)
                area = (notface['xpos'],
                        notface['ypos'], 
                        notface['xpos']+notface['w'],
                        notface['ypos']+notface['h'])
                img = img.crop(area)
                img = img.resize((12,12))
                if notface['id']%trainvalidratio == 0:
                        notface['filename'] = os.path.join(validdir,"notface","{}.jpg".format(notface['id']))
                else:
                        notface['filename'] = os.path.join(traindir,"notface","{}.jpg".format(notface['id']))

                img.save(notface['filename'])
                print("12detect-notface",notface)
                picnum += 1
def perterb(BB):
        si = random.choice(sn)
        xi = random.choice(xn)
        yi = random.choice(yn)
        newBB = {}
        newBB["id"] = BB["id"]
        newBB["filename"] = BB["filename"]
        newBB["xpos"] = BB["xpos"] - xi
        newBB["ypos"] = BB["ypos"] - yi
        newBB["w"] = BB["w"]/si
        newBB["h"] = BB["h"]/si
        return newBB,(si,xi,yi)

def prepcalib12(numface,trainvalidratio=10):
        traindir = datadir + "adj12/train/"
        validdir = datadir + "adj12/validation/"
        tagcsv = datasetdir + "umdfaces_batch1/umdfaces_batch1_ultraface.csv"
        with open(tagcsv, 'r') as csvfile:
                data = list(csv.reader(csvfile))    

        lengthofcsv = len(data)
        for i in range(numface):
                position = random.randrange(0, lengthofcsv)
                face = umdcsvtobb(data[position])
                face,pert = perterb(face)
                img = Image.open(face['filename'])
                area = (face['xpos'],
                        face['ypos'], 
                        face['xpos']+face['w'],
                        face['ypos']+face['h'])
                img = img.crop(area)
                img = img.resize((12,12))
                if face['id']%trainvalidratio == 0:
                        savedir = validdir
                else:
                        savedir = traindir
                face['filename'] = os.path.join(savedir,"face","{}.jpg".format(face['id']))
                with open( os.path.join(savedir,"tag","{}.txt".format(face['id'])),"w+") as fh:
                        fh.write(",".join(map(str,pert))) 
                img.save(face['filename'])
                print("12calib: ",face)
                
if __name__ == "__main__":
        prepcalib12(2000)
        prepdetect12(10000,10000)
