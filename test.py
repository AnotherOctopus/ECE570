from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from scipy.misc import imread, imsave

# returns a compiled model
# identical to the previous one
def runframe(model,frame,wsize):
        step = 4
        print frame.shape
        maxX = frame.shape[1]
        maxY = frame.shape[2]
        data = np.empty((frame.shape[0],int((maxX-wsize)/step)+1,int((maxY-wsize)/step)+1))
        for xi,x in enumerate(range(wsize/2,maxX - wsize/2,step)):
                for yi,y in enumerate(range(wsize/2,maxY - wsize/2,step)):
                        lowwindx = x - wsize/2
                        lowwindy = y - wsize/2
                        maxwindx = x + wsize/2
                        maxwindy = y + wsize/2
                        print (data.shape)
                        print (xi,yi)
                        print (x,y)
                        print (lowwindx,lowwindy,maxwindx,maxwindy)
                        data[:,xi,yi] = model.predict(frame[:,lowwindx:maxwindx,lowwindy:maxwindy,:]).reshape(99)
        return data
"""
testfile = [
        "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/detect12/train/face/118.jpg",
        "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/detect12/train/notface/41.jpg"
]
model = load_model('first_try.h5')
for fi in testfile:
        img = imread(fi)
        img = img[np.newaxis,...]
        print model.predict(img)
"""
if __name__ == "__main__":
        testfile = "/home/cephalopodoverlord/DroneProject/Charles570/ECE570/data/test/bill_gates_0017.jpg"
        model = load_model('first_try.h5')
        rawimg = imread(testfile,mode='RGB')
        cv2.imshow("image",rawimg)
        minscale = 0.01
        scalestep = 0.01
        scale = 1
        i = 0
        X = rawimg.shape[0]
        Y = rawimg.shape[1]
        data = np.empty((int((scale-minscale)/scalestep), X,Y,3 ))
        while scale > minscale:
                img = rawimg.copy()
                img.resize((int(X*scale),int(Y*scale),3))
                img = np.pad(img,((0,X-img.shape[0]),(0,Y-img.shape[1]),(0,0)),'constant' )
                data[i,:,:,:] = img
                i+=1
                scale -= scalestep
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                        break

        confidence = runframe(model,data,12)
        np.save("confidence.npy",confidence)
