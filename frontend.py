from detect12model import detect12
from calib12model import calib12
from detect24model import detect24
from calib24model import calib24
from detect48model import detect48
from calib48model import calib48
from detectdense48 import dense48
from PIL import Image
import matplotlib.pyplot as plt
class plotmodel():
    def __init__(self):
        self.figcnt = 0
    def displayhist(self,name,hist):
        plt.figure(self.figcnt)
        try:
            plt.plot(hist['val_acc'])
        except:
            plt.plot(hist['val_categorical_accuracy'])
        plt.title('{} Model Validation accuracy'.format(name))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        #plt.legend(['Train', 'Test'], loc='upper left')
        self.figcnt += 1

if __name__ == "__main__":
    d12 = detect12()
    c12 = calib12()
    d24 = detect24()
    c24 = calib24()
    d48 = detect48()
    c48 = calib48()
    dd48 = dense48()


    dd48.compile()
    d12.compile()
    c12.compile()
    d24.compile()
    c24.compile()
    d48.compile()
    c48.compile()
    #dd48.train()
    #df12hist = d12.train("facedetect12.h5","data/faces/detect12/train","data/faces/detect12/validation")
    cf12hist = c12.train("facecalib12.h5","data/faces/adj12/train","data/faces/adj12/validation")
    #df24hist = d24.train("facedetect24.h5","data/faces/detect24/train","data/faces/detect24/validation")
    cf24hist = c24.train("facecalib24.h5","data/faces/adj24/train","data/faces/adj12/validation")
    #df48hist = d48.train("facedetect48.h5","data/faces/detect48/train","data/faces/detect48/validation")
    cf48hist = c48.train("facecalib48.h5","data/faces/adj48/train","data/faces/adj12/validation")


    #dh12hist = d12.train("handdetect12.h5","data/hands/detect12/train","data/hands/detect12/valid",tags=["hand","nothand"])
    ch12hist = c12.train("handcalib12.h5","data/hands/adj12/train","data/hands/adj12/valid",tags=["hand","nothand"])
    #dh24hist = d24.train("handdetect24.h5","data/hands/detect24/train","data/hands/detect24/valid",tags=["hand","nothand"])
    ch24hist = c24.train("handcalib24.h5","data/hands/adj24/train","data/hands/adj12/valid",tags=["hand","nothand"])
    #dh48hist = d48.train("handdetect48.h5","data/hands/detect48/train","data/hands/detect48/valid",tags=["hand","nothand"])
    ch48hist = c48.train("handcalib48.h5","data/hands/adj48/train","data/hands/adj12/valid",tags=["hand","nothand"])

    p = plotmodel()
    p.displayhist("detect face 12",df12hist.history)
    p.displayhist("detect hand 12",dh12hist.history)
 #   p.displayhist("calib 12",c12hist.history)
    p.displayhist("detect face 24",df24hist.history)
    p.displayhist("detect hand 24",dh24hist.history)
 #  p.displayhist("calib 24",c24hist.history)
    p.displayhist("detect face 48",df48hist.history)
    p.displayhist("detect hand 48",dh48hist.history)
 #  p.displayhist("calib 48",c48hist.history)
    #dd48hist = dd48.train()

    plt.show()
