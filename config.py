
#calibration perterbations
sn = [0.83,0.91,1.0,1.1,1.21]
xn = [-0.17,0,0.17]
yn = [-0.17,0,0.17]
adjclass = [",".join([str(si),str(xi),str(yi)]) for si in sn for xi in xn for yi in yn]
adjclassV = [(si,xi,yi) for si in sn for xi in xn for yi in yn]

CALIB12THRESH = 0.00
CALIB24THRESH = 0.00
CALIB48THRESH = 0.00
MINFACE = 60
L1SIZE = 12
L2SIZE = 24
L3SIZE = 48
DETECT12THRESH = 0.8
DETECT24THRESH = 0.8
DETECT48THRESH = 0.8
IOUTHRESH = 0.1
STEP = 4
MAXCONF = 65536
DISPLAY = True
