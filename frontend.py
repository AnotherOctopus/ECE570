from detect12model import detect12
from calib12model import calib12
from detect24model import detect24
from calib24model import calib24
from detect48model import detect48
from calib48model import calib48
from detectdense48 import dense48


if __name__ == "__main__":
    d12 = detect12()
    c12 = calib12()
    d24 = detect24()
    c24 = calib24()
    d48 = detect48()
    c48 = calib48()
    dd48 = dense48()
    dd48.compile()
    #print dd48.model.summary()

    

    d12.compile()
    c12.compile()
    d24.compile()
    c24.compile()
    d48.compile()
    c48.compile()

    #d12.train()
    c12.train()
    #d24.train()
    c24.train()
    #d48.train()
    c48.train()
