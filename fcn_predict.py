# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:06:28 2018

@author: Dell
"""
import numpy as np
def img2xyz(xr=[0.25,0.01,24],yr=[-0.2,0.013,32],zr=[3.07,0.01,18],imgs=[]):
     xyz=[]
     for i in range(0,len(imgs)):
        mvalue=np.amax(imgs[i])
        idx=np.where(imgs[i]==mvalue)
        xyz=xyz+[[xr[0]+xr[1]*idx[0][0],yr[0]+yr[1]*idx[1][0],zr[0]+zr[1]*idx[2][0],mvalue,i,idx[0][0],idx[1][0],idx[2][0]]]
     return xyz;
      
def output_result1(r=[],imgs=[],namout='test_xyz.txt'):
     xyz=img2xyz(xr=r[0],yr=r[1],zr=r[2],imgs=imgs)
     with open(namout,'w') as f:
         for i in xyz:
             f.write("{:.6f} {:.6f} {:.6f} {:.6f}\n".format(i[0],i[1],i[2],i[3]))

from keras.models import *
import sgydata
import scipy.io as sio

def predict():
    print('load testing samples.')
    r=[[34.9751,0.0315,80],[-98.4047,0.0225,128],[0.000,0.4,30]]
    wave_test,loca_true=sgydata.load_sgylist_xyz1(shuffle='false',sgylist=['./waveform_data/','testing_samples.txt'],
                                sgyr=[0,-1,1],xr=r[0],yr=r[1],zr=r[2],r=0.05,shiftdata=[list(range(20,50))+list(range(-200,-20)),0])
    loca_true=np.reshape(loca_true,(len(loca_true),80,128,30)) 
    
    print('save true labels (true location images).')
    #sio.savemat('test_truelabel.mat', {'loca_true':loca_true})
    
    print('load trained network model.')
    model=load_model('./FCNloca.hdf5')
#    model=load_model('D:/cnnloca/shift_filter/model20_50_-200_-20a1/unet.hdf5')

    
    print('location prediction.')
    loca_predict = model.predict(wave_test, batch_size=1, verbose=1)
    
    print('save predicted location images.')
    sio.savemat('test_predictedlabel.mat', {'loca_predict':loca_predict,'xyzrange':r,'wave_test':wave_test})

    print('output location results.')
    output_result1(r=r,imgs=loca_predict,namout='test_xyz.txt')
    print('end predict')

if __name__ == '__main__':
    predict()

