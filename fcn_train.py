import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import sgydata
#np.random.seed(5)
np.random.seed(15)

class FCNloca(object):

  def __init__(self, img_rows = 30, img_cols = 2048):

    self.img_rows = img_rows
    self.img_cols = img_cols

  def load_data(self):
    r=[[34.9751,0.0315,80],[-98.4047,0.0225,128],[0.000,0.4,30]]
    wave_train,loca_train=sgydata.load_sgylist_xyz1(sgylist=['./waveform_data/','training_samples.txt'],
            sgyr=[0,-1,1],xr=r[0],yr=r[1],zr=r[2],r=0.05,
            shiftdata=[list(range(20,50))+list(range(-200,-20)),1])
    loca_train=np.reshape(loca_train,(len(loca_train),80,128,30)) 
    print('end load_data()...')
    return wave_train, loca_train

  def get_network(self):

    inputs = Input((self.img_rows, self.img_cols,3))
    
    conv1 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print("conv1 shape:",conv1.shape)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(1, 4))(conv1)
    print("pool1 shape:",pool1.shape)

    conv2 = Conv2D(128,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print("conv2 shape:",conv2.shape)
    conv2 = Conv2D(128,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(1, 4))(conv2)
    print("pool2 shape:",pool2.shape)

    conv3 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print("conv3 shape:",conv3.shape)
    conv3 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 4))(conv3)
    print("pool3 shape:",pool3.shape)

    conv4 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(3, 4))(drop4)

    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(drop5))
#    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(512,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    conv7 = Conv2D(256,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    conv8 = Conv2D(64,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(64,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv9 = Conv2D(32,kernel_size=(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv10 = Conv2D(30, 1, activation = 'sigmoid')(conv9)
    

    model = Model(input = inputs, output = conv10)
    print('conv10:', conv10.shape)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    model.compile(optimizer = 'sgd', loss = losses.mean_squared_error)
    return model


  def train(self):

    print("loading data")
    wave_train, loca_train = self.load_data()
    print("loading data done")
    model = self.get_network()
    print("got network")

    model_checkpoint = ModelCheckpoint('FCNloca.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    print('Fitting model...')
    hist=model.fit(wave_train, loca_train, batch_size=4, nb_epoch=200, verbose=1,validation_split=0.1, shuffle=True, callbacks=[model_checkpoint])

    #model.save('test.h5')
    f=open('FCNloca.log','w')
    f.write(str(hist.history))
    f.close()


if __name__ == '__main__':
	fcnloca = FCNloca()
	fcnloca.train()



