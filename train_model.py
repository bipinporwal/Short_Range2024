import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K
from keras.models import Sequential
from keras.layers import Conv3D, Conv2D
from keras.layers import ConvLSTM2D
#from tensorflow.keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Lambda
from keras import optimizers
from keras.layers import Conv2DTranspose,TimeDistributed
import keras
import xarray as xr
from keras import callbacks
import argparse
import os
import utils

import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

class datagen(keras.utils.Sequence):
  def __init__(self, dataset, batch_size, leadtime, in_frames, ext_variables = [],data_format = 'channels_first', \
                normalization='max_min',mean=None, std=None, max=None, min=None, mode='trainortest'):
    '''
    Assuming data format= channels first
    '''
    self.dataset = dataset
    self.batch_size = batch_size
    self.leadtime = leadtime
    self.in_frames = in_frames
    self.data_format = data_format
    self.normalization=normalization
    self.ext_variables=ext_variables
    self.mode=mode
    """
    assert(self.dataset.dims['time']-self.leadtime > 0) # only proceed if enough samples are there
    print(self.dataset.dims['time']-self.leadtime > 0)
    """
    if mode=='trainortest':
      self.samples = self.dataset.dims['time'] - (self.in_frames + self.leadtime-1) # effective number of samples
    elif mode=="forecast":
      self.samples = self.dataset.dims['time'] - (self.in_frames + self.leadtime-2)
      
    self.on_epoch_end()
    if self.normalization=='max_min':
        self.max = self.dataset.max(('time', 'lat', 'lon')).compute() if max is None else max
        self.min = self.dataset.min(('time', 'lat', 'lon')).compute() if min is None else min
        self.dataset=(self.dataset - self.min) / (self.max - self.min)
    
    """
    if self.normalization=='std':
      self.mean = self.dataset.mean(('time', 'lat', 'lon')).compute() if mean is None else mean #
      self.std = self.dataset.std('time').mean(('lat', 'lon')).compute() if std is None else std
      self.dataset = (self.dataset - self.mean) / self.std
    
    elif self.normalization=='max_min':
      self.max = self.dataset.max(('time', 'lat', 'lon')).compute() if max is None else max
      self.min = self.dataset.min(('time', 'lat', 'lon')).compute() if min is None else min
      self.dataset = 2 * ((self.dataset - self.min) / (self.max - self.min)) + 1
    """
    self.dataset.load()
  
  def on_epoch_end(self):
    'Store an np.ndarray of numbers of length of total number of samples'
    self.indexes = np.arange(self.samples)
  
  def __len__(self):
    'Return the number of batches per epoch that will be supplied depending on total number of samples'
    return int(np.floor(self.samples / self.batch_size))
  
  def __getitem__(self, index):
    'Generate (one) index-th batch of data'
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    X, y = self.core_generator(indexes)
    return X, y
  
  def core_generator(self, indexes):
    '''
    Output X, y of shape (batch_size, timestep, x, y
    '''
    X = []
    y = []
    q7=[]
    q85=[]
    spr=[]
    oro=[]
    sw1=[]
    


    # make supervised splits
    for i in range(len(indexes)):
      X.append(self.dataset['rain'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      if self.mode=="trainortest":
        y.append(self.dataset['rain'].isel(time=indexes[i] + (self.in_frames + self.leadtime - 1)).values)
      q7.append(self.dataset['q700'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      q85.append(self.dataset['q850'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      spr.append(self.dataset['sp'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      oro.append(self.dataset['z'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      sw1.append(self.dataset['swvl1'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)



        #No need to rollaxis for y since appending lat x lon 
    X=np.array(X)
    undefpts = np.where(X!=X)
    X[undefpts]=-1
    validpts = np.where(X>=0)
    X[validpts] = np.exp(X[validpts])#
    X[validpts] = np.power(X[validpts],7)
    X[undefpts] = 0#
    if self.mode=="trainortest":
      y=np.array(y)
      undefpts2=np.where(y!=y)
      y[undefpts2]=-1
      validpts2 = np.where(y>=0)
      y[validpts2] = np.exp(y[validpts2])#
      y[validpts2] = np.power(y[validpts2],7)
      y[undefpts2] = 0#
    q7=np.array(q7)
    q85=np.array(q85)
    spr=np.array(spr)
    oro=np.array(oro)
    sw1=np.array(sw1)
    
    if self.data_format == 'channels_first':
      im= X[:, :, np.newaxis, :, :]
      sph7=q7[:, :, np.newaxis, :, :]
      sph85=q85[:, :, np.newaxis, :, :]
      spr=spr[:, :, np.newaxis, :, :]
      o=oro[:, :, np.newaxis, :, :]
      sw1=sw1[:, :, np.newaxis, :, :]

      vars={'q700':sph7,'q850':sph85,'spr':spr, 'oro':o, 'sw1':sw1}
      if len(self.ext_variables)==0:
        data=im
      elif len(self.ext_variables)==5:
        data=np.concatenate([im, sph7, sph85, spr, o, sw1], axis=2)
      else:
        for ind, var in enumerate(self.ext_variables):
          if ind==0:
            data=np.concatenate([im, vars[var]], axis=2)
          else:
            data=np.concatenate([data,vars[var]], axis=2)
            

      if self.mode=="trainortest":
        X, y = data, y[:, np.newaxis, :, :]
      else:
        X, y = data, y
    """
    elif self.data_format == 'channels_last':
      X, y = X[:, :, :, :, np.newaxis], y[:, :, :, np.newaxis]
    """
    return X, y

def validate_model_exists(ext_variables, leadtime):
  try:
    var_string='_'.join(ext_variables)
  except:
    var_string=''
  return f'ld{LEADTIME}_{len(ext_variables)}extvars_{var_string}.h5' in os.listdir(os.path.join(dir,'models/'))
  
#Train the model...
def train_model(model,data, ext_variables, LEADTIME):
  try:
    var_string='_'.join(ext_variables)
  except:
    var_string=''
  logdir = os.path.join(dir, f'logs/ld{LEADTIME}_{len(ext_variables)}extvars_{var_string}')
  epochs={i: 1000+200*i for i in range(6)} 
  nepoch=epochs[len(ext_variables)]
  
  T=data.max(('time', 'lat', 'lon')).compute()
  dg_train=datagen(data.sel(time=slice('1979','2008')), batch_size=32, leadtime=LEADTIME, in_frames=5, max=T, ext_variables=ext_variables)
  dg_valid=datagen(data.sel(time=slice('2009','2010')), batch_size=32, leadtime=LEADTIME, in_frames=5, max=T, ext_variables=ext_variables)

  # Tensor board for visualization

  tbcallback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=False)


  model.fit(dg_train,epochs=nepoch,validation_data=dg_valid,workers=6,
          use_multiprocessing=False,callbacks=[tbcallback])
##-- Changed by Harshal on 4 July 2023. --- Model is being save in the current directory.

#  model.save(os.path.join("/home/hpcs_rnd/Short_range_Namit/train_model", f'ld{LEADTIME}_{len(ext_variables)}vars_{var_string}.h5'))
  model.save(os.path.join(dir, f'models/ld{LEADTIME}_{len(ext_variables)}extvars_{var_string}.h5'))


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
  dir=os.path.dirname(__file__)
  parser = argparse.ArgumentParser()
  parser.add_argument('-l','--list', nargs='+', help='choose variables from q700, q850, spr, oro, sw1 \\n example usage: python train_model.py -l q700 spr', required=False)
  args = parser.parse_args()
  EXT_VARIABLES=args.list
  if EXT_VARIABLES==None:
    EXT_VARIABLES=[]
  EXT_VARIABLES = sorted(EXT_VARIABLES)
  LEADTIME=1
  
  if validate_model_exists(EXT_VARIABLES, LEADTIME):
    print('trained model already exists in models folder')
  else:
    #load dataset..
    imd_z=xr.open_dataset(os.path.join(dir,"data/rf_allvar_1979-2019.nc"))
    #Defining the model
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=4,kernel_size=(3,3),padding='same',input_shape=(None,len(EXT_VARIABLES)+1,129,135),return_sequences=True,data_format='channels_first'))
    #seq.add(TimeDistributed(Conv2DTranspose(filters=20,kernel_size=(3,3),padding='valid',activation='relu')))
    seq.add(ConvLSTM2D(filters=8,kernel_size=(3,3),padding='same',return_sequences=True,data_format='channels_first'))
    #seq.add(TimeDistributed(Conv2DTranspose(filters=40,kernel_size=(3,3),padding='valid',activation='relu')))
    seq.add(ConvLSTM2D(filters=8,kernel_size=(3,3),padding='same',return_sequences=True,data_format='channels_first'))
    #seq.add(TimeDistributed(Conv2DTranspose(filters=60,kernel_size=(3,3),padding='valid',activation='relu')))
    seq.add(ConvLSTM2D(filters=16,kernel_size=(3,3),padding='same',return_sequences=True,data_format='channels_first'))
    seq.add(ConvLSTM2D(filters=16,kernel_size=(3,3),padding='same',return_sequences=False,data_format='channels_first'))
    seq.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same',data_format='channels_first'))
    seq.add(Conv2D(filters=1,kernel_size=(3,3),activation='relu',padding='same',data_format='channels_first'))
    #seq.add(Conv3DTranspose(filters=1,kernel_size=(1,1,1),activation='relu',padding='valid'))
    Adam = optimizers.Adam(lr=10**-4)
    seq.compile(loss='mean_squared_error',optimizer=Adam,metrics=['accuracy','mae'])
    train_model(seq, imd_z, EXT_VARIABLES, LEADTIME)
