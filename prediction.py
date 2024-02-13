import os
import xarray as xr
from keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt
import utils
plt.style.use('ggplot')
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
dir=os.path.dirname(__file__)
parser = argparse.ArgumentParser()
parser.add_argument('-l','--list', nargs='+', help='choose variables from q700, q850, spr, oro, sw1 \\n example usage: python train_model.py -l q700 spr', required=False)
parser.add_argument('-d', '--date', help='specify the date, format=YYYY-MM-DD', required=True)
parser.add_argument('-n', '--num_forecasts', help='specify the date, format=YYYY-MM-DD')
args = parser.parse_args()
date = args.date
num_forecasts=args.num_forecasts
if not num_forecasts:
  num_forecasts=1
EXT_VARIABLES=args.list
if EXT_VARIABLES==None:
  EXT_VARIABLES=[]
EXT_VARIABLES = sorted(EXT_VARIABLES)
LEADTIME=1
# Load Data
imd_z=xr.open_dataset(os.path.join(dir, "data/rf_allvar_1979-2019.nc"))

LEADTIME=1


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


# load model
try:
    var_string='_'.join(EXT_VARIABLES)
except:
    var_string=''

seq_path = os.path.join(dir, 'models')

print(os.path.join(seq_path, f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}.h5'))

try:
    seq = load_model(os.path.join(seq_path, f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}.h5'))
    print('Model file found')
except Exception as e:
    print(e)

print(seq.summary())

T=imd_z.max(('time', 'lat', 'lon')).compute()


low = 0
high = 55
step = 5
zstr=[]
for i in range(low, high+step, step):
  zstr.append(str(i))
print(zstr)
zint=[]
for i in range(low, high+step, step):
  zint.append((i))
print(zint)
lon_low = 14
lon_high = 135
lon_step = 20
lon_zstr=[]
for i in range(lon_low, lon_high, lon_step):
  lon_zstr.append(str(66.5+i*.25))
print(lon_zstr)
lon_zint=[]
for i in range(lon_low, lon_high, lon_step):
  lon_zint.append((i))
print(lon_zint)
lat_low = 14
lat_high = 120
lat_step = 20
lat_zstr=[]
for i in range(lat_low, lat_high, lat_step):
  lat_zstr.append(str(6.5+i*.25))
print(lat_zstr)
lat_zint=[]
for i in range(lat_low, lat_high, lat_step):
  lat_zint.append((i))
print(lat_zint)

print(date)
first_d= np.datetime64(date) - np.timedelta64(4, 'D')
print(first_d)
dat=imd_z.sel(time=slice(first_d,date))
print((imd_z.sel(time=slice('2011-08-02','2011-08-07'))['time'].values))
dg_test=datagen(dat, batch_size=1, leadtime=LEADTIME, in_frames=5, max=T, ext_variables=EXT_VARIABLES, mode="forecast") #to be used as input for prediction

for i in range(1, int(num_forecasts)+1):
  # Model Prediction
  if i!=1:
    next_day = dat['time'].values[1:][-1] + np.timedelta64(1, 'D')
    print(next_day)
    extended_dates = np.append(dat['time'].values[1:], next_day)
    ds= xr.Dataset({
    "rain":(["time", "lat", "lon"], np.vstack((dat['rain'].values[1:],yhat[0]))),
    "q700":(["time", "lat", "lon"], np.vstack((dat["q700"].values[1:], dat["q700"].values[-1][np.newaxis, :, :]))),
    "q850":(["time", "lat", "lon"], np.vstack((dat["q850"].values[1:], dat["q850"].values[-1][np.newaxis, :, :]))),
    "sp":(["time", "lat", "lon"], np.vstack((dat["sp"].values[1:], dat["sp"].values[-1][np.newaxis, :, :]))),
    "z":(["time", "lat", "lon"], dat["z"].values),
    "swvl1":(["time", "lat", "lon"], np.vstack((dat["swvl1"].values[1:], dat["swvl1"].values[-1][np.newaxis, :, :])))
    },
        coords={
            "time":extended_dates,
            "lat":np.arange(6.5,38.75, 0.25),
            "lon":np.arange(66.5,100.25, 0.25)
        }
    )
    dg_test=datagen(ds, batch_size=1, leadtime=LEADTIME, in_frames=5, max=T, ext_variables=EXT_VARIABLES, mode="forecast") 
    dat=ds
  yhat=seq.predict(dg_test)

  # Convert back to Normal Space

  try:
      T=T.rain.values
  except:
      pass

  y=np.array(dat.rain.values[0])
  undefpts2=np.where(y!=y)
  y[undefpts2]=-1
  validpts2 = np.where(y>=0)
  y[validpts2] = np.exp(y[validpts2])#
  y[validpts2] = np.power(y[validpts2],7)
  y[undefpts2] = 0#


  # Convert back into normal space
  vals=y.reshape(1,1,129,135)
  a = np.where(vals>=1)
  b = np.where(vals==0)

  yhat[:,:,:,:][a] = np.power(yhat[:,:,:,:][a],1/7)
  yhat[:,:,:,:][a] = np.log(yhat[:,:,:,:][a])
  yhat[:,:,:,:][a] = yhat[:,:,:,:][a]*(T)
  c = np.where(yhat[:,:,:,:]<=0)
  yhat[:,:,:,:][c] = 0
  yhat[:,:,:,:][b] = np.nan


  plt.imshow(yhat[0,0,:,:],cmap='nipy_spectral_r',vmin=0,vmax=50)  # set vmin and vmax as arguments in imshow to adjust for scale
  ax1 = plt.axes()
  plt.rcParams['font.family'] = 'sans-serif'
  plt.rcParams['font.sans-serif'] = 'Times New Roman'
  ax1.set_ylim(0,129)
  ax1.set_xticks(lon_zint)
  ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
  ax1.set_yticks(lat_zint)
  ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
  plt.title('Lead Day %i Prediction '%i)
  plt.colorbar(ticks=zint).ax.set_yticklabels( zstr,weight='bold', fontsize=14)
  plt.grid()
  plt.savefig('./forecast_plots/'+'forecast_ld%d.png'% i,dpi=400,format='png',figsize=(5, 9))
  plt.clf()