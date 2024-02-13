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
args = parser.parse_args()
EXT_VARIABLES=args.list
if EXT_VARIABLES==None:
  EXT_VARIABLES=[]
EXT_VARIABLES = sorted(EXT_VARIABLES)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
imd_z=xr.open_dataset(os.path.join(dir,"data/rf_allvar_1979-2019.nc"))
#select from q700, q850, spr, oro, sw1
#EXT_VARIABLES=sorted(['q700','q850','spr','oro','sw1'])
LEADTIME=1


class datagen(keras.utils.Sequence):
  def __init__(self, dataset, batch_size, leadtime, in_frames, ext_variables = [],data_format = 'channels_first', \
                normalization='max_min',mean=None, std=None, max=None, min=None):
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
  
    """
    assert(self.dataset.dims['time']-self.leadtime > 0) # only proceed if enough samples are there
    print(self.dataset.dims['time']-self.leadtime > 0)
    """
    self.samples = self.dataset.dims['time'] - (self.in_frames + self.leadtime) # effective number of samples

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
      y.append(self.dataset['rain'].isel(time=indexes[i] + (self.in_frames + self.leadtime - 1)).values)
      q7.append(self.dataset['q700'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      q85.append(self.dataset['q850'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      spr.append(self.dataset['sp'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      oro.append(self.dataset['z'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)
      sw1.append(self.dataset['swvl1'].isel(time=slice(indexes[i], indexes[i] + self.in_frames)).values)



        #No need to rollaxis for y since appending lat x lon 
    X=np.array(X)
    y=np.array(y)
    undefpts = np.where(X!=X)
    X[undefpts]=-1
    validpts = np.where(X>=0)
    X[validpts] = np.exp(X[validpts])#
    X[validpts] = np.power(X[validpts],7)
    X[undefpts] = 0#
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
            

      X, y = data, y[:, np.newaxis, :, :]
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
dg_test=datagen(imd_z.sel(time=slice('2010-12-27','2016-01-01')), batch_size=1, leadtime=LEADTIME, in_frames=5, max=T, ext_variables=EXT_VARIABLES)
yhat=seq.predict(dg_test)

y1=[]
for x in dg_test.indexes:
  y1.append(dg_test[x][1][0])
y1=np.array(y1)


print(yhat.shape)
print(y1.shape)

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

T=imd_z.max(('time', 'lat', 'lon')).compute().rain.values
# Convert back into normal space

a = np.where((y1[:,:,:,:])>=1)
b = np.where((y1[:,:,:,:])==0)

yhat[:,:,:,:][a] = np.power(yhat[:,:,:,:][a],1/7)
yhat[:,:,:,:][a] = np.log(yhat[:,:,:,:][a])
yhat[:,:,:,:][a] = yhat[:,:,:,:][a]*(T)
c = np.where(yhat[:,:,:,:]<=0)
yhat[:,:,:,:][c] = 0
yhat[:,:,:,:][b] = np.nan


y1[:,:,:,:][a] = np.power(y1[:,:,:,:][a],1/7)
y1[:,:,:,:][a] =np.log(y1[:,:,:,:][a])
y1[:,:,:,:][a] = y1[:,:,:,:][a]*(T)
y1[:,:,:,:][b] = np.nan

path=os.path.join(dir, f'test_plots/{len(EXT_VARIABLES)}extvars_{var_string}/')
isExist = os.path.exists(path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(path)

M = np.mean(y1[:,:,:,:],axis=0)
m = np.mean(yhat[:,:,:,:],axis=0)
print(np.max(M))
print(np.max(m))
#####Computing Normal Correlation#####
m1 = np.mean(y1[:,:,:,:])
m2 = np.mean(yhat[:,:,:,:])

crf = []

for j in range(1):
  cv1 = np.zeros((129,135),dtype=float)
  cv2 = np.zeros((129,135),dtype=float)
  cv3 = np.zeros((129,135),dtype=float)
  cv  = np.zeros((129,135),dtype=float)
  for i in range(0,y1.shape[0]):
    aa = y1[i,j,:,:]-M[j,:,:]
    bb = yhat[i,j,:,:]-m[j,:,:]
    cv1+=np.multiply(aa,bb)
    cv2+=np.power(aa,2)
    cv3+=np.power(bb,2)
  c = np.multiply(np.sqrt(cv2),np.sqrt(cv3))
  cv = np.divide(cv1,c,where=(c!=0))
  cv[np.where(c==0)] = 0 #aim is to make correlation = nan where c =0
  crf.append(cv)


crf = np.array(crf)


q = np.where(crf==0)
crf[q]=np.nan

np.save(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_crf_imdrf', crf)

plt.imshow(crf[0],interpolation='none',cmap='nipy_spectral',vmin=0, vmax=1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
ax1 = plt.axes()
ax1.set_ylim(0,129)
ax1.set_xticks(lon_zint)
ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
ax1.set_yticks(lat_zint)
ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
#plt.title('Pattern Correlation(Lead Day1)')
plt.colorbar(ticks=[0.0,0.2,0.4,0.6,0.8,1]).ax.set_yticklabels( ['0','0.2','0.4','0.6','0.8','1.0'],weight='bold', fontsize=14)
plt.grid()
plt.savefig(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_patcorr.png',dpi=400,format='png',figsize=(5, 9))
plt.clf()

se = np.zeros((129,135),dtype=float)
maer=[]
for i in range(y1.shape[0]):
  se += abs(yhat[i,0,:,:]-y1[i,0,:,:])
maer=se/(y1.shape[0])   
maer=np.array(maer)

np.save(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_mae_imdrf', maer)


plt.imshow(maer,interpolation='none',cmap='nipy_spectral',vmin=0, vmax=15)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
ax1 = plt.axes()
ax1.set_ylim(0,129)
ax1.set_xticks(lon_zint)
ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
ax1.set_yticks(lat_zint)
ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
#plt.title('Pattern RMSE(Lead Day1)')
plt.colorbar(ticks=[0,2,4,6,8,10,12,14]).ax.set_yticklabels( ['0','2','4','6','8','10','12','14'],weight='bold', fontsize=14)
plt.grid()
plt.savefig(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_mae.png',dpi=400,format='png',figsize=(5, 9))
plt.clf()

yt=y1.copy()
yh=yhat.copy()
y1=y1/np.nanmax(y1)
yhat=yhat/np.nanmax(yhat)
y1=np.exp(y1)
yhat=np.exp(yhat)

se = np.zeros((129,135),dtype=float)
mape=[]
for i in range(y1.shape[0]):
  se += abs((y1[i,0,:,:]-yhat[i,0,:,:])/y1[i,0,:,:])
mape=100*se/(y1.shape[0])   
mape=np.array(mape)

print(np.nanmin(mape))
print(np.nanmax(mape))
np.save(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_mape_exp_imdrf', mape)

plt.imshow(mape,interpolation='none',cmap='nipy_spectral',vmin=0, vmax=5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
ax1 = plt.axes()
ax1.set_ylim(0,129)
ax1.set_xticks(lon_zint)
ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
ax1.set_yticks(lat_zint)
ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
#plt.title('Pattern RMSE(Lead Day1)')
#plt.colorbar(ticks=[0,0.25,0.50,0.75,1,1.25,1.50,1.75,2]).ax.set_yticklabels( ['0.00','0.25','0.50','0.75','1.00','1.25','1.50','1.75','2.00'],weight='bold', fontsize=14)
plt.colorbar(ticks=[0,1,2,3,4,5]).ax.set_yticklabels( ['0','1','2','3','4','5'],weight='bold', fontsize=14)
plt.grid()
plt.savefig(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_mape_exp.png',dpi=400,format='png',figsize=(5, 9))
plt.clf()


y1=yt+1
yhat=yh+1

print(np.nanmin(y1))
print(np.nanmax(y1))
print(np.nanmin(yhat))
print(np.nanmax(yhat))

se = np.zeros((129,135),dtype=float)
mape=[]
for i in range(y1.shape[0]):
  se += abs((y1[i,0,:,:]-yhat[i,0,:,:])/y1[i,0,:,:])
mape=100*se/(y1.shape[0])   
mape=np.array(mape)

print(np.nanmin(mape))
print(np.nanmax(mape))
np.save(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_mape_lin_imdrf', mape)

plt.imshow(mape,interpolation='none',cmap='nipy_spectral',vmin=100, vmax=350)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
ax1 = plt.axes()
ax1.set_ylim(0,129)
ax1.set_xticks(lon_zint)
ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
ax1.set_yticks(lat_zint)
ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
#plt.title('Pattern RMSE(Lead Day1)')
plt.colorbar(ticks=[100,150,200,250,300,350]).ax.set_yticklabels( ['100','150','200','250','300','350'],weight='bold', fontsize=14)
plt.grid()
plt.savefig(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_mape_lin.png',dpi=400,format='png',figsize=(5, 9))
plt.clf()

y1=yt.copy()
yhat=yh.copy()

se = np.zeros((129,135),dtype=float)
rm=[]
for i in range(y1.shape[0]):
  aa = (yhat[i,0,:,:]-y1[i,0,:,:])
  se += np.square(aa)
rmse=np.sqrt(se/(y1.shape[0]))    
rm.append(rmse)
rm=np.array(rm)
np.save(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_rm_imdrf', rm)

plt.imshow(rm[0],interpolation='none',cmap='nipy_spectral',vmin=0, vmax=30)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
ax1 = plt.axes()
ax1.set_ylim(0,129)
ax1.set_xticks(lon_zint)
ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
ax1.set_yticks(lat_zint)
ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
#plt.title('Pattern RMSE(Lead Day1)')
#plt.colorbar(ticks=[100,150,200,250,300,350]).ax.set_yticklabels( ['100','150','200','250','300','350'],weight='bold', fontsize=14)
plt.colorbar()
plt.grid()
plt.savefig(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_rmse.png',dpi=400,format='png',figsize=(5, 9))
plt.clf()


#ystd=np.load('/home/namit/project/analysis/imd_std.npy')
ystd=np.std(y1[:,0,:,:], axis=0)

plt.imshow(ystd,interpolation='none',cmap='nipy_spectral',vmin=0, vmax=30)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
ax1 = plt.axes()
ax1.set_ylim(0,129)
ax1.set_xticks(lon_zint)
ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
ax1.set_yticks(lat_zint)
ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
#plt.title('Pattern RMSE(Lead Day1)')
#plt.colorbar(ticks=[100,150,200,250,300,350]).ax.set_yticklabels( ['100','150','200','250','300','350'],weight='bold', fontsize=14)
plt.colorbar()
plt.grid()
plt.savefig(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_imd_std.png',dpi=400,format='png',figsize=(5, 9))
plt.clf()

nrmse=rm[0]/ystd

plt.imshow(nrmse,interpolation='none',cmap='nipy_spectral',vmin=0.5, vmax=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'

ax1 = plt.axes()
ax1.set_ylim(0,129)
ax1.set_xticks(lon_zint)
ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=17)
ax1.set_yticks(lat_zint)
ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=17)
#plt.colorbar()
#plt.title('Pattern RMSE(Lead Day1)')
#plt.colorbar(ticks=[100,150,200,250,300,350]).ax.set_yticklabels( ['100','150','200','250','300','350'],weight='bold', fontsize=14)
plt.colorbar(ticks=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]).ax.set_yticklabels(['0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2'], weight='bold', fontsize=17)
plt.grid()
plt.savefig(path+f'ld{LEADTIME}_{len(EXT_VARIABLES)}extvars_{var_string}'+'_imd_nrmse.png',dpi=400,format='png',figsize=(5, 9))
plt.clf()
print(T)
print(y1.shape)

