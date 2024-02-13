from fastapi import FastAPI, Request, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
import os
import xarray as xr
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import argparse
import keras
from io import BytesIO
import base64
from starlette.responses import StreamingResponse
import utils
from keras.models import Sequential
plt.style.use('ggplot')
import argparse
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Conv3D, Conv2D
from keras.layers import ConvLSTM2D
from fastapi import File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# EXT_VARIABLES = ["oro", "q700", "q850", "spr", "sw1"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


# Helper function to encode images
def encode_image(image_bytes):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"

@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    # Replace this with your actual list of plot paths or URLs
    plots = [
        "./forecast_plots/forecast_ld1.png",
        "./forecast_plots/forecast_ld2.png",
        "./forecast_plots/forecast_ld3.png",
    ]
    encoded_plots = [encode_image(open(plot, "rb").read()) for plot in plots]
    return templates.TemplateResponse("results.html", {"request": request, "encoded_plots": encoded_plots})

@app.get("/")
def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, date: str = Form(...), num_forecasts: int = Form(...), ext_variables: str = Form(...)):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
    dir=os.path.dirname(__file__)
    
    ext_variables_list = []
    if ext_variables.lower() == "all":
        ext_variables_list = ["oro", "q700", "q850", "spr", "sw1"]


    # Load Data
    imd_z=xr.open_dataset(os.path.join(dir, "data/rf_allvar_1979-2019.nc"))

    LEADTIME=1


    class datagen(keras.utils.Sequence):
        def __init__(self, dataset, batch_size, leadtime, in_frames, ext_variables = [],data_format = 'channels_last', \
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
                # elif len(self.ext_variables)==5:
                #   data=np.concatenate([im, sph7, sph85, spr, o, sw1], axis=2)
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

            elif self.data_format == "channels_last":
                im = X[:, :, :, :, np.newaxis]
                sph7 = q7[:, :, :, :, np.newaxis]
                sph85 = q85[:, :, :, :, np.newaxis]
                spr = spr[:, :, :, :, np.newaxis]
                o = oro[:, :, :, :, np.newaxis]
                sw1 = sw1[:, :, :, :, np.newaxis]

                vars = {'q700': sph7, 'q850': sph85, 'spr': spr, 'oro': o, 'sw1': sw1}
                if len(self.ext_variables) == 0:
                    data = im
                else:
                    for ind, var in enumerate(self.ext_variables):
                        if ind == 0:
                            data = np.concatenate([im, vars[var]], axis=-1)
                        else:
                            data = np.concatenate([data, vars[var]], axis=-1)
                if self.mode=="trainortest":
                    X, y = data, y[:, :, :, np.newaxis]
                else:
                    X, y = data, y
                
            return X, y

    seq_channels_last = Sequential()
    seq_channels_last.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), padding='same', 
                                    input_shape=(None, 129, 135, len(ext_variables_list) + 1),
                                    return_sequences=True, data_format='channels_last'))

    seq_channels_last.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', 
                                    return_sequences=True, data_format='channels_last'))
    seq_channels_last.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', 
                                    return_sequences=True, data_format='channels_last'))
    seq_channels_last.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', 
                                    return_sequences=True, data_format='channels_last'))
    seq_channels_last.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', 
                                    return_sequences=False, data_format='channels_last'))
    seq_channels_last.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', 
                                padding='same', data_format='channels_last'))
    seq_channels_last.add(Conv2D(filters=1, kernel_size=(3, 3), activation='relu', 
                                padding='same', data_format='channels_last'))
    
    print(ext_variables_list)
    try:
        var_string = '_'.join(ext_variables_list)
    except:
        var_string = ''

    seq_path = os.path.join(dir, 'models')

    print(os.path.join(seq_path, f'ld{LEADTIME}_{len(ext_variables_list)}extvars_{var_string}.h5'))

    # Initialize seq outside the try block
    seq = None

    # Check if external variables are selected
    if ext_variables_list is not None:
        try:
            # If no external variables are selected, use the ld1_0extvars_.h5 model
            if len(ext_variables_list) == 0:
                var_string = ''
                model_path = os.path.join(seq_path, f'ld{LEADTIME}_{len(ext_variables_list)}extvars_{var_string}.h5')
            else:
                # Otherwise, construct the var_string and load the appropriate model
                try:
                    var_string = '_'.join(ext_variables_list)
                except:
                    var_string = ''
                model_path = os.path.join(seq_path, f'ld{LEADTIME}_{len(ext_variables_list)}extvars_{var_string}.h5')

            original_model = load_model(model_path)
            seq_channels_last.set_weights(original_model.get_weights())
            seq = seq_channels_last
            print('Model file found')
            print(seq.summary())
        except Exception as e:
            print(e)
    else:
        print("No external variables selected. Loading default model.")

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
    dg_test=datagen(dat, batch_size=1, leadtime=LEADTIME, in_frames=5, max=T, ext_variables=ext_variables_list, mode="forecast", data_format="channels_last") #to be used as input for prediction

    # Initialize plot_bytes outside the loop
    plot_bytes = BytesIO()
    encoded_plots = []
    for i in range(1, int(num_forecasts)+1):
    # Model Prediction
        if i!=1:
            next_day = dat['time'].values[1:][-1] + np.timedelta64(1, 'D')
            print(next_day)
            extended_dates = np.append(dat['time'].values[1:], next_day)
            ds= xr.Dataset({
            "rain":(["time", "lat", "lon"], np.vstack((dat['rain'].values[1:],yhat[:,:,:,0]))),
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
            dg_test=datagen(ds, batch_size=1, leadtime=LEADTIME, in_frames=5, max=T, ext_variables=ext_variables_list, mode="forecast", data_format="channels_last") 
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
        vals=y.reshape(1,129,135,1)
        a = np.where(vals>=1)
        b = np.where(vals==0)

        yhat[:,:,:,:][a] = np.power(yhat[:,:,:,:][a],1/7)
        yhat[:,:,:,:][a] = np.log(yhat[:,:,:,:][a])
        yhat[:,:,:,:][a] = yhat[:,:,:,:][a]*(T)
        c = np.where(yhat[:,:,:,:]<=0)
        yhat[:,:,:,:][c] = 0
        yhat[:,:,:,:][b] = np.nan

        # Save the plot image to a BytesIO object
        plot_bytes = BytesIO()
        plt.imshow(yhat[0,:,:,0],cmap='nipy_spectral_r',vmin=0,vmax=50)  # set vmin and vmax as arguments in imshow to adjust for scale
        ax1 = plt.axes()
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        ax1.set_ylim(0,129)
        ax1.set_xticks(lon_zint)
        ax1.set_xticklabels(lon_zstr, weight='bold', fontsize=14)
        ax1.set_yticks(lat_zint)
        ax1.set_yticklabels(lat_zstr, weight='bold', fontsize=14)
        plt.title('Lead Day %i Prediction'%i)
        plt.colorbar(ticks=zint).ax.set_yticklabels( zstr,weight='bold', fontsize=14)
        plt.grid()
        plt.savefig(plot_bytes, format='png', bbox_inches='tight')
        plt.clf()
        
        # Set the BytesIO cursor to the beginning before sending it
        plot_bytes.seek(0)

        # Encode the plot image
        encoded_plot = base64.b64encode(plot_bytes.read()).decode("utf-8")
        plot_data_uri = f"data:image/png;base64,{encoded_plot}"

        # Append each plot_data_uri to the list
        encoded_plots.append(plot_data_uri)        

    # Now, you can use the list of encoded_plots as needed in your template response
    return templates.TemplateResponse("results.html", {"request": request, "encoded_plots": encoded_plots})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)