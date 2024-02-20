import numpy as np
import xarray as xr
import os
#from dask.distributed import Client
path = 'data/'
file_list = ['ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-K003.hindcast_1m_19580101_20161231_psi.nc',
             'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-K005.wind90_1m_19580101_20161231_psi.nc',
             'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-K004.thermhal90_1m_19580101_20161231_psi.nc',
             'ORCA025.L46.LIM2vp.JRA.XIOS2-KPW001.RYF90_1m_19580101_20161231_psi.nc']
#TODO
x_min, x_max = -82, 20 #lon limits

def load_n_slice(file, x_min = -82,x_max = 20 ):
    ds = xr.open_dataset(path + file)#dia
    ds = ds.set_coords(('nav_lat', 'nav_lon'))
    ds = ds.where((ds['nav_lon'] >= x_min) & (ds['nav_lon'] <= x_max))
    ds = ds.sobarstf
    return ds

    
def load_psi_std (file):   
    ds = load_n_slice(file)    
    bar_std = ds.sobarstf.std(dim = 'time_counter')
    return bar_std

#%% select cape hatteras
gulf_x = (-80, -70)#_boxmin, x_boxmax = -80, -65 #lon limits
gulf_y = (30, 36.5)#y_boxmin, y_boxmax = 30, 51
gulf_x_list = [gulf_x[0],gulf_x[0],gulf_x[1],gulf_x[1],gulf_x[0]]
gulf_y_list = [gulf_y[0],gulf_y[1],gulf_y[1],gulf_y[0],gulf_y[0]]

#LS
#x_start = 59
#x_end = 64
lab_x = (-60, -52)#_boxmin, x_boxmax = -80, -65 #lon limits
lab_y = (59, 64)#y_boxmin, y_boxmax = 30, 51
lab_x_list = [lab_x[0],lab_x[0],lab_x[1],lab_x[1],lab_x[0]]
lab_y_list = [lab_y[0],lab_y[1],lab_y[1],lab_y[0],lab_y[0]]



def calc_timeseries(file):
    
    ds = load_n_slice(file)
    ts = np.zeros(ds.shape[0])#initiate ts array
        #slice to box
    ds_cape = ds.where((ds['nav_lon'] >= gulf_x[0]) & (ds['nav_lon'] <= gulf_x[1]))
    ds_cape = ds_cape.where((ds_cape['nav_lat'] >= gulf_y[0]) & (ds_cape['nav_lat'] <= gulf_y[1]))
    ds_cape = ds_cape.dropna(dim=('y'), how='all').dropna(dim=('x'), how='all')
        
    for t in range(ds.shape[0]):#iterate over time
        
        ds_capem = ds_cape.isel(time_counter = t)
        ind_xm = np.zeros(ds_capem.shape[0]-1, dtype = 'int')
        ind_ym = np.arange(ds_capem.shape[0]-1)
        strf_val = np.zeros(ds_capem.shape[0]-1)
            
        for i in range(ds_capem.shape[0]-1):#iterate over latitudes in box
                #print
                
            diffs = np.ediff1d(ds_capem.isel(y = i).where(ds_capem.isel(y = i)> 0.5e7).values)
            if np.size(np.where(diffs < 0)[0]) > 0: #in case if no local maxima, ind = 17 
                ind_xm[i] = np.where(diffs < 0)[0][0]    
            
            else:
                ind_xm[i] = np.nanargmax(np.ediff1d(ds_capem.isel(y = i)))
                 #find general maximum at latitude
            
            strf_val[i] = ds_capem.isel(x = ind_xm[i], y = ind_ym[i]).values 
            strf_ind = np.argmax(strf_val)
            
        ts[t] = ds_capem.isel(x = slice(ind_xm[strf_ind]-1, ind_xm[strf_ind]+2),
                                       y = slice(strf_ind-1, strf_ind+2)).mean().values
            
    return ts


def fix_nans(arr):
    mask = np.isnan(arr)  # Create a boolean mask where True indicates NaN values
    mean_val = np.nanmean(arr)  # Calculate the mean of the array, excluding NaN values
    arr[mask] = mean_val  # Replace NaN values with the mean value
    return arr


#%%filter
def smooth_arr(arr,window_size = 23):
    
    window = np.hanning(window_size)

    padded_arr = np.pad(arr, (window_size // 2, window_size // 2), mode='edge')

    smooth_arr = np.convolve(padded_arr, window, mode='valid')
    #scale to unfiltered data
    scaling_factor = np.nanmean(arr) / np.nanmean(smooth_arr)
    
    return smooth_arr* scaling_factor
