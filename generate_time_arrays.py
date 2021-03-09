import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pytz
from timezonefinder import TimezoneFinder
import pickle



def main():

  dp = Dataset('../dpfile.nc', 'r')
  t2m = dp.variables['ta'][:]
  lat = dp.variables['lat'][:]
  lon = dp.variables['lon'][:]
  dp.close()

  for m in np.arange(2, 13):
    print(m)
    year_i = 2003
    year_f = 2003
    m_f = m+1
    if m == 12:
      year_f = 2004
      m_f = 1
    tempo = np.arange(datetime(year_i,m,1,3,0), datetime(year_f,m_f,1,3,0), timedelta(hours=3)).astype(datetime)

    t_arr = np.empty((len(tempo),172,172), dtype=object)
    t, i, j = t_arr.shape

    for i in range(0, lon.shape[0]):
      for j in range(0, lon.shape[0]):
        t_arr[:,i,j] = tempo

    utc = pytz.utc
    tf = TimezoneFinder()
    
    for t in range(0, len(tempo)):
      for i in range(0,lon.shape[0]):
        for j in range(0, lon.shape[0]):
          #print(t, i, j)
          a = utc.localize(t_arr[t,i,j])
          # using a high latitude as I want time as a longitude function, not geopolitical
          tz = tf.timezone_at(lng=float(lon[i,j]), lat=85)
          tzone = pytz.timezone(tz)
          t_arr[t,i,j] = a.astimezone(tzone)
        
    pickle.dump(t_arr, open('time_array_{0:02d}.p'.format(m), "wb"))


if __name__ == '__main__':
    main()
