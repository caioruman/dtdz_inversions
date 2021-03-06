import numpy as np
import pandas as pd
import sys
import calendar
from scipy import stats

import numpy.ma as ma

from glob import glob
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib as mpl # in python
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
from netCDF4 import Dataset

import pickle


datai = 1976
dataf = 2005

dataif = 2070
dataff = 2099

def plotMaps_pcolormesh(data, figName, values, mapa, lons2d, lats2d, stations, var):
  '''
  fnames: List of filenames. Usually 2 (clim mean and future projection)
  varnames: list of variables to plot
  titles: list of the titles
  '''

  # GPCP
  fig = plt.figure(1, figsize=(14, 22), frameon=False, dpi=150)
  bn = BoundaryNorm(values, ncolors=len(values) - 1)

  b = Basemap(projection='npstere',boundinglat=50,lon_0=-90,resolution='l', round=True)
  x, y = b(lons2d, lats2d)

  img = b.pcolormesh(x, y, data, cmap=mapa, vmin=values[0], vmax=values[-1], norm=bn)
    #img = b.contourf(x, y, newdata2, cmap=cmap, norm=bn, levels=values, extend='both')

  b.drawcoastlines()
  cbar = b.colorbar(img, pad=0.75, ticks=values)
  cbar.ax.tick_params(labelsize=20)
  #cbar.ax.set_yticklabels(values)
  cbar.outline.set_linewidth(1)
  cbar.outline.set_edgecolor('black')

  b.drawcountries(linewidth=0.5)
  b.drawstates(linewidth=0.5)

  #parallels = np.arange(0.,81,10.)
  # labels = [left,right,top,bottom]
  #b.drawparallels(parallels,labels=[True,True,True,True], fontsize=16)
  meridians = np.arange(0.,351.,45.)
  b.drawmeridians(meridians,labels=[True,True,True,True], fontsize=16)
  for item in stations:
#        print(item)
#        print(stations)
#        print
    x, y = b(item[1], item[0])
    if var == "DT" or var == "DT0" or var == "DT12":
      vv = item[2]
    else:
      vv = item[3]

    img = b.scatter(x, y, c=vv, s=80, cmap=mapa, norm=bn, edgecolors='black')

  plt.subplots_adjust(top=0.75, bottom=0.25)


  plt.savefig('{0}.png'.format(figName), pad_inches=0.0, bbox_inches='tight')
  plt.close()

def calcStats(var1, var2):

  mean1 = np.nanmean(var1, axis=0)
  mean2 = np.nanmean(var2, axis=0)

  std1 = np.nanstd(var1, axis=0)
  std2 = np.nanstd(var2, axis=0)

  t_test, p = stats.ttest_ind(var1, var2, axis=0, nan_policy='omit')

  return t_test, p, mean1, mean2, std1, std2

def save_pickle(pickle_folder, per, var, t_test, p, mean1, std1, mean2, std2, name):

  pickle.dump(t_test, open('{0}/{4}_t_test_{5}_{1}{2}_{3}.p'.format(pickle_folder, datai, dataif, per, var, name), "wb"))
  pickle.dump(p, open('{0}/{4}_p_{5}_{1}{2}_{3}.p'.format(pickle_folder, datai, dataif, per, var, name), "wb"))
  pickle.dump(mean1, open('{0}/{3}_mean_{4}_{1}_{2}.p'.format(pickle_folder, datai, per, var, name), "wb"))
  pickle.dump(std1, open('{0}/{3}_std_{4}_{1}_{2}.p'.format(pickle_folder, datai, per, var, name), "wb"))
  pickle.dump(mean2, open('{0}/{3}_mean_{4}_{1}_{2}.p'.format(pickle_folder, dataif, per, var, name), "wb"))
  pickle.dump(std2, open('{0}/{3}_std_{4}_{1}_{2}.p'.format(pickle_folder, dataif, per, var, name), "wb"))


"""
 -- Calculate the mean and coefficient of variance / std 
    for the AIRS data and the GEM-ERA simulation
    Plot the data: AIRS, GEM-ERA, and GEM-ERA minus AIRS

 -- Periods: DJF and JJA
"""

# simulation
exp = ["PanArctic_0.5d_ERAINT_NOCTEM_RUN"]
exp = ["PanArctic_0.5d_CanHisto_NOCTEM_RUN", "PanArctic_0.5d_CanHisto_NOCTEM_R2",
       "PanArctic_0.5d_CanHisto_NOCTEM_R3", "PanArctic_0.5d_CanHisto_NOCTEM_R4",
       "PanArctic_0.5d_CanHisto_NOCTEM_R5"]

exprcp45 = ["PanArctic_0.5d_CanRCP45_NOCTEM_RUN", "PanArctic_0.5d_CanRCP45_NOCTEM_R2",
       "PanArctic_0.5d_CanRCP45_NOCTEM_R3", "PanArctic_0.5d_CanRCP45_NOCTEM_R4",
       "PanArctic_0.5d_CanRCP45_NOCTEM_R5"]

main_folder = "/pixel/project01/cruman/ModelData/GEM_SIMS"
#main_folder = "/home/caioruman/Documents/McGill/NC/dtdz/"
val_folder = "/pixel/project01/cruman/Data/AIRS/AIRS_daily/Inversion"
pickle_folder = "/pixel/project01/cruman/Data/Pickle/t-test"

# Sounding Data
#sounding_file = "/home/cruman/project/cruman/Scripts/soundings/inv_list_DJF.dat"

# Variables:
# FQ_925, DT_925, DTDZ_925
# FQ_850, DT_850, DTDZ_850

period = ["DJF", "JJA"]#, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Nov', 'Dec']
period = [[(12, 1, 2), "DJF"], [(6, 7, 8), "JJA"]]

name = "GEMCAM_RCP45"

rcp45 = True

# The arrays are too large to load all variables at once
var = ["FQ_925", "FQ_850", "DT_925", "DT_850", "DTDZ_925", "DTDZ_850"]
var = ["FQ_925", "DT_925", "DTDZ_925"]
var = ["DT_925", "DTDZ_925"]
#var = ["DTDZ_925"]

from matplotlib.colors import  ListedColormap
# Open the monthly files

for vv in var:
  
  for per in period:

    print(vv, per)
    init = True

    # for dt and dtdz only
    # 8 data per day. 3 months. DJF: 31, 31, 28 = 90 days a year. JJA: 30, 31, 31 = 92 days a year. times 30 years
    # 5 ensembles. DJF: 108000; JJA: 110400
    if per[1] == "JJA":
      aa = 110400
    else:
      aa = 108000
    aux_full = np.zeros([aa,172,172])
    aux_full_f = np.zeros([aa,172,172])
    j = 0

    #for year in range(datai, dataf+1):
    for i in range(0, 30):
      year = datai + i
      yearf = dataif + i
    
      for month in per[0]:
        print(year, month)
      
        # Opening the model file
#        for ee in exp:
        for ee in range(0,5):

          
          ff = "{0}/{3}/{1}/Inversion_{1}{2:02d}.nc".format(main_folder, year, month, exp[ee])

          if rcp45:
            fff = "{0}/{3}/{1}/Inversion_{1}{2:02d}.nc".format(main_folder, yearf, month, exprcp45[ee])
          else:  
            fff = "{0}/{3}/{1}/Inversion_{1}{2:02d}.nc".format(main_folder, yearf, month, exp[ee])
          print(ff)
  #        sys.exit()
          arq = Dataset(ff, 'r')

          arqf = Dataset(fff, 'r')
        
          if (vv == "FQ_925" or vv == "FQ_850"):

            aux = arq.variables[vv][:][0]
            aux = aux[np.newaxis,:,:]

            aux_f = arqf.variables[vv][:][0]
            aux_f = aux_f[np.newaxis, :, :]

          else:
            aux = arq.variables[vv][:]
  
            aux_f = arqf.variables[vv][:]
            
          mm = aux.shape[0]
          aux_full[j:j+mm,:,:] = aux
          aux_full_f[j:j+mm,:,:] = aux_f

          j = j + mm
          #print(aux_f)
          #print(aux_full_f[j-mm:j,:,:])

          lats2d = arq.variables['lat']
          lons2d = arq.variables['lon']

          arq.close()
          arqf.close()
      # adding the variables to the array/list

    t_test, p, mean1, mean2, std1, std2 = calcStats(aux, aux_f)

    # Saving things to pickle to save processing time.
    save_pickle(pickle_folder, per[1], vv, t_test, p, mean1, std1, mean2, std2, name)
    print('file saved', vv, per[1])

