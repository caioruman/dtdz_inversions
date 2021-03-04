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


datai = 2003
dataf = 2015

def plotMaps_pcolormesh(data, figName, values, mapa, lons2d, lats2d):
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

def save_pickle(pickle_folder, per, var, t_test, p, mean1, std1, mean2, std2):

  pickle.dump(t_test, open('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  pickle.dump(p, open('{0}/{4}_p_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  pickle.dump(mean1, open('{0}/{4}_mean_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  pickle.dump(std1, open('{0}/{4}_std_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  pickle.dump(mean2, open('{0}/{4}_mean_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  pickle.dump(std2, open('{0}/{4}_std_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))

  #pickle.dump(t_test, open('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  #pickle.dump(p, open('{0}/{4}_p_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  #pickle.dump(mean1, open('{0}/{4}_mean_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  #pickle.dump(std1, open('{0}/{4}_std_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  #pickle.dump(mean2, open('{0}/{4}_mean_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))
  #pickle.dump(std2, open('{0}/{4}_std_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "wb"))


"""
 -- Calculate the mean and coefficient of variance / std 
    for the AIRS data and the GEM-ERA simulation
    Plot the data: AIRS, GEM-ERA, and GEM-ERA minus AIRS

 -- Periods: DJF and JJA
"""

# simulation
exp = ["PanArctic_0.5d_ERAINT_NOCTEM_RUN"]
#exp = ["PanArctic_0.5d_CanHisto_NOCTEM_RUN", "PanArctic_0.5d_CanHisto_NOCTEM_R2",
#       "PanArctic_0.5d_CanHisto_NOCTEM_R3", "PanArctic_0.5d_CanHisto_NOCTEM_R4",
#       "PanArctic_0.5d_CanHisto_NOCTEM_R5"]

main_folder = "/pixel/project01/cruman/ModelData/GEM_SIMS"
#main_folder = "/home/caioruman/Documents/McGill/NC/dtdz/"
val_folder = "/pixel/project01/cruman/Data/AIRS/AIRS_daily/Inversion"
pickle_folder = "/pixel/project01/cruman/Data/Pickle"

# Sounding Data
#sounding_file = "/home/cruman/project/cruman/Scripts/soundings/inv_list_DJF.dat"

# Variables:
# FQ_925, DT_925, DTDZ_925
# FQ_850, DT_850, DTDZ_850

period = ["DJF", "JJA"]#, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Nov', 'Dec']
period = [((12, 1, 2), "DJF"), ((6, 7, 8), "JJA")]

from matplotlib.colors import  ListedColormap
# Open the monthly files

for per in period:

  initV = True
  init = True

  for year in range(datai, dataf+1):
    for month in per[0]:
      print(year, month)
      
      # Opening the model file
      for ee in exp:
        ff = "{0}/{3}/{1}/Inversion_{1}{2:02d}.nc".format(main_folder, year, month, ee)
        #print(ff)
        arq = Dataset(ff, 'r')
        
        if init:

          fq_925 = arq.variables['FQ_925'][:][0]
          fq_850 = arq.variables['FQ_850'][:][0]
 #         dt_925 = arq.variables['DT_925'][:]
 #         dt_850 = arq.variables['DT_850'][:]
 #         dtdz_925 = arq.variables['DTDZ_925'][:]
 #         dtdz_850 = arq.variables['DTDZ_850'][:]

          lats2d = arq.variables['lat']
          lons2d = arq.variables['lon']
          init = False

        else:

          fq_925 = np.vstack((arq.variables['FQ_925'][:][0], fq_925))
          fq_850 = np.vstack((arq.variables['FQ_850'][:][0], fq_850))
 #         dt_925 = np.vstack((arq.variables['DT_925'][:], dt_925 ))
 #         dt_850 = np.vstack((arq.variables['DT_850'][:], dt_850))
 #         dtdz_925 = np.vstack((arq.variables['DTDZ_925'][:], dtdz_925))
 #         dtdz_850 = np.vstack((arq.variables['DTDZ_850'][:], dtdz_850))

        arq.close()
      # adding the variables to the array/list
#      print(fq_925.shape)
      # Opening the AIRS files
      ff = "{0}/{1}/Inversion_{1}{2:02d}.crcm5grid.nc".format(val_folder, year, month)
#      print(ff)
      arq = Dataset(ff, 'r')

      if initV:

        fq_925v = arq.variables['FQ_925'][:][0]
        fq_850v = arq.variables['FQ_850'][:][0]
 #       dt_925v = arq.variables['DT_925'][:]
 #       dt_850v = arq.variables['DT_850'][:]
 #       dtdz_925v = arq.variables['DTDZ_925'][:]
 #       dtdz_850v = arq.variables['DTDZ_850'][:]

        initV = False

      else:

        fq_925v = np.vstack((arq.variables['FQ_925'][:][0], fq_925v))
        fq_850v = np.vstack((arq.variables['FQ_850'][:][0], fq_850v))
 #       dt_925v = np.vstack((arq.variables['DT_925'][:], dt_925v ))
 #       dt_850v = np.vstack((arq.variables['DT_850'][:], dt_850v))
 #       dtdz_925v = np.vstack((arq.variables['DTDZ_925'][:], dtdz_925v))
 #       dtdz_850v = np.vstack((arq.variables['DTDZ_850'][:], dtdz_850v))

      arq.close()
#      fq_925v[fq_925v > 1] = 20
#      print(fq_925v.shape)
#      print((fq_925v == 20).sum())
    
#      sys.exit()

  #t_test, p, mean1, mean2, std1, std2 = calcStats(dt_925, dt_925v)
  
  # Saving things to pickle to save processing time.
  #save_pickle(pickle_folder, per, 'dt_925', t_test, p, mean1, std1, mean2, std2)

  #t_test, p, mean1, mean2, std1, std2 = calcStats(dt_850, dt_850v)
  
  # Saving things to pickle to save processing time.
  #save_pickle(pickle_folder, per, 'dt_850', t_test, p, mean1, std1, mean2, std2)

  t_test_925, p_925, mean_925, mean_925v, std_925, std_925v = calcStats(fq_925, fq_925v)
    
  # Saving things to pickle to save processing time.
  #save_pickle(pickle_folder, per, 'fq_925', t_test, p, mean1, std1, mean2, std2)

  t_test_850, p_850, mean_850, mean_850v, std_850, std_850v = calcStats(fq_850, fq_850v)
  
  # Saving things to pickle to save processing time.
  #save_pickle(pickle_folder, per, 'fq_850', t_test, p, mean1, std1, mean2, std2)

  # Plotting the differences

  figName = "fig_{0}_{1}_{2}".format(datai, 'freq_925', per[1])
  
  colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']

  #v = abs(max(np.nanmax(data), np.nanmin(data), key=abs))

  values = np.linspace(-100, 100, len(colors)+1)
  mean_925 = mean_925*100
  mean_925v = mean_925v*100
  
  data = mean_925 - mean_925v

  cmap = mpl.colors.ListedColormap(colors)

  plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d)

  # Plotting the values
  values = np.arange(0,101,10)
  colors = ['#ffffff', '#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']

  cmap = mpl.colors.ListedColormap(colors)

  figName = "fig_{0}_{1}_{2}_meanGEM".format(datai, 'freq_925', per[1])
  
  plotMaps_pcolormesh(mean_925, figName, values, cmap, lons2d, lats2d)

  figName = "fig_{0}_{1}_{2}_meanAIRS".format(datai, 'freq_925', per[1])
  plotMaps_pcolormesh(mean_925v, figName, values, cmap, lons2d, lats2d)

  # Plotting the differences

  figName = "fig_{0}_{1}_{2}".format(datai, 'freq_850', per[1])
  
  colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']

  #v = abs(max(np.nanmax(data), np.nanmin(data), key=abs))

  values = np.linspace(-100, 100, len(colors)+1)
  mean_850 = mean_850*100
  mean_850v = mean_850v*100
  
  data = mean_850 - mean_850v

  cmap = mpl.colors.ListedColormap(colors)

  plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d)

  # Plotting the values
  values = np.arange(0,101,10)
  colors = ['#ffffff', '#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']

  cmap = mpl.colors.ListedColormap(colors)

  figName = "fig_{0}_{1}_{2}_meanGEM".format(datai, 'freq_850', per[1])
  
  plotMaps_pcolormesh(mean_850, figName, values, cmap, lons2d, lats2d)

  figName = "fig_{0}_{1}_{2}_meanAIRS".format(datai, 'freq_850', per[1])
  plotMaps_pcolormesh(mean_850v, figName, values, cmap, lons2d, lats2d)