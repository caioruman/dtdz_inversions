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
#   for item in stations:
# #        print(item)
# #        print(stations)
# #        print
#     x, y = b(item[1], item[0])
#     if var == "DT" or var == "DT0" or var == "DT12":
#       vv = item[2]
#     else:
#       vv = item[3]

#     img = b.scatter(x, y, c=vv, s=80, cmap=mapa, norm=bn, edgecolors='black')

  plt.subplots_adjust(top=0.75, bottom=0.25)


  plt.savefig('{0}.png'.format(figName), pad_inches=0.0, bbox_inches='tight')
  plt.close()


"""
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
period = [(12, 1, 2), (6, 7, 8)]
period = [[(12, 1, 2), "DJF"], [(6, 7, 8), "JJA"]]
# lat lon
arq = Dataset('/pixel/project01/cruman/ModelData/GEM_SIMS/PanArctic_0.5d_ERAINT_NOCTEM_RUN/2014/Inversion_201412.nc','r')
lats2d = arq.variables['lat'][:]
lons2d = arq.variables['lon'][:]

from matplotlib.colors import  ListedColormap
# Open the monthly files
vars = ['dt_925', 'dt_850', 'fq_925', 'fq_850', 'dtdz_925', 'dtdz_850']
vars = ['fq_925', 'fq_850']
for per in period:
  for var in vars:
  
  # Opening the files
    per2 = str(per[0]).replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '_')
    print('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var))
    t_test = pickle.load( open('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    p = pickle.load( open('{0}/{4}_p_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    mean_gem = pickle.load( open('{0}/{4}_mean_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    std_gem = pickle.load( open('{0}/{4}_std_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    mean_airs = pickle.load( open('{0}/{4}_mean_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    std_airs = pickle.load( open('{0}/{4}_std_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    
#    print(mean_airs.shape)
#    print(mean_airs)
    #sig = p.copy()
    #sig[p > 0.05] = np.nan

  # Figures for the mean and std of each variable, 

    # Plotting the differences

    figName = "fig_{0}_{1}_{2}".format(datai, var, per[1])
    
    colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
  
    #v = abs(max(np.nanmax(data), np.nanmin(data), key=abs))

    if (var == "dt_925" or var == "dt_850"):
      values = np.linspace(-9, 9, len(colors)+1)
      if per[1] == "JJA":
        values = np.linspace(-4.5, 4.5, len(colors)+1)
    elif (var == "dtdz_925" or var == "dtdz_850"):
      values = np.linspace(-9, 9, len(colors)+1)
      if per[1] == "JJA":
        values = np.linspace(-4.5, 4.5, len(colors)+1)
      mean_gem = mean_gem*100   # original units: K/dm
      mean_airs = mean_airs*1000 # original units: K/m
    else:
      values = np.linspace(-100, 100, len(colors)+1)
      mean_gem = mean_gem*100
      mean_airs = mean_airs*100
      print(mean_gem.shape)
      print(mean_airs.shape)

    data = mean_gem - mean_airs

    cmap = mpl.colors.ListedColormap(colors)

    plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d)

    # Plotting the values

    #colors = ['#ffffff', '#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
#    colors = ['#ffffff','#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
    colors = ['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    if (var == "dt_925" or var == "dt_850"):
      values = np.arange(0,13,1.5)
      
    elif (var == "dtdz_925" or var == "dtdz_850"):
      values = np.arange(0,13,1.5)
      #mean_gem = mean_gem*100   # original units: K/dm
      #mean_airs = mean_airs*1000 # original units: K/m
    else:
      values = np.arange(0,101,10)
      colors = ['#ffffff', '#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']

    cmap = mpl.colors.ListedColormap(colors)

    figName = "fig_{0}_{1}_{2}_meanGEM".format(datai, var, per[1])
    
    plotMaps_pcolormesh(mean_gem, figName, values, cmap, lons2d, lats2d)

    figName = "fig_{0}_{1}_{2}_meanAIRS".format(datai, var, per[1])
    plotMaps_pcolormesh(mean_airs, figName, values, cmap, lons2d, lats2d)

    # plotting std
  
        #
    
