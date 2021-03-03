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


datai = 2002
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
# lat lon
arq = Dataset('/pixel/project01/cruman/ModelData/GEM_SIMS/PanArctic_0.5d_ERAINT_NOCTEM_RUN/2014/Inversion_201412.nc','r')
lats2d = arq.variables['lat'][:]
lons2d = arq.variables['lon'][:]

from matplotlib.colors import  ListedColormap
# Open the monthly files
vars = ['dt_925', 'dt_850', 'fq_925', 'fq_850', 'dtdz_925', 'dtdz_850']

for per in period:
  for var in vars:
  
  # Opening the files
    print('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var))
    t_test = pickle.load( open('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "rb"))
    p = pickle.load( open('{0}/{4}_p_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "rb"))
    mean_gem = pickle.load( open('{0}/{4}_mean_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "rb"))
    std_gem = pickle.load( open('{0}/{4}_std_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "rb"))
    mean_airs = pickle.load( open('{0}/{4}_mean_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "rb"))
    std_airs = pickle.load( open('{0}/{4}_std_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per, var), "rb"))
  
    sig = p[p < 0.05]

  # Figures for the mean and std of each variable, 

    figName = "fig_{0}_{1}_{2}".format(datai, per)

    data = mean_gem - mean_airs

    values = np.arange(-12,13,2)
    
    colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
    #colors = [(255,255,255),(255,249,190),(255,223,34),(248,159,28),(243,111,33),(239,66,36),(238,40,35),(208,40,35),(189,36,41),(241,105,160)]
    #colors = np.array(colors)/255.
    v = abs(max(data, key=abs))
    values = np.linspace(-v, v, len(colors))

    cmap = mpl.colors.ListedColormap(colors)

    plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d, var)
    sys.exit()
  
        #
    if var == "DZ" or var == "ZBAS":
      values = np.arange(0,1001,100)
      colors = ['#ffffff', '#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
    elif var == "FREQ" or var == "FQ_B" or var == "FQ_T" or var == "FQ_DIF":
      values = np.arange(0,101,10)
      
      #colors = ['#ffffff', '#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
      #colors = ['#ffffff','#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
      colors= ['#632264', '#342157','#2e5aa7','#2e74ba','#166f48','#79af44','#e9de33','#e9a332','#e3732b','#d74a35','#d1304f']
      colors = ['#ffffff', '#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']
      #values = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])*100
    elif var == "deltaT":
      
      
      values = np.arange(-6,7,1)
      colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']

    else:
      values = np.arange(-12,13,2)
      values = np.arange(0,22,2)
      colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
      colors = [(255,255,255),(255,249,190),(255,223,34),(248,159,28),(243,111,33),(239,66,36),(238,40,35),(208,40,35),(189,36,41),(241,105,160)]
      colors = np.array(colors)/255.
      #colors = colors[::-1]
      #data = data*(-1)


    cmap = mpl.colors.ListedColormap(colors)

    plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d, stations, var)
