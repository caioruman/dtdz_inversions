import numpy as np
import sys
import calendar

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
#from colormath.color_objects import *
#from colormath.color_conversions import convert_color


datai = 2003
dataf = 2015

#def plotMaps_pcolormesh(data, figName, values, mapa, lons2d, lats2d):
def plotMaps_pcolormesh(data, fig, ax, values, mapa, title, lons2d, lats2d, pvalue=None):
  '''
  
  '''
  ax.set_title(title)

  bn = BoundaryNorm(values, ncolors=len(values) - 1)

  b = Basemap(projection='npstere',boundinglat=50,lon_0=-90,resolution='l', round=True, ax=ax)
  x, y = b(lons2d, lats2d)

  img = b.pcolormesh(x, y, data, cmap=mapa, vmin=values[0], vmax=values[-1], norm=bn)
    #img = b.contourf(x, y, newdata2, cmap=cmap, norm=bn, levels=values, extend='both')

  if pvalue is not None:
    ax.contourf(pvalue, hatches=['////'])

  b.drawcoastlines()
  cbar = b.colorbar(img, pad=0.75, ticks=values, ax=ax)
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
  
  return fig
  
"""
    Plot the data: AIRS, GEM-ERA, and GEM-ERA minus AIRS

 -- Periods: DJF and JJA
"""

main_folder = "/pixel/project01/cruman/ModelData/GEM_SIMS"
#main_folder = "/home/caioruman/Documents/McGill/NC/dtdz/"
pickle_folder = "/pixel/project01/cruman/Data/Pickle/t-test"


period = ["DJF", "JJA"]#, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Nov', 'Dec']
#period = [(12, 1, 2), (6, 7, 8)]
#period = [[(12, 1, 2), "DJF"], [(6, 7, 8), "JJA"]]
# lat lon
arq = Dataset('/pixel/project01/cruman/ModelData/GEM_SIMS/PanArctic_0.5d_ERAINT_NOCTEM_RUN.OLD/2014/Inversion_201412.nc','r')
lats2d = arq.variables['lat'][:]
lons2d = arq.variables['lon'][:]

from matplotlib.colors import  ListedColormap

#vars = ['dt_925', 'dt_850', 'dtdz_925', 'dtdz_850', 'tt_850', 'tt_925', 't2m']
vars = ['DT_925', 'FQ_925']

rcps = ['RCP45', 'RCP85']

#vars = ['fq_925', 'fq_850']
for per in period:

  for rcp in rcps:

    for var in vars:
    
    # Open stuff
    # For each RCP, open: current data (mean and std), future data (mean and std), p_value
    # Plot: current data, future data, difference with significance values

    # DT_925_p_GEMCAM_RCP45_19762040_DJF.p
    # DT_925_mean_GEMCAM_RCP85_1976_DJF.p
      mean_1976 = pickle.load( open('{0}/{1}_mean_GEMCAM_{2}_1976_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))
      mean_2040 = pickle.load( open('{0}/{1}_mean_GEMCAM_{2}_2040_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))
      mean_2070 = pickle.load( open('{0}/{1}_mean_GEMCAM_{2}_2070_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))

      std_1976 = pickle.load( open('{0}/{1}_std_GEMCAM_{2}_1976_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))
      std_2040 = pickle.load( open('{0}/{1}_std_GEMCAM_{2}_2040_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))
      std_2070 = pickle.load( open('{0}/{1}_std_GEMCAM_{2}_2070_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))

      p_1976_2040 = pickle.load( open('{0}/{1}_p_GEMCAM_{2}_19762040_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))
      p_1976_2070 = pickle.load( open('{0}/{1}_p_GEMCAM_{2}_19762070_{3}.p'.format(pickle_folder, var, rcp, per), "rb"))

      # Figures for the mean and std of each variable, 

      # Plotting the differences

      figName = "fig_{0}_{1}_{2}".format(datai, var, per[1])
      
      #colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
      colors_diff = ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8',"#ffffff", "#ffffff",'#fee090','#fdae61','#f46d43','#d73027', '#a50026']
      colors_mean = ['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
      #v = abs(max(np.nanmax(data), np.nanmin(data), key=abs))

      if (var == "DT_925" or var == "DT_850"):
        values_diff = np.linspace(-9, 9, len(colors_diff)+1)
        values = np.arange(0,13,1.5)
        if per[1] == "JJA":
          values_diff = np.linspace(-4.5, 4.5, len(colors_diff)+1)
      elif (var == "dtdz_925" or var == "dtdz_850"):
        values_diff = np.linspace(-9, 9, len(colors_diff)+1)
        if per[1] == "JJA":
          values_diff = np.linspace(-4.5, 4.5, len(colors_diff)+1)
        #mean_gem = mean_gem*100   # original units: K/dm
        #mean_airs = mean_airs*1000 # original units: K/m

      else:
        # Frequency plots
        values_diff = np.arange(-60,61,10)
        values = np.arange(0,101,10)

        #mean_gem = mean_gem*100
        colors_mean = ['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']

      
      fig, axs = plt.subplots(3, 3, figsize=(14, 22), frameon=False, dpi=150)

      #  (0,0) (cur_mean)  (1,0) (fut_mean_2040)  (2,0) (fut_mean_2070)
      #  (0,1) (cur_std )  (1,1) (fut_std )       (2,1) (fut_std)
      #  (0,2)             (1,2) (diff)           (2,2)

      # Plotting means:
      cmap = mpl.colors.ListedColormap(colors_mean)

      fig = plotMaps_pcolormesh(mean_1976, fig, axs[0,0], values, cmap, 'Mean Values Current Climate', lons2d, lats2d)
      fig = plotMaps_pcolormesh(mean_2040, fig, axs[1,0], values, cmap, 'Mean Values 2040:2069', lons2d, lats2d)
      fig = plotMaps_pcolormesh(mean_2070, fig, axs[2,0], values, cmap, 'Mean Values 2070:2099', lons2d, lats2d)

      # Plotting STD
      fig = plotMaps_pcolormesh(std_1976, fig, axs[0,1], values, cmap, 'Std Values Current Climate', lons2d, lats2d)
      fig = plotMaps_pcolormesh(std_2040, fig, axs[1,1], values, cmap, 'Std Values 2040:2069', lons2d, lats2d)
      fig = plotMaps_pcolormesh(std_2070, fig, axs[2,1], values, cmap, 'Std Values 2070:2099', lons2d, lats2d)

      # Plotting Differences
      future_2040 = mean_2040.filled(0) - mean_1976
      future_2070 = mean_2070.filled(0) - mean_1976

      p_1976_2040[p_1976_2040 > 0.1] = np.nan
      p_1976_2070[p_1976_2070 > 0.1] = np.nan

      fig = plotMaps_pcolormesh(future_2040, fig, axs[1,1], values, cmap, 'Std Values 2040:2069', lons2d, lats2d, p_1976_2040)
      fig = plotMaps_pcolormesh(future_2070, fig, axs[2,1], values, cmap, 'Std Values 2070:2099', lons2d, lats2d, p_1976_2070)

      plt.subplots_adjust(top=0.75, bottom=0.25)

      plt.savefig('{0}_{1}_{2}_allplot.png'.format(rcp, var, per), pad_inches=0.0, bbox_inches='tight')

      plt.close()
      
      
      
      #plotMaps_pcolormesh(mean_gem, figName, values, cmap, lons2d, lats2d)

      
    
