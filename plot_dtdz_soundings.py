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
from matplotlib.colors import ListedColormap

import pickle
#from colormath.color_objects import *
#from colormath.color_conversions import convert_color


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
arq = Dataset('/pixel/project01/cruman/ModelData/GEM_SIMS/PanArctic_0.5d_ERAINT_NOCTEM_RUN.OLD/2014/Inversion_201412.nc','r')
lats2d = arq.variables['lat'][:]
lons2d = arq.variables['lon'][:]

dic = '/pixel/project01/cruman/ModelData/GEM_SIMS/PanArctic_0.5d_ERAINT_NOCTEM_RUN'

# read station data
df = pd.read_csv('inversion_results.csv', sep=';')
df_lat = pd.read_csv('latlonlist_v2.txt', sep=';', header=None, names=['id', 'name', 'lat1', 'lat', 'lon1', 'lon'])

for index, row in df_lat.iterrows():

  # Getting the index of the row
  index_df = df.index[df['stationName'] == row['name']]
  
  # Setting the lat/lon values from the lalonlist file
  df.loc[index_df, 'lat'] = row['lat']
  df.loc[index_df, 'lon'] = row['lon']
    
df = df.dropna()

# Open the monthly files
vars = ['dt_925', 'dt_850', 'dtdz_925', 'dtdz_850', 'tt_850', 'tt_925', 't2m']
vars = ['tt_850', 'tt_925', 't2m', 'fq_850']
vars = ['dt_850', 'dtdz_850', 'tt_850', 'fq_850']

def geo_idx(dd, dd_array, type="lat"):
  '''
    search for nearest decimal degree in an array of decimal degrees and return the index.
    np.argmin returns the indices of minium value along an axis.
    so subtract dd from all values in dd_array, take absolute value and find index of minimum.
    
    Differentiate between 2-D and 1-D lat/lon arrays.
    for 2-D arrays, should receive values in this format: dd=[lat, lon], dd_array=[lats2d,lons2d]
  '''
  if type == "lon" and len(dd_array.shape) == 1:
    dd_array = np.where(dd_array <= 180, dd_array, dd_array - 360)

  if (len(dd_array.shape) < 2):
    geo_idx = (np.abs(dd_array - dd)).argmin()
  else:
    if (dd_array[1] < 0).any():
      dd_array[1] = np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360)

    a = abs( dd_array[0]-dd[0] ) + abs(  np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360) - dd[1] )
    i,j = np.unravel_index(a.argmin(), a.shape)
    geo_idx = [i,j]

  return geo_idx

def getDataModel(mean_gem, std_gem, lat, lon, lats2d, lons2d):

  points = geo_idx([lat, lon], np.array([lats2d, lons2d]))

  m_data = mean_gem[points[0], points[1]]
  std_data = std_gem[points[0], points[1]]

  print(lat, lon, lats2d[points[0], points[1]], lons2d[points[0], points[1]])
  sys.exit()

  return m_data, std_data

#vars = ['fq_925', 'fq_850']
for per in period:

  # read GEM-ERA data:
  arq = Dataset('{0}/AIRS.{1}.2003.2015.nc'.format(dic, per[1]))
  arq_std = Dataset('{0}/AIRS.{1}.2003.2015.std.nc'.format(dic, per[1]))

  for var in vars:
    
    mean_gem = np.squeeze(arq.variables[var.upper()][:])
    std_gem = np.squeeze(arq_std.variables[var.upper()][:])

    print(mean_gem.shape)

  # Opening the files
    per2 = str(per[0]).replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '_')
    #print('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var))
    #t_test = pickle.load( open('{0}/{4}_t_test_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    #p = pickle.load( open('{0}/{4}_p_GEM-ERA_minus_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    #mean_gem = pickle.load( open('{0}/{4}_mean_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    #std_gem = pickle.load( open('{0}/{4}_std_GEM-ERA_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    mean_airs = pickle.load( open('{0}/{4}_mean_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    std_airs = pickle.load( open('{0}/{4}_std_AIRS_{1}{2}_{3}.p'.format(pickle_folder, datai, dataf, per2, var), "rb"))
    #print(mean_airs)
#    print(mean_airs.shape)
#    print(mean_airs)
    #sig = p.copy()
    #sig[p > 0.05] = np.nan

  # Figures for the mean and std of each variable, 

    # Plotting the differences

    figName = "fig_{0}_{1}_{2}".format(datai, var, per[1])
    
    colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', "#ffffff", "#ffffff", '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
    colors = ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8',"#ffffff", "#ffffff",'#fee090','#fdae61','#f46d43','#d73027', '#a50026']
  
    #v = abs(max(np.nanmax(data), np.nanmin(data), key=abs))

    if (var == "dt_925" or var == "dt_850"):
      values = np.linspace(-9, 9, len(colors)+1)
      svar = 'deltaT'
      if per[1] == "JJA":
        values = np.linspace(-4.5, 4.5, len(colors)+1)
    elif (var == "dtdz_925" or var == "dtdz_850"):
      svar = 'dtdz'
      values = np.linspace(-9, 9, len(colors)+1)
      if per[1] == "JJA":
        values = np.linspace(-4.5, 4.5, len(colors)+1)
      mean_gem = mean_gem*100   # original units: K/dm
      #std_gem = std_gem*100
      mean_airs = mean_airs*1000 # original units: K/m
      #std_airs = std_airs*1000
    elif (var == "t2m" or var == "tt_850" or var == "tt_925"):
      values = np.linspace(-9, 9, len(colors)+1)
      values = np.arange(-6, 7, 1)
    else:
      svar = 'freq'
      values = np.linspace(-100, 100, len(colors)+1)
      values = np.arange(-60,61,10)
      mean_gem = mean_gem*100
      mean_airs = mean_airs*100
      print(mean_gem.shape)
      print(mean_airs.shape)

    # Fill lists with station data.
    # Model minus Station data
    if not (var == "t2m" or var == "tt_850" or var == "tt_925"):
      # Filter the season and fill station data list
      lat = []
      lon = []
      data_station = []
      data_model = []
      data_model_std = []
      #print(df.head())
      for index, row in df[df['season'] == per[1]].iterrows():
        #print(index, row)
        lat.append(row['lat'])
        lon.append(row['lon'])
        m_data, std_data = getDataModel(mean_gem, std_gem, row['lat'], row['lon'], lats2d, lons2d)
        data_model.append(m_data)
        data_model_std.append(std_data)
        data_station.append(row[svar])
        df.loc[index, 'model_{0}_mean'.format(var)] = m_data
        df.loc[index, 'model_{0}_std'.format(var)] = std_data


    # Plotting the difference
    # Masking data over land
      data = mean_gem - mean_airs
      cmap = mpl.colors.ListedColormap(colors)

      plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d, lat, lon, data_model, data_station)
    else:
      data = mean_gem - mean_airs
      cmap = mpl.colors.ListedColormap(colors)

      plotMaps_pcolormesh(data, figName, values, cmap, lons2d, lats2d)

    # Plotting the values

    #colors = ['#ffffff', '#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
#    colors = ['#ffffff','#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
    colors = ['#ffffff','#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    if (var == "dt_925" or var == "dt_850"):
      values = np.arange(0,13,1.5)
      cmap = mpl.colors.ListedColormap(colors)
    elif (var == "dtdz_925" or var == "dtdz_850"):
      values = np.arange(0,13,1.5)
      cmap = mpl.colors.ListedColormap(colors)
      mean_gem = mean_gem*100   # original units: K/dm
      mean_airs = mean_airs*1000 # original units: K/m
    elif (var == "t2m" or var == "tt_850" or var == "tt_925"):
      if (per[1] == "DJF"):
        values = np.arange(-39,13,3)
      else:
        values = np.arange(-10,22,2)

      cmap = mpl.cm.get_cmap('viridis', len(values))
      mean_gem = mean_gem - 273.15
    else:
      values = np.arange(0,101,10)
      colors = ['#ffffff', '#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']
      cmap = mpl.colors.ListedColormap(colors)

    figName = "fig_{0}_{1}_{2}_meanGEM".format(datai, var, per[1])
    
    plotMaps_pcolormesh(mean_gem, figName, values, cmap, lons2d, lats2d)

    mean_airs = mean_airs - 273.15
    figName = "fig_{0}_{1}_{2}_meanAIRS".format(datai, var, per[1])
    plotMaps_pcolormesh(mean_airs, figName, values, cmap, lons2d, lats2d)

    # Standard Deviation and Coefficient of Variation
    # values = np.arange(0, 11, 1)

    # figName = "fig_{0}_{1}_{2}_stdGEM".format(datai, var, per[1])
    
    # plotMaps_pcolormesh(std_gem, figName, values, cmap, lons2d, lats2d)

    # values = np.arange(0, 11, 1)

    # figName = "fig_{0}_{1}_{2}_stdAIRS".format(datai, var, per[1])
    # plotMaps_pcolormesh(std_airs, figName, values, cmap, lons2d, lats2d)

    # values = np.arange(0, 2.1, 0.2)

    # figName = "fig_{0}_{1}_{2}_CVGEM".format(datai, var, per[1])

    # plotMaps_pcolormesh(std_gem/mean_gem, figName, values, cmap, lons2d, lats2d)

    # values = np.arange(0, 2.1, 0.2)

    # figName = "fig_{0}_{1}_{2}_CVAIRS".format(datai, var, per[1])
    # plotMaps_pcolormesh(std_airs/mean_airs, figName, values, cmap, lons2d, lats2d)

  arq.close()
    # plotting std

df.to_csv('station_values_with_model.csv', index=False)  
        #
    
