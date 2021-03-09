import numpy as np
import pandas as pd
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

#from rpn.rpn import RPN
#from rpn.domains.rotated_lat_lon import RotatedLatLon


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

datai = 1986
dataf = 2015

# simulation
exp = "PanArctic_0.5d_ERAINT_NOCTEM_RUN"
#exp = "PanArctic_0.5d_CanHisto_NOCTEM_RUN"

main_folder = "/home/cruman/scratch/glacier2/GEM/Output/{0}".format(exp) #PanArctic_0.5d_ERAINT_NOCTEM_RUN/InversionV2/"
main_folder = "/home/caioruman/Documents/McGill/NC/dtdz/"

#r_MF = RPN("/home/cruman/scratch/glacier/Data/Geophys/PanArctic0.5/pan_artic_mf_0.5")

#Z_surface = np.squeeze(r_MF.variables['MF'][:])
#lons2d, lats2d = r_MF.get_longitudes_and_latitudes_for_the_last_read_rec()

#r_MF.close()

# Sounding Data
#sounding_file = "/home/cruman/project/cruman/Scripts/soundings/inv_list_DJF.dat"

period = ["DJF", "JJA", 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Nov', 'Dec']
period = ["DJF", "JJA", "JFM", "JAS"]
varlist = ["DZ", "FREQ", "ZBAS", "DT", "DQ"]
varlist = ["DT_T", "DT_B", "DT", "FREQ", "FQ_T", "FQ_B", "DT_TG", "DT_BG", "deltaT"]
varlist = ["FQ_DIF"]
#height = [925, 900, 850]
height = ["925_1000"]

from matplotlib.colors import  ListedColormap
# Open the monthly files

for per in period:
    for h in height:

        #read file
        #Inversion_925_1000_ERA_DJF_1986-2015.nc
        #file = "{2}/InversionV2/dtdz/Inversion_{0}_ERA_dtdz_{1}_{3}-{4}.nc".format(h, per, main_folder, datai, dataf)
        file = "{4}/Inversion_{0}_ERA_dtdz_{1}_{2}-{3}.nc".format(h, per, datai, dataf, main_folder)

        print(file)

        arq = Dataset(file, 'r')

#        for var in varlist:
        var = "FQ_DIF"

   #     if var == "deltaT":
        data1 = np.squeeze(arq.variables["DT_T"][:])
  #      else:
        data = np.squeeze(arq.variables[var][:])
        lons2d = np.squeeze(arq.variables["lon"][:])
        lats2d = np.squeeze(arq.variables["lat"][:])

        figName = "{0}_{1}_{2}_{3}_{4}_dtdz".format(var, exp, per, h, datai)
            #figName = "{0}_testeSummer".format(var)

            #open station file
            #sta = open('/home/cruman/project/cruman/Scripts/soundings/inv_list_{0}.dat'.format(per), 'r')

        stations = []
#            for line in sta:
#                aa = line.replace("''", '').split(';')
#                if aa[0] == "Station_number":
#                    continue

                #Station_number;Latitude;Longitude;Inv_00;Inv_P_00;Inv_12;Inv_P_12;Inv_TT;Inv_P_TT;TotalYear;TotalYearTT
#                if var == "FQR12" or var == "DT12":
                    #ksksk
                    #stations.append((float(aa[1]),float(aa[2]),float(aa[5]),float(aa[6])))
#                    stations.append((float(aa[1]),float(aa[2]),float(aa[3]),float(aa[4])))
#                elif var == "FQR0" or var == "DT0":
                    #asdasd
                    #stations.append((float(aa[1]),float(aa[2]),float(aa[3]),float(aa[4])))
#                    stations.append((float(aa[1]),float(aa[2]),float(aa[5]),float(aa[6])))
#                else:
                    #asdasd
#                    stations.append((float(aa[1]),float(aa[2]),float(aa[7]),float(aa[8])))

            #sta.close()


            #
        if var == "DZ" or var == "ZBAS":
            values = np.arange(0,1001,100)
            colors = ['#ffffff', '#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
        elif var == "FREQ" or var == "FQ_B" or var == "FQ_T" or var == "FQ_DIF":
                # Wrong calculation in the algorigtm, must subtract from 100
            values = np.arange(0,101,10)
                #line_colors = cmap(np.linspace(0,100,10))

                #cmap2 = ListedColormap(line_colors)
                #data = (1- data)*100
            data = data*100
                #if var != "FREQ":
                #    # read FREQ
                #    freq = np.squeeze(arq.variables['FREQ'][:])*100
                #    data = data*100/freq


                #colors = ['#ffffff', '#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
                #colors = ['#ffffff','#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506']
            colors= ['#632264', '#342157','#2e5aa7','#2e74ba','#166f48','#79af44','#e9de33','#e9a332','#e3732b','#d74a35','#d1304f']
            colors = ['#ffffff', '#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']
                #values = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])*100
        elif var == "deltaT":
            data2 = np.squeeze(arq.variables["DT_B"][:])

            data = data - data2
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
