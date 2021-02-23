import numpy as np
import pandas as pd
import sys
import calendar
import argparse

import numpy.ma as ma

from glob import glob
from datetime import date, datetime, timedelta

from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn import level_kinds

from netCDF4 import Dataset
import time
import pickle

from pathlib import Path

'''
Check the inversions in each output step.
Calculate the dtdz and deltaT.
If inversion exists, save the values.

Read:
    - GPHeight
    - TT

Saves:
   - Inversion between 925hpa and 2m temperature. Inversion between 850hPa and 2m temperature.
   - For all outputs with inversion present.

'''

parser=argparse.ArgumentParser(description='Calculate Inversion Stuff', formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument("-op", "--opt-arg", type=str, dest='opcional', help="Algum argumento opcional no programa", default=False)
parser.add_argument("yeari", type=int, help="Initial Year", default=0)
parser.add_argument("yearf", type=int, help="Final Year", default=0)
parser.add_argument("simname", type=str, help="Simulation Name", default=0)
parser.add_argument("eticket", type=str, help="Simulation eticket", default=0)
args=parser.parse_args()

#yeari = args.yeari
#yearf = args.yearf
exp = args.simname
eticket = args.eticket

datai = args.yeari
dataf = args.yearf

def main():

# simulation
    #exp = "PanArctic_0.5d_ERAINT_NOCTEM_RUN"
    #exp = "PanArctic_0.5d_CanHisto_NOCTEM_RUN"
    #exp = "PanArctic_0.5d_CanHisto_NOCTEM_R2"

    #main_folder = "/home/cruman/scratch/glacier2/GEM/Output/{0}".format(exp)
    #main_folder = "/home/cruman/scratch/glacier/GEM/Output/{0}".format(exp)
    # On cedar
    main_folder = "/home/cruman/projects/rrg-sushama-ab/teufel/{0}".format(exp)
    #output_folder = "/home/cruman/projects/rrg-sushama-ab/cruman/{0}".format(exp)
    output_folder = "/home/cruman/scratch/inversion/{0}".format(exp)

    #eticket = "PAN_ERAI_DEF"
    #eticket = "PAN_CAN85_CT"
    #eticket = "PAN_CAN85_R2"

    # Open the monthly files
    for yy in range(datai, dataf+1):

        for mm in range(1,13):

            # Pressure level file
            print(yy, mm)
            #if yy == 1979 and mm == 1:
            #    k = 1
            #else:
            k = 0

            print("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, yy, mm))
    #        print k
            arq_dp = glob("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, yy, mm))[k]
            #print "{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, yy, mm)
            #print arq_dp
            #sys.exit()

            # Surface file
            print("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, yy, mm))
            arq_dm = glob("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, yy, mm))[k]

            # Physics file
            arq_pm = glob("{0}/Samples/{1}_{2}{3:02d}/pm*".format(main_folder, exp, yy, mm))[k]


            r_dp = RPN(arq_dp)
    #        r_pm = RPN(arq_pm)
            r_dm = RPN(arq_dm)


            # Temperature
            var = r_dp.get_4d_field('TT')
            dates_tt = list(sorted(var.keys()))
            #print(var[dates_tt[0]].keys())
            level_tt = [*var[dates_tt[0]].keys()]
            
            #print(level_tt)
            #levels = [lev for lev in var.sorted_levels]

            var_3d = []
            for key in level_tt:
                var_3d.append(np.asarray([var[d][key] for d in dates_tt]))
            tt = np.array(var_3d) + 273.15
            #print(tt.shape)
            #sys.exit()

            var = r_dp.get_4d_field('GZ')
            dates_tt = list(sorted(var.keys())) 
            level_tt = [*var[dates_tt[0]].keys()]
            var_3d = []
            for key in level_tt:
                var_3d.append(np.asarray([var[d][key] for d in dates_tt]))
            gz = np.array(var_3d)
    #        tt = r_dp.variables['TT']

            # Temperature on pressure levels
            #tt = np.squeeze(tt[:])+273.15

            # 2m Temperature
            var = r_dm.get_4d_field('TT', label=eticket)
            dates_tt = list(sorted(var.keys()))
            #key = var[dates_tt[0]].keys()[0]
            key = [*var[dates_tt[0]].keys()][0]
            var_3d = np.asarray([var[d][key] for d in dates_tt])
            tt_dm = var_3d.copy() + 273.15

            lons2d, lats2d = r_dm.get_longitudes_and_latitudes_for_the_last_read_rec()

            tim = tt.shape[1]
            ii = tt.shape[2]
            jj = tt.shape[3]

            #start_time = time.time()

            datefield = datetime(yy, mm, 1, 0, 0, 0)

            #mean_deltaT, frequency, mean_deltaT_bottom, freq_bot, mean_deltaT_middle, freq_mid, mean_deltaT_top, freq_top, mean_deltaT_bottom_g, freq_bot_g, mean_deltaT_middle_g, freq_mid_g, mean_deltaT_top_g, freq_top_g, mean_diff, freq_diff  = inversion_calculations(tt[0,:,:,:], tt[3,:,:,:], tt[1,:,:,:], tt[2,:,:,:])

            deltaT_925, dtdz_925, frequency_925, deltaT_850, dtdz_850, frequency_850 = inversion_calculations(tt_dm, tt[3,:,:,:], tt[5,:,:,:], gz[3,:,:,:], gz[5,:,:,:])
            
            len_time = float(deltaT_925.shape[0])
            #print(deltaT_925)
            #print(dtdz_925)
            #print(frequency_925)

            fname = "{2}/{0}/Inversion_{0}{1:02d}.nc".format(yy, mm, output_folder)
            Path("{0}/{1}".format(output_folder, yy)).mkdir(parents=True, exist_ok=True)

            #save_pickle(yy, mm, output_folder, deltaT_925)
            #pickle.dump(deltaT_925, open('{0}/{1}/inversion_deltaT925_{1}{2:02d}.p'.format(output_folder, yy, mm), "wb"))

            #vars=[("FREQ", frequency), ("DT", mean_deltaT), ("FQ_B", freq_bot), ("DT_B", mean_deltaT_bottom), ("FQ_M", freq_mid), ("DT_M", mean_deltaT_middle),
            #    ("FQ_T", freq_top), ("DT_T", mean_deltaT_top), ("FQ_BG", freq_bot_g), ("DT_BG", mean_deltaT_bottom_g), ("FQ_MG", freq_mid_g), ("DT_MG", mean_deltaT_middle_g),
            #    ("FQ_TG", freq_top_g), ("DT_TG", mean_deltaT_top_g), ("FQ_DIF", freq_diff), ("DT_DIF", mean_diff)]
    #        vars=[("FREQ", mean_fr), ("DT", mean_dt)]
            vars=[("FQ_925", frequency_925), ("DT_925", deltaT_925), ("DTDZ_925", dtdz_925),
                  ("FQ_850", frequency_850), ("DT_850", deltaT_850), ("DTDZ_850", dtdz_850)]

            for var in vars:
                pickle.dump(var[1], open('{0}/{1}/inversion_{3}_{1}{2:02d}.p'.format(output_folder, yy, mm, var[0]), "wb"))

            #save_netcdf(fname, vars, datefield, lats2d, lons2d, len_time)
            sys.exit()
            r_dp.close()
            #r_pm.close()
            r_dm.close()

            #sys.exit()

#exp = "PanArctic_0.5d_CanHisto_NOCTEM_RUN"
#main_folder = "/glacier/caioruman/GEM/Output/{0}".format(exp)
#eticket = "PAN_CAN85_CT"

#r_MF = RPN("/glacier/caioruman/Data/Geophys/PanArctic0.5/pan_artic_mf_0.5")

#Z_surface = np.squeeze(r_MF.variables['MF'][:])

#r_MF.close()

def save_pickle(yy, mm, output_folder, data):

    pickle.dump(data, open('{0}/{1}/inversion_deltaT925_{1}{2:02d}.p'.format(output_folder, yy, mm), "wb"))

    return None

def replace_nan(data):

    data[data == np.nan] = -999

    return data

def inversion_calculations(t2m, tt_925, tt_850, gz_925, gz_850):
    """
        Calculate Inversion Strengh, Inversion Frequency
    """

    # Inversion calculation. 925
    deltaT_925 = tt_925 - t2m
    dtdz_925 = deltaT_925/gz_925

    # Inversion calculation. 850
    deltaT_850 = tt_850 - t2m
    dtdz_850 = deltaT_850/gz_850

    len_time = float(deltaT_925.shape[0])

    count_bool = np.greater(deltaT_925, 0)
    count = np.count_nonzero(count_bool.astype(np.int), axis=0)

    frequency_925 = count/len_time

    count_bool = np.greater(deltaT_850, 0)
    count = np.count_nonzero(count_bool.astype(np.int), axis=0)

    frequency_850 = count/len_time

    # purging negative values
    aux = deltaT_925.copy()*np.nan
    count_bool = np.less_equal(deltaT_925, 0)
    np.copyto(deltaT_925, aux, where=count_bool)
    np.copyto(dtdz_925, aux, where=count_bool)

    count_bool = np.less_equal(deltaT_850, 0)
    np.copyto(deltaT_850, aux, where=count_bool)
    np.copyto(dtdz_850, aux, where=count_bool)
    
#    return mean_dt_time, mean_fr_time
    return deltaT_925, dtdz_925, frequency_925, deltaT_850, dtdz_850, frequency_850


def save_netcdf(fname, vars, datefield, lat, lon, tempo):

    # model to get lat/lon
    #file = "/HOME/caioruman/Scripts/PBL/Inversion/ERA/netcdf/netcdf_model.nc"
#    file = "/home/cruman/Documents/HomeUQAM/Scripts/PBL/Inversion/ERA/netcdf/netcdf_model.nc"
    
#    data = Dataset(file)
    # read lat,lon
    #lat = data.variables['lat'][:]
    #lon = data.variables['lon'][:]

    #print lon
    #lon = np.where(lon > 0, lon, lon + 360)

    nx = 172
    ny = 172
#    tempo = 8
    #tempo = 1
    data_tipo = "3 hourly"

    # Precisa mudar para utilizar o caminho completo
    ncfile = Dataset('{0}'.format(fname), 'w')
    #varcontent.shape = (nlats, nlons)

    # Crio as dimensoes latitude, longitude e tempo
    ncfile.createDimension('x', nx)
    ncfile.createDimension('y', ny)
    ncfile.createDimension('time', tempo)

    # Crio as variaveis latitude, longitude e tempo
    # createVariable( NOMEVAR, TIPOVAR, DIMENSOES )
    lats_nc = ncfile.createVariable('lat', np.dtype("float32").char, ('y','x'))
    lons_nc = ncfile.createVariable('lon', np.dtype("float32").char, ('y','x'))
    time = ncfile.createVariable('time', np.dtype("float32").char, ('time',))

    # Unidades
    lats_nc.units = 'degrees_north'
    lats_nc._CoordinateAxisType = "Lat"
    lons_nc.units = 'degrees_east'
    lons_nc._CoordinateAxisType = "Lon"
    time.units = '{1} since {0}'.format(datefield.strftime('%Y-%m-%d %H:%M'), data_tipo)

    #Writing lat and lon
    lats_nc[:] = lat
    lons_nc[:] = lon

    # write data to variables along record (unlimited) dimension.
    # same data is written for each record.
    for var in vars:

        var_nc = ncfile.createVariable(var[0], np.dtype('float32').char, ('time', 'y', 'x'))
        var_nc.units = "some unit"
        var_nc.coordinates = "lat lon"
        var_nc.grid_desc = "rotated_pole"
        var_nc.cell_methods = "time: point"
        var_nc.missing_value = np.nan
        var_nc = var[1]

    # close the file.
    ncfile.close()
#    data.close()


if __name__ == '__main__':
    main()
