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
            
            print(level_tt)
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

            
            #print(levels)
            print(tt_dm.shape)
            print(tt.shape)
            sys.exit()

            # Array containing the inversion base level


            #start_time = time.time()

            datefield = datetime(yy, mm, 1, 0, 0, 0)

            #mean_deltaT, frequency, mean_deltaT_bottom, freq_bot, mean_deltaT_middle, freq_mid, mean_deltaT_top, freq_top, mean_deltaT_bottom_g, freq_bot_g, mean_deltaT_middle_g, freq_mid_g, mean_deltaT_top_g, freq_top_g, mean_diff, freq_diff  = inversion_calculations(tt[0,:,:,:], tt[3,:,:,:], tt[1,:,:,:], tt[2,:,:,:])

            deltaT_925, dtdz_925, deltaT_850, dtdz_850 = inversion_calculations(tt_dm, tt[3,:,:,:], tt[1,:,:,:], tt[2,:,:,:])

            fname = "{2}/InversionV2/Inversion_925_1000_ERA_dtdz_{0}{1:02d}.nc".format(yy, mm, output_folder)
            #vars=[("FREQ", frequency), ("DT", mean_deltaT), ("FQ_B", freq_bot), ("DT_B", mean_deltaT_bottom), ("FQ_M", freq_mid), ("DT_M", mean_deltaT_middle),
            #    ("FQ_T", freq_top), ("DT_T", mean_deltaT_top), ("FQ_BG", freq_bot_g), ("DT_BG", mean_deltaT_bottom_g), ("FQ_MG", freq_mid_g), ("DT_MG", mean_deltaT_middle_g),
            #    ("FQ_TG", freq_top_g), ("DT_TG", mean_deltaT_top_g), ("FQ_DIF", freq_diff), ("DT_DIF", mean_diff)]
    #        vars=[("FREQ", mean_fr), ("DT", mean_dt)]
            save_netcdf(fname, vars, datefield, lats2d, lons2d)
            
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

def replace_nan(data):

    data[data == np.nan] = -999

    return data

def calc_dtdz(deltaT_lvl, deltaT, count, m_deltaT, count_bool, count_n):
    # only using points were there are inversions
    count_bool_inv = np.less_equal(deltaT, 0)   
    aux = deltaT.copy()*np.nan
    # masking points where there arent inversions
    np.copyto(deltaT_lvl, aux, where=count_bool_inv)

    count_bool_lvl_g = np.greater(deltaT_lvl, 0)
    count_bool_lvl = np.less_equal(deltaT_lvl, 0)

    count_l = np.count_nonzero(count_bool_lvl.astype(np.int), axis=0)
    count_g = np.count_nonzero(count_bool_lvl_g.astype(np.int), axis=0)
   
    count_bool_dt = np.greater(m_deltaT, 0)
    aux = m_deltaT.copy()*np.nan
    count_l = count_l.astype(np.float64)
    count_g = count_g.astype(np.float64)

    np.copyto(count_l, aux, where=~count_bool_dt)
    np.copyto(count_g, aux, where=~count_bool_dt)


    # Frequency of each type
    freq_l = (count - count_l)/count_n
    freq_g = (count - count_g)/count_n

    deltaT_lvl_l = deltaT_lvl.copy()
    deltaT_lvl_g = deltaT_lvl.copy()

    # Calculating the means
    np.copyto(deltaT_lvl_l, aux, where=count_bool_lvl)
    np.copyto(deltaT_lvl_g, aux, where=count_bool_lvl_g)

    return freq_l, freq_g, np.nanmean(deltaT_lvl_l, axis=0), np.nanmean(deltaT_lvl_g, axis=0)

def inversion_calculations(base_level_tt, top_level_tt, level975, level950):
    """
        Calculate Inversion Strengh, Inversion Frequency
    """

    # Inversion calculation, without other addons
    deltaT = top_level_tt - base_level_tt

    # 925 - 950
    delta_T_top = top_level_tt - level950

    # 975 - 1000
    delta_T_bottom = level975 - base_level_tt

    # 950 - 925
    delta_T_middle = level950 - level975

    len_time = float(deltaT.shape[0])

    count_bool = np.greater(deltaT, 0)
    count = np.count_nonzero(count_bool.astype(np.int), axis=0)

    frequency = count/len_time

    # mean of deltaT
    aux = deltaT.copy()*np.nan
    count_bool = np.less_equal(deltaT, 0)
    np.copyto(deltaT, aux, where=count_bool)    
    mean_deltaT = np.nanmean(deltaT, axis=0)

    # applying the true/false field to the levels
    np.copyto(delta_T_top, aux, where=count_bool)
    np.copyto(delta_T_bottom, aux, where=count_bool)
    np.copyto(delta_T_middle, aux, where=count_bool)

    diff = delta_T_top - delta_T_bottom
    count_diff_bool = np.greater(diff, 0)
    count_diff = np.count_nonzero(count_diff_bool.astype(np.int), axis=0)

#    count_nan = ~np.isnan(deltaT)
#    count_diff_n = np.count_nonzero(count_nan.astype(np.int), axis=0)
    
#    print(count_diff)
#    print(count)
#    sys.exit()
    freq_diff = count_diff/count
    mean_diff = np.nanmean(diff, axis=0)


    freq_bot, freq_bot_g, mean_deltaT_bottom, mean_deltaT_bottom_g = calc_dtdz(delta_T_bottom, deltaT, count, mean_deltaT, count_bool, len_time)
    freq_mid, freq_mid_g, mean_deltaT_middle, mean_deltaT_middle_g = calc_dtdz(delta_T_middle, deltaT, count, mean_deltaT, count_bool, len_time)
    freq_top, freq_top_g, mean_deltaT_top, mean_deltaT_top_g = calc_dtdz(delta_T_top, deltaT, count, mean_deltaT, count_bool, len_time)


    mean_dt_time = []
    mean_fr_time = []

    trange = np.arange(0,24,3)

    #Calculate means for each timestep output
#    for i in trange:
#        #print(i/3)
#        ii = int(i/3)        
#
#        aux = delta_T[ii::8,:,:]
#
#        aux_fr_count = np.less_equal(aux, 0)
#        aux_fr = np.count_nonzero(aux_fr_count.astype(np.int), axis=0)
#        aux_fr = aux_fr/float(aux.shape[0])
#
#        np.copyto(aux, aux.copy()*np.nan, where=aux_fr_count)
#        aux = np.nanmean(aux, axis=0)
#        mean_dt_time.append(aux)
#
#        count_bool_dt = np.greater_equal(aux, 0)
#
#        aux2 = count_bool_dt.copy()*np.nan
#        np.copyto(aux_fr, aux2, where=~count_bool_dt)
#        mean_fr_time.append(aux_fr)
#

#    return mean_dt_time, mean_fr_time
    return mean_deltaT, frequency, mean_deltaT_bottom, freq_bot, mean_deltaT_middle, freq_mid, mean_deltaT_top, freq_top, mean_deltaT_bottom_g, freq_bot_g, mean_deltaT_middle_g, freq_mid_g, mean_deltaT_top_g, freq_top_g, mean_diff, freq_diff



def save_netcdf(fname, vars, datefield, lat, lon):

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
    tempo = 1
    data_tipo = "hours"

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
        var_nc[:,:,:] = var[1]

    # close the file.
    ncfile.close()
#    data.close()


if __name__ == '__main__':
    main()
