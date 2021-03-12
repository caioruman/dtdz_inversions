import numpy as np
import pandas as pd
import xarray as xr
import sys
import calendar
import argparse

from collections import OrderedDict

import pickle

import numpy.ma as ma

from glob import glob
from datetime import date, datetime, timedelta

from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn import level_kinds

from netCDF4 import Dataset
import time

from pathlib import Path

'''
Check the inversions in each output step.
Calculate the dtdz and deltaT.
If inversion exists, save the values.
And tries to mimic the AIRS satelite swaths.

AIRS satelite swaths:
 - Passes each area at the same time twice daily: at 1h30 am (ascending) and 1h30 pm (descending)


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
    output_folder = "/home/cruman/scratch/inversion2/{0}".format(exp)

    #eticket = "PAN_ERAI_DEF"
    #eticket = "PAN_CAN85_CT"
    #eticket = "PAN_CAN85_R2"

    # Open the monthly files
    for yy in range(datai, dataf+1):

        for mm in range(1,13):

            
            print(yy, mm)
            #if yy == 1979 and mm == 1:
            #    k = 1
            #else:
            k = 0

            #print("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, yy, mm))
    # Pressure level file
            arq_dp = glob("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, yy, mm))[k]

            # Surface file
            arq_dm = glob("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, yy, mm))[k]

            # Physics file
            #arq_pm = glob("{0}/Samples/{1}_{2}{3:02d}/pm*".format(main_folder, exp, yy, mm))[k]


            r_dp = RPN(arq_dp)
            print(arq_dp)
    #        r_pm = RPN(arq_pm)
            r_dm = RPN(arq_dm)

            # Temperature
            var = r_dp.get_4d_field('TT')
            dates_tt = list(sorted(var.keys()))
            #print(var[dates_tt[0]].keys())
            level_tt = [*var[dates_tt[0]].keys()]
            
            #levels = [lev for lev in var.sorted_levels]

            var_3d = []
            for key in level_tt:
                var_3d.append(np.asarray([var[d][key] for d in dates_tt]))
            tt = np.array(var_3d) + 273.15

            lons2d, lats2d = r_dp.get_longitudes_and_latitudes_for_the_last_read_rec()

            ds = createXarray(lons2d, lats2d, dates_tt)

            ds = addVarXarray(ds, ("TT_850", tt[5,:,:,:], "K"), lons2d, lats2d, dates_tt)
            ds = addVarXarray(ds, ("TT_925", tt[3,:,:,:], "K"), lons2d, lats2d, dates_tt)

            var = r_dp.get_4d_field('GZ')
            dates_tt = list(sorted(var.keys())) 
            level_tt = [*var[dates_tt[0]].keys()]
            var_3d = []
            for key in level_tt:
                var_3d.append(np.asarray([var[d][key] for d in dates_tt]))
            gz = np.array(var_3d)

            ds = addVarXarray(ds, ("GZ_850", gz[5,:,:,:], "dm"), lons2d, lats2d, dates_tt)
            ds = addVarXarray(ds, ("GZ_925", gz[3,:,:,:], "dm"), lons2d, lats2d, dates_tt)
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

            r_dp.close()
            #r_pm.close()
            r_dm.close()

            ds = addVarXarray(ds, ("T2M", tt_dm, "K"), lons2d, lats2d, dates_tt)

            # Resampling the array
            print("starting th resample")
            ds = ds.resample(time="1H", loffset='30min').interpolate("linear")

            print("finish resample")
            # Loading the local time arrays
            t_arr_22 = pickle.load( open( "/home/cruman/scratch/time_array_sample_day_22times_boolean.p", "rb" ) )
            t_arr_24 = pickle.load( open( "/home/cruman/scratch/time_array_sample_day_24times_boolean.p", "rb" ) )

            # apply the algoritm
            # Getting only data where the local time is 1h30
            ini = True
            for k in np.arange(1,len(dates_tt)/8):
                if ini:
                    t_arr = np.vstack((t_arr_22, t_arr_24))
                    ini = False
                else:
                    t_arr = np.vstack((t_arr, t_arr_24))
            

            t_arr[t_arr==0] = np.nan

            # Applying np.nan where t_arr equals np.nan
            print("Applying np.nan where the local time isnt 1h30")
            ds_airs_like = xr.where(t_arr==1, ds, t_arr)

            # Now I can do the inversion calculations.
            print("Calculating inversions")

            deltaT_925, dtdz_925, frequency_925, deltaT_850, dtdz_850, frequency_850 = inversion_calculations(ds.T2M, ds.TT_925, ds.TT_850, ds.GZ_925, ds.GZ_850)

            fname = "{2}/{0}/Inversion_{0}{1:02d}.nc".format(yy, mm, output_folder)

            Path("{0}/{1}".format(output_folder, yy)).mkdir(parents=True, exist_ok=True)

            vars=[("FQ_925", frequency_925, "%"), ("DT_925", deltaT_925, "K"), ("DTDZ_925", dtdz_925, "K/dm"),
                  ("FQ_850", frequency_850, "%"), ("DT_850", deltaT_850, "K"), ("DTDZ_850", dtdz_850, "K/dm"),
                  ("T2M", ds.T2M, "K"), ("TT_925", ds.TT_925, "K"), ("TT_850", ds.TT_850, "K")]

            ds2 = createXarray(lons2d, lats2d, ds.time)
            #
            for var in vars:
                if (var[0] == "FQ_925" or var[0] == "FQ_850"):
                    ds2 = addVarXarray(ds2, (var[0], var[1], var[2]), lons2d, lats2d, ds.time[0])
                else:
                    ds2 = addVarXarray(ds2, (var[0], var[1], var[2]), lons2d, lats2d, ds.time)
            
            print("saving netcdf")
            ds.to_netcdf(fname)

            #sys.exit()
            #r_dp.close()
            #r_pm.close()
            #r_dm.close()

            sys.exit()

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
    # change to xr.where
    deltaT_925 = xr.where(count_bool, deltaT_925, aux)
    dtdz_925 = xr.where(count_bool, dtdz_925, aux)
    #np.copyto(deltaT_925, aux, where=count_bool)
    #np.copyto(dtdz_925, aux, where=count_bool)

    count_bool = np.less_equal(deltaT_850, 0)
    # change to xr.where
    deltaT_850 = xr.where(count_bool, deltaT_850, aux)
    dtdz_850 = xr.where(count_bool, dtdz_850, aux)
    #np.copyto(deltaT_850, aux, where=count_bool)
    #np.copyto(dtdz_850, aux, where=count_bool)
    
#    return mean_dt_time, mean_fr_time
    return deltaT_925, dtdz_925, frequency_925, deltaT_850, dtdz_850, frequency_850


def createXarray(lons2d, lats2d, data_range):
    """
    
    """
    ds = xr.Dataset(
        coords=dict(
            lon=(["y", "x"], lons2d),
            lat=(["y", "x"], lats2d),
            time=data_range,
            #reference_time=reference_time,
        ),
        attrs=dict(description="Temperature Inversion related data."),
    )
    
    # Make the time dimension unlimited when writing to netCDF.
    ds.encoding['unlimited_dims'] = ('time',)

    return ds

def addVarXarray(ds, var_data, lons2d, lats2d, data_range):
    """
    
    """
    if len(data_range) == 1:
        da = xr.DataArray(
            data = var_data[1],
            dims=["y", "x"],
            coords={"lat": (("y", "x"), lats2d), "lon": (("y", "x"), lons2d)},
            attrs  = {
            '_FillValue': np.nan,
            'units'     : var_data[2],
            'missing_value': np.nan,
            }
        )
    else:
        da = xr.DataArray(
            data = var_data[1],
            dims=["time", "y", "x"],
            coords={"lat": (("y", "x"), lats2d), "lon": (("y", "x"), lons2d), "time": data_range},
            attrs  = {
            '_FillValue': np.nan,
            'units'     : var_data[2],
            'missing_value': np.nan,
            }
        )
    
    ds = ds.assign(var=da)
    ds = ds.rename({"var": var_data[0]})

    return ds

if __name__ == '__main__':
    main()
