import numpy as np
import sys
import calendar
import argparse

from glob import glob
from datetime import date, datetime, timedelta

from netCDF4 import Dataset
import time
import pickle

from pathlib import Path

'''
Check the inversions in each output step.
Calculate the dtdz and deltaT.
If inversion exists, save the values.
Now for the AIRS dataset
Combines the Ascending and Descending phases

Read:
    - GPHeight
    - TT_A

Saves:
   - Inversion between 925hpa and 2m temperature. Inversion between 850hPa and 2m temperature.
   - For all outputs with inversion present.

'''

parser=argparse.ArgumentParser(description='Calculate Inversion Stuff', formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument("-op", "--opt-arg", type=str, dest='opcional', help="Algum argumento opcional no programa", default=False)
parser.add_argument("yeari", type=int, help="Initial Year", default=0)
parser.add_argument("yearf", type=int, help="Final Year", default=0)
#parser.add_argument("simname", type=str, help="Simulation Name", default=0)
#parser.add_argument("eticket", type=str, help="Simulation eticket", default=0)
args=parser.parse_args()

#yeari = args.yeari
#yearf = args.yearf
#exp = args.simname
#eticket = args.eticket

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
    main_folder = "netcdf"
    main_folder = "/pixel/project01/cruman/Data/AIRS/AIRS_daily"
    #output_folder = "/home/cruman/projects/rrg-sushama-ab/cruman/{0}".format(exp)
    #output_folder = "/home/cruman/scratch/inversion/{0}".format(exp)
    output_folder = "/pixel/project01/cruman/Data/AIRS/AIRS_daily/Inversion"

    #AIRS.2008.01.01.L3.RetStd001.v7.0.3.0.G20189141133.hdf.nc4

    #eticket = "PAN_ERAI_DEF"
    #eticket = "PAN_CAN85_CT"
    #eticket = "PAN_CAN85_R2"

    # Open the monthly files
    for yy in range(datai, dataf+1):

        for mm in range(1,13):

            print(yy, mm)

            #print("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, yy, mm))
    #        print k
            arq = sorted(glob("{0}/AIRS.{1}.{2:02d}.*.L3.RetStd001.v7.0.3.0.*.hdf.nc4".format(main_folder, yy, mm)))
            print(arq)
            #sys.exit()
            tt_aux = []
            tt_std_aux = []
            tt_ct_aux = []
            t2m_aux = []
            t2m_std_aux = []
            t2m_ct_aux = []
            gph_aux = []

            for f in arq:
                print(f)
                nc = Dataset(f, 'r')

                tt_a = nc.variables["Temperature_A"][:].filled(fill_value=np.nan)
                tt_d = nc.variables["Temperature_D"][:].filled(fill_value=np.nan)

                tt_std_a = nc.variables["Temperature_A_sdev"][:].filled(fill_value=np.nan)
                tt_std_d = nc.variables["Temperature_D_sdev"][:].filled(fill_value=np.nan)

                tt_ct_a = nc.variables["Temperature_A_ct"][:].filled(fill_value=np.nan)
                tt_ct_d = nc.variables["Temperature_D_ct"][:].filled(fill_value=np.nan)

                t2m_a = nc.variables["SurfAirTemp_A"][:].filled(fill_value=np.nan)
                t2m_d = nc.variables["SurfAirTemp_D"][:].filled(fill_value=np.nan)

                t2m_std_a = nc.variables["SurfAirTemp_A_sdev"][:].filled(fill_value=np.nan)
                t2m_std_d = nc.variables["SurfAirTemp_D_sdev"][:].filled(fill_value=np.nan)

                t2m_ct_a = nc.variables["SurfAirTemp_A_ct"][:].filled(fill_value=np.nan)
                t2m_ct_d = nc.variables["SurfAirTemp_D_ct"][:].filled(fill_value=np.nan)
                #t2m_ct_a[t2m_ct_a==-9999] = 0
                #t2m_ct_d[t2m_ct_d==-9999] = 0

                gph_a = nc.variables["GPHeight_A"][:].filled(fill_value=np.nan)
                gph_d = nc.variables["GPHeight_D"][:].filled(fill_value=np.nan)
                
                lons2d = nc.variables["Longitude"][:]
                lats2d = nc.variables["Latitude"][:]

                #print(lons2d.shape)
                #sys.exit()

                # Wrong, I just use the two values. No need to merge
                #tt, tt_std, tt_ct = merge_values(tt_a, tt_d, tt_std_a, tt_std_d, tt_ct_a, tt_ct_d)
                #t2m, t2m_std, t2m_ct = merge_values(t2m_a, t2m_d, t2m_std_a, t2m_std_d, t2m_ct_a, t2m_ct_d)
                
                #gph = np.nanmean(np.stack((gph_a, gph_d)), axis=0)

                tt_aux.append(tt_a)
                tt_aux.append(tt_d)
                tt_std_aux.append(tt_std_a)
                tt_std_aux.append(tt_std_d)
                tt_ct_aux.append(tt_ct_a)
                tt_ct_aux.append(tt_ct_d)
                t2m_aux.append(t2m_a)
                t2m_aux.append(t2m_d)
                t2m_std_aux.append(t2m_std_a)
                t2m_std_aux.append(t2m_std_d)
                t2m_ct_aux.append(t2m_ct_a)
                t2m_ct_aux.append(t2m_ct_d)
                gph_aux.append(gph_a)
                gph_aux.append(gph_d)

            tt = np.array(tt_aux)
            tt_std = np.array(tt_std_aux)
            tt_ct = np.array(tt_ct_aux)
            t2m = np.array(t2m_aux)
            t2m_std = np.array(t2m_std_aux)
            t2m_ct = np.array(t2m_ct_aux)
            gph = np.array(gph_aux)                      
            #print(t2m.shape)
            #print(gph.shape)
            #print(tt.shape)
            #(time, z, lat, lon)
            #sys.exit()
            
            datefield = datetime(yy, mm, 1, 0, 0, 0)

            deltaT_925, dtdz_925, frequency_925, deltaT_850, dtdz_850, frequency_850 = inversion_calculations(t2m, tt[:,1,:,:], tt[:,2,:,:], gph[:,1,:,:], gph[:,2,:,:])
            
            len_time = float(deltaT_925.shape[0])
            #print(deltaT_925)
            #print(dtdz_925)
            #print(frequency_925)

            fname = "{2}/{0}/Inversion_{0}{1:02d}.nc".format(yy, mm, output_folder)
            Path("{0}/{1}".format(output_folder, yy)).mkdir(parents=True, exist_ok=True)

            #save_pickle(yy, mm, output_folder, deltaT_925)
            #pickle.dump(deltaT_925, open('{0}/{1}/inversion_deltaT925_{1}{2:02d}.p'.format(output_folder, yy, mm), "wb"))

            vars=[("FQ_925", frequency_925), ("DT_925", deltaT_925), ("DTDZ_925", dtdz_925),
                ("FQ_850", frequency_850), ("DT_850", deltaT_850), ("DTDZ_850", dtdz_850),
                ("T2M", t2m), ("TT_925", tt[:,1,:,:]), ("TT_850", tt[:,2,:,:])]

            #
            #for var in vars:
            #    pickle.dump(var[1], open('{0}/{1}/inversion_{3}_{1}{2:02d}.p'.format(output_folder, yy, mm, var[0]), "wb"))

            save_netcdf_1d(fname, vars, datefield, lats2d, lons2d, len_time, t2m)
            #sys.exit()
                

            #sys.exit()

#exp = "PanArctic_0.5d_CanHisto_NOCTEM_RUN"
#main_folder = "/glacier/caioruman/GEM/Output/{0}".format(exp)
#eticket = "PAN_CAN85_CT"

#r_MF = RPN("/glacier/caioruman/Data/Geophys/PanArctic0.5/pan_artic_mf_0.5")

#Z_surface = np.squeeze(r_MF.variables['MF'][:])

#r_MF.close()

def merge_values(var1, var2, std1, std2, ct1, ct2):
    var1[var1==-9999] = np.nan
    var2[var2==-9999] = np.nan

    var = np.nanmean(np.stack((var1, var2)), axis=0)

    std1[std1==-9999] = np.nan
    std2[std1==-9999] = np.nan
    
    # average of the std: sqrt of the average of the variance
    std = np.sqrt(np.nanmean(np.stack((np.power(var1, 2), np.power(var2, 2))), axis=0))

    ct1[ct1==-9999] = 0
    ct2[ct2==-9999] = 0

    ct = ct1+ct2
    
    return var, std, ct

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
    count = np.count_nonzero(count_bool.astype(int), axis=0)

    frequency_925 = count/len_time

    count_bool = np.greater(deltaT_850, 0)
    count = np.count_nonzero(count_bool.astype(int), axis=0)

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


def save_netcdf_1d(fname, vars, datefield, lat, lon, tempo, tt):

    ny = len(lat)
    nx = len(lon)
#    tempo = 8
    #tempo = 1
    data_tipo = "hours"
    print(lon.shape)

    # Precisa mudar para utilizar o caminho completo
    ncfile = Dataset('{0}'.format(fname), 'w', format='NETCDF4')
    #ncfile = f.createGroup('Inversion AIRS Data')
    #varcontent.shape = (nlats, nlons)

    # Crio as dimensoes latitude, longitude e tempo
    ncfile.createDimension('lon', nx)
    ncfile.createDimension('lat', ny)
    ncfile.createDimension('time', None)

    # Crio as variaveis latitude, longitude e tempo
    # createVariable( NOMEVAR, TIPOVAR, DIMENSOES )
    lats_nc = ncfile.createVariable('lat', 'f4', ('lat',))
    lons_nc = ncfile.createVariable('lon', 'f4', ('lon',))
    time = ncfile.createVariable('time', 'i4', ('time',))

    #print(lat)
    #print(lats_nc)
    lats_nc[:] = lat
    lons_nc[:] = lon
    time[0] = datefield.toordinal()

    # Unidades
    lats_nc.units = 'degrees_north'
    lats_nc._CoordinateAxisType = "Lat"
    lons_nc.units = 'degrees_east'
    lons_nc._CoordinateAxisType = "Lon"
    time.units = '{1} since {0}'.format(datefield.strftime('%Y-%m-%d %H:%M'), data_tipo)

    #Writing lat and lon
    
    # write data to variables along record (unlimited) dimension.
    # same data is written for each record.
    for var in vars:
        #print(var[1].shape)
        #print(var[0])
        var_nc = ncfile.createVariable(var[0], np.dtype('float32').char, ('time', 'lat', 'lon'))
        var_nc.units = "some unit"
        var_nc.coordinates = "lat lon"
        #var_nc.grid_desc = "rotated_pole"
        var_nc.cell_methods = "time: point"
        var_nc.missing_value = np.nan
        if (var[0] == "FQ_925" or var[0] == "FQ_850"):
#            print(var[1])
            var_nc[0,:,:] = var[1]     
            #var_nc.time = 1      
        else:
            var_nc[:,:,:] = var[1]

    # close the file.
    ncfile.close()
#    data.close()


if __name__ == '__main__':
    main()
