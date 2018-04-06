from GEE_wrappers import GEE_extent
from GEE_wrappers import GEE_pt
import numpy as np
import os
import matplotlib.pyplot as plt

def get_map(minlon, minlat, maxlon, maxlat, outpath,
            sampling=100,
            year=None, month=None, day=None,
            tracknr=None,
            overwrite=False,
            ex_t_mask=None,
            tempfilter=True,
            mask='Globcover',
            masksnow=True):

    maskcorine=False
    maskglobcover=False

    if mask == 'Globcover':
        maskglobcover = True
    elif mask == 'Corine':
        maskcorine = True
    else:
        print(mask + ' is not recognised as a valid land-cover classification')
        return

    if year is not None:

        GEE_interface = GEE_extent(minlon, minlat, maxlon, maxlat, outpath, sampling=sampling)
        # retrieve S1
        GEE_interface.get_S1(year, month, day,
                             tempfilter=tempfilter,
                             applylcmask=maskcorine,
                             mask_globcover=maskglobcover,
                             trackflt=tracknr,
                             masksnow=masksnow)
        # retrieve GLDAS
        GEE_interface.get_gldas()
        # get Globcover
        GEE_interface.get_globcover()
        # get the SRTM
        GEE_interface.get_terrain()

        outname = 'SMCmap_' + \
                  str(GEE_interface.S1_DATE.year) + '_' + \
                  str(GEE_interface.S1_DATE.month) + '_' + \
                  str(GEE_interface.S1_DATE.day)

        # Estimate soil moisture
        GEE_interface.estimate_SM()

        GEE_interface.GEE_2_disk(name=outname, timeout=False)

        GEE_interface = None

    else:

        # if no specific date was specified extract entire time series
        GEE_interface = GEE_extent(minlon, minlat, maxlon, maxlat, outpath, sampling=sampling)

        # get list of S1 dates
        dates = GEE_interface.get_S1_dates(tracknr=tracknr)
        dates = np.unique(dates)

        for dateI in dates:
            # retrieve S1
            GEE_interface.get_S1(dateI.year, dateI.month, dateI.day,
                                 tempfilter=tempfilter,
                                 applylcmask=maskcorine,
                                 mask_globcover=maskglobcover,
                                 trackflt=tracknr,
                                 masksnow=masksnow)
            # retrieve GLDAS
            GEE_interface.get_gldas()
            # get Globcover
            GEE_interface.get_globcover()
            # get the SRTM
            GEE_interface.get_terrain()

            outname = 'SMCmap_' + \
                      str(GEE_interface.S1_DATE.year) + '_' + \
                      str(GEE_interface.S1_DATE.month) + '_' + \
                      str(GEE_interface.S1_DATE.day)

            if overwrite == False and os.path.exists(outpath + outname + '.tif'):
                print(outname + ' already done')
                continue

            GEE_interface.GEE_2_disk(name=outname, timeout=False)

        GEE_interface = None

def get_ts(lon, lat,
           workpath,
           footprint=50,
           masksnow=True,
           calc_anomalies=False,
           create_plots=False):
    """Get S1 soil moisture time-series"""

    # initialize GEE point object
    gee_pt_obj = GEE_pt(lon, lat, workpath, buffer=footprint)
    sm_ts = gee_pt_obj.extr_SM(masksnow=masksnow, calc_anomalies=calc_anomalies)

    # create plots
    if create_plots == True:
        if calc_anomalies == False:
            # plot s1 soil moisture vs gldas_downscaled
            fig, ax = plt.subplots(figsize=(6.5,2.7))
            line1, = ax.plot(sm_ts.index, sm_ts,
                             color='b',
                             linestyle='-',
                             marker='+',
                             label='S1 Soil Moisture',
                             linewidth=0.2)
            ax.set_ylabel('Soil Moisture [%-Vol.]')
            plotname = 's1_sm_' + str(lon) + '_' + str(lat) + '.png'
        else:
            fig, ax = plt.subplots(figsize=(6.5,2.7))
            line1, = ax.plot(sm_ts.index, sm_ts['ANOM'].values,
                             color='r',
                             linestyle='-',
                             marker='+',
                             label='S1 Soil Moisture Anomaly',
                             linewidth=0.2)
            x0 = [sm_ts.index[0], sm_ts.index[-1]]
            y0 = [0, 0]
            line2, = ax.plot(x0, y0,
                             color='k',
                             linestyle='--',
                             linewidth=0.2)
            ax.set_ylabel('Soil Moisture Anomaly [%-Vol.]')
            plt.legend(handles=[line1, line2])
            plotname = 's1_sm_anom' + str(lon) + '_' + str(lat) + '.png'

        plt.setp(ax.get_xticklabels(), fontsize=6)
        plt.savefig(workpath + plotname, dpi=300)

    return sm_ts



