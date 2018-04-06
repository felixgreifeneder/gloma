from GEE_wrappers import GEE_extent
import numpy as np
import os
import ee
import pickle

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

def get_ts(lon, lat, outpath, sampling, tracknr, modelpath, create_plot=True, save_as_txt=True, calc_anomalies=False, name=None):
    """Get S1 soil moisture time-series"""

    # get S1 time-series
    s1_ts = extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=sampling, maskwinter=False, lcmask=False, trackflt=tracknr,
                                 masksnow=False, varmask=False, dual_pol=False)

    # load SVR model
    MLmodel = pickle.load(open(modelpath, 'rb'))

    # calculate temporal statistics
    temp_ts = s1_ts[str(tracknr)][1]['sig0'].astype(np.float)
    temp_ts_lin = np.power(10, temp_ts / 10.)
    k1 = np.mean(np.log(temp_ts_lin))
    k2 = np.std(np.log(temp_ts_lin))

    # create stack
    ts_length = len(temp_ts)

    fmat = np.hstack((temp_ts.reshape(ts_length, 1),
                          np.repeat(k1, ts_length).reshape(ts_length, 1),
                          np.repeat(k2, ts_length).reshape(ts_length, 1)))
    dates = s1_ts[str(tracknr)][0]

    # estimate SMC
    ssm_estimated = np.full(len(dates), -9999, dtype=np.float)
    for i in range(len(dates)):
        nn_model = MLmodel.SVRmodel
        nn_scaler = MLmodel.scaler
        fvect = nn_scaler.transform(fmat[i, :].reshape(1, -1))
        ssm_estimated[i] = nn_model.predict(fvect)

    valid = np.where(ssm_estimated != -9999)
    ssm_ts = pd.Series(ssm_estimated[valid], index=dates[valid], name='S1')
    ssm_ts.sort_index(inplace=True)
    #ssm_ts = ssm_ts

    if calc_anomalies == True:
        # get GLDAS timeseries
        gldas_ts = extr_GLDAS_ts_GEE(lon, lat, bufferSize=sampling)

        # correct GLDAS for local variations
        gldas_ts_matched = df_match(ssm_ts, gldas_ts, window=0.5)
        gldas_s1 = pd.concat([ssm_ts, gldas_ts_matched['GLDAS']], axis=1, join='inner').dropna()
        lin_fit = LinearRegression()
        lin_fit.fit(gldas_s1['GLDAS'].values.reshape(-1,1), gldas_s1['S1'].values)
        # calibrate GLDAS
        gldas_tmp = lin_fit.predict(gldas_ts.values.reshape(-1,1))
        gldas_ts = pd.Series(gldas_tmp, index=gldas_ts.index)

        # plot s1 soil moisture vs gldas_downscaled
        plt.figure()
        tmp = pd.concat([gldas_s1['S1'], pd.Series(lin_fit.predict(gldas_s1['GLDAS'].values.reshape(-1, 1)),
                                                   index=gldas_s1['GLDAS'].index, name='GLDAS')], axis=1, join='inner')
        tmp.plot(ylim=(5, 40), figsize=(6.3, 2.6))
        plt.ylabel('Soil Moisture [%]')
        plt.title(name)
        plt.minorticks_on()
        plt.show()
        outfile = outpath + 's1ts_gldasPredts' + str(lon) + '_' + str(lat)
        plt.savefig(outfile + '.png')
        plt.close()
        plt.figure()
        gldas_s1.plot(ylim=(5, 40), figsize=(6.3, 2.6))
        plt.ylabel('Soil Moisture [%]')
        plt.title(name)
        plt.show()
        outfile = outpath + 's1ts_gldasts' + str(lon) + '_' + str(lat)
        plt.savefig(outfile + '.png')
        plt.close()

        # calculate climatology
        gldas_clim = anomaly.calc_climatology(gldas_ts, moving_avg_clim=30)
        gldas_clim = pd.DataFrame(pd.Series(gldas_clim, name='S1'))
        # plot climatology
        plt.figure()
        gldas_clim.plot(ylim=(5,40), figsize=(6.3, 2.6))
        plt.ylabel('Soil Moisture [%]')
        plt.xlabel('DOY')
        plt.title('Climatology: ' + name)
        plt.show()
        outfile = outpath + 'climatology' + str(lon) + '_' + str(lat)
        plt.savefig(outfile + '.png')
        plt.close()

        # plot anomalies
        fig, axs = plt.subplots(figsize=(6.3, 2.6))
        plot_clim_anom(ssm_ts, clim=gldas_clim, axes=[axs])
        for tick in axs.get_xticklabels():
             tick.set_rotation(45)
             tick
        outfile = outpath + 's1ts_anomalies' + str(lon) + '_' + str(lat)

        plt.title(name)
        plt.ylabel('Soil Moisture [%]')
        plt.ylim((5, 40))
        plt.savefig(outfile + '.png')
        plt.close()

        # save gldas time-series
        outfile = outpath + 'gldasts' + str(lon) + '_' + str(lat)
        csvout = np.array([m.strftime("%d/%m/%Y") for m in gldas_ts_matched.index], dtype=np.str)
        csvout2 = np.array(gldas_ts_matched['GLDAS'], dtype=np.str)
        # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
        with open(outfile + '.csv', 'w') as text_file:
            [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

    if create_plot == True:
        plt.figure(figsize=(6.3, 2.6))
        # plt.plot(xx,yy)
        # plt.show()
        ssm_ts.plot(ylim=(5,40))
        plt.ylabel('Soil Moisture [%]')
        plt.title(name)
        plt.show()

        #if name == None:
        outfile = outpath + 's1ts' + str(lon) + '_' + str(lat)
        #else:
        #    outfile = outpath + 's1ts_' + name

        plt.savefig(outfile + '.png')
        plt.close()

    if save_as_txt == True:
        csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts.index], dtype=np.str)
        csvout2 = np.array(ssm_ts, dtype=np.str)
        # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
        with open(outfile + '.csv', 'w') as text_file:
            [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

    print("Done")
    return (ssm_ts, gldas_clim)

