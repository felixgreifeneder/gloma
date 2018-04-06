__author__ = 'felix'

import numpy as np
import sgrt.common.grids.Equi7Grid as Equi7
import sgrt.common.utils.SgrtTile as SgrtTile
from osgeo import gdal
from osgeo.gdalconst import *
from netCDF4 import Dataset, num2date
from datetime import datetime
import ee
import datetime as dt
import time
from sgrt_tools.access_google_drive import gdrive
import pandas as pd
import math

# extract ERA-Land/Interim soil moisture at position lat lon
def extr_ERA_SMC(path, lon, lat):
    # load dataset
    eraFile = Dataset(path, 'r')

    eraLon = np.array(eraFile.variables['longitude'])
    eraLat = np.array(eraFile.variables['latitude'])
    eratmp = eraFile.variables['time']
    eratmp =  num2date(eratmp[:], units=eratmp.units, calendar=eratmp.calendar)
    eraTime = np.zeros(len(eratmp))
    for i in range(len(eratmp)): eraTime[i] = eratmp[i].toordinal()
    #eraTime = np.array(eraFile.variables['time'])
    #eraSMC = np.array(eraFile.variables['swvl1'])

    dist = np.ones([len(eraLat), len(eraLon)])
    dist[:,:] = 9999

    # calculate distance between each grid point and lon, lat
    for i in range(len(eraLon)):
        for j in range(len(eraLat)):
            dist[j,i] = np.sqrt(np.square(eraLon[i]-lon) + np.square(eraLat[j]-lat))

    # # get the nearest pixel
    # nearest = np.unravel_index(dist.argmin(), dist.shape)
    #
    # SMCts = np.zeros([len(eraTime), 2])
    # SMCts[:,1] = np.array(eraFile.variables['swvl1'][:, nearest[0], nearest[1]])
    # for i in range(len(eraTime)): SMCts[i,0] = eraTime[i]


    # get the four nearest pixels
    fourNearest = []
    weights = []
    for i in range(4):
        tmp = np.unravel_index(dist.argmin(), dist.shape)
        fourNearest.append([tmp[0], tmp[1]])
        weights.append(dist[tmp[0], tmp[1]])
        dist[tmp[0], tmp[1]] = 9999

    weights = np.array(weights)
    weights = weights.max() - weights

    # retrieve SMC
    SMCtsw = np.zeros([len(eraTime), 2])
    SMCtsw[1,:] = -9999
    SMCtsp = np.zeros([len(eraTime), 5])
    SMCtsp[1::,:] = -9999
    #load smc data
    for i in range(4):
        SMCtsp[:,i+1] = np.array(eraFile.variables['swvl1'][:,fourNearest[i][0], fourNearest[i][1]])

    # compute weighted average
    for i in range(len(eraTime)):
        SMCtsw[i,1] = np.sum(SMCtsp[i,1::]*weights) / weights.sum()
        SMCtsw[i,0] = eraTime[i]
        #SMCts[1,i] = SMCw

    eraFile.close()
    return(SMCtsw)


def multitemporalDespeckle(images, radius, units, opt_timeWindow=None):

    def mapMeanSpace(i):
        reducer = ee.Reducer.mean()
        kernel = ee.Kernel.square(radius, units)
        mean = i.reduceNeighborhood(reducer, kernel).rename(bandNamesMean)
        ratio = i.divide(mean).rename(bandNamesRatio)
        return(i.addBands(mean).addBands(ratio))

    if opt_timeWindow == None:
        timeWindow = dict(before=-3, after=3, units='month')
    else:
        timeWindow = opt_timeWindow

    bandNames = ee.Image(images.first()).bandNames()
    bandNamesMean = bandNames.map(lambda b: ee.String(b).cat('_mean'))
    bandNamesRatio = bandNames.map(lambda b: ee.String(b).cat('_ratio'))

    # compute spatial average for all images
    meanSpace = images.map(mapMeanSpace)

    # computes a multi-temporal despeckle function for a single image

    def multitemporalDespeckleSingle(image):
        t = image.date()
        fro = t.advance(ee.Number(timeWindow['before']), timeWindow['units'])
        to = t.advance(ee.Number(timeWindow['after']), timeWindow['units'])

        meanSpace2 = ee.ImageCollection(meanSpace).select(bandNamesRatio).filterDate(fro, to) \
                .filter(ee.Filter.eq('relativeOrbitNumber_start', image.get('relativeOrbitNumber_start')))

        b = image.select(bandNamesMean)

        return(b.multiply(meanSpace2.sum()).divide(meanSpace2.count()).rename(bandNames)).set('system:time_start', image.get('system:time_start'))

    return meanSpace.map(multitemporalDespeckleSingle)


def extr_USGS_LC(lon, lat, bufferSize=20):

    ee.Initialize()
    lc_image = ee.Image(ee.ImageCollection("USGS/NLCD").toList(100).get(-1))
    roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    lc = lc_image.reduceRegion(ee.Reducer.mode(), roi).getInfo()
    return(lc)


def extr_MODIS_MOD13Q1_ts_GEE(lon,lat, bufferSize=20):

    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

        # mask image
        immask = image.select('SummaryQA').eq(ee.Image(0))
        image = image.updateMask(immask)

        reduced_img_data = image.reduceRegion(ee.Reducer.median(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data})

    # load collection
    gee_l8_collection = ee.ImageCollection('MODIS/006/MOD13Q1')

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    gee_l8_fltd = gee_l8_collection.filterBounds(gee_roi)

    # extract time series
    gee_l8_mpd = gee_l8_fltd.map(createAvg)
    tmp = gee_l8_mpd.getInfo()

    EVI = np.array([x['properties']['result']['EVI'] for x in tmp['features']], dtype=np.float)

    ge_dates = np.array([datetime.strptime(x['id'], '%Y_%m_%d') for x in tmp['features']])

    valid = np.where(np.isfinite(EVI))

    # cut out invalid values
    EVI = EVI[valid]
    ge_dates = ge_dates[valid]

    return ((ge_dates, {'EVI': EVI}))


def extr_MODIS_MOD13Q1_array_reGE(minlon, minlat, maxlon, maxlat, year, month, day):

    ee.Initialize()

    def mask(image):
        # mask image
        immask = image.select('SummaryQA').eq(ee.Image(0))
        image = image.updateMask(immask)
        return (image)

    def mosaic_custom(image, mosaic):
        tmpmosaic = ee.Image(mosaic)
        tmpimage = ee.Image(image)
        return(tmpmosaic.where(tmpimage.select('SummaryQA').eq(0), tmpimage))

    # load collection
    modis_collection = ee.ImageCollection('MODIS/006/MOD13Q1')

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])
    # filter
    doi = dt.date(year=year, month=month, day=day)
    sdate = doi - dt.timedelta(days=100)
    edate = doi + dt.timedelta(days=100)

    modis_fltd = modis_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))
    # modis_fltd_before = modis_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), doi.strftime('%Y-%m-%d'))
    # modis_fltd_after = modis_collection.filterBounds(roi).filterDate(doi.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))
    #
    # # get EVI composites before and after the day-of-interest
    # modis_img_before = ee.Image(modis_fltd_before.iterate(mosaic_custom, ee.Image(modis_fltd_before.first())))
    # tmp = modis_fltd_after.toList(500).reverse()
    # if tmp.length().getInfo() == 0:
    #     raise Exception('Empty list!')
    # modis_img_after = ee.Image(tmp.iterate(mosaic_custom, ee.Image(tmp.get(0))))
    #
    # # interpolate EVI for the day-of-interest
    # doy_aft = modis_img_after.select('DayOfYear').where(modis_img_after.select('DayOfYear').lt(modis_img_before.select('DayOfYear')),
    #                                                     modis_img_after.select('DayOfYear').add(ee.Image(365)))
    # doy_bef = modis_img_before.select('DayOfYear').where(modis_img_before.select('DayOfYear').lt(doi.timetuple().tm_yday),
    #                                                      modis_img_before.select('DayOfYear').add(ee.Image(365)))
    #
    # doy_diff = doy_aft.subtract(doy_bef)
    # evi_diff = modis_img_after.select('EVI').subtract(modis_img_before.select('EVI'))
    # coeff = evi_diff.divide(doy_diff)
    #
    # mask_combo = modis_img_after.select('SummaryQA').eq(0).And(modis_img_before.select('SummaryQA').eq(0))
    # if doi.year != sdate.year:
    #     doi_nmbr = doi.timetuple().tm_yday + 365
    # else:
    #     doi_nmbr = doi.timetuple().tm_yday
    # doi_img = ee.Image(doi_nmbr).subtract(doy_bef)
    #
    # evi_interpolated = modis_img_before.select('EVI').add(doi_img.multiply(coeff)).updateMask(mask_combo).clip(roi)

    # create a list of availalbel dates
    tmp = modis_fltd.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.datetime.strptime(x, '%Y_%m_%d').date() for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter collection for respective dates
    edate = date_selected + dt.timedelta(days=1)
    modis_fltd = modis_fltd.filterDate(date_selected.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # mosaic scenes
    modis_fltd = modis_fltd.map(mask)
    modis_mosaic = ee.Image(modis_fltd.mosaic().clip(roi))

    # return (evi_interpolated, doi)
    return (modis_mosaic, date_selected)


def extr_GLDAS_ts_GEE(lon, lat, bufferSize=20, yearlist=None):

    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)


        reduced_img_data = image.reduceRegion(ee.Reducer.median(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data})

    if yearlist == None:
        #yearlist = range(1987,2018)
        yearlist = range(2011,2018)

    SM_list = list()

    for iyear in yearlist:
        # ee.Reset()
        # ee.Initialize()
        print(iyear)
        # load collection
        if iyear < 2000:
            GLDAS_collection = ee.ImageCollection('NASA/GLDAS/V20/NOAH/G025/T3H').select('SoilMoi0_10cm_inst')
        else:
            GLDAS_collection = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H').select('SoilMoi0_10cm_inst')
        GLDAS_collection = GLDAS_collection.filterDate(str(iyear)+'-01-01', str(iyear)+'-12-31')
        GLDAS_collection = GLDAS_collection.filter(ee.Filter.calendarRange(16,18,'hour'))

        # clip
        roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        GLDAS_collection = GLDAS_collection.map(lambda image: image.clip(roi))

        # extract time series
        time_series = GLDAS_collection.map(createAvg)
        tmp = time_series.getInfo()

        SM = np.array([x['properties']['result']['SoilMoi0_10cm_inst'] for x in tmp['features']], dtype=np.float)

        ge_dates = np.array([datetime.strptime(x['id'], 'A%Y%m%d_%H%M') for x in tmp['features']])

        valid = np.where(np.isfinite(SM))

        # cut out invalid values
        SM = SM[valid]
        ge_dates = ge_dates[valid]

        SM_series = pd.Series(SM, index=ge_dates, copy=True, name='GLDAS')

        SM_list.append(SM_series)

    return (pd.concat(SM_list))


def extr_L8_ts_GEE(lon, lat, bufferSize=20):

    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

        # mask image
        immask = image.select('cfmask').eq(ee.Image(0))
        image = image.updateMask(immask)

        reduced_img_data = image.reduceRegion(ee.Reducer.median(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data})

    def setresample(image):
        image = image.resample()
        return (image)


    # load collection
    gee_l8_collection = ee.ImageCollection('LANDSAT/LC8_SR').map(setresample)

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    gee_l8_fltd = gee_l8_collection.filterBounds(gee_roi)

    # extract time series
    gee_l8_mpd = gee_l8_fltd.map(createAvg)
    tmp = gee_l8_mpd.getInfo()

    b1 = np.array([x['properties']['result']['B1'] for x in tmp['features']], dtype=np.float)
    b2 = np.array([x['properties']['result']['B2'] for x in tmp['features']], dtype=np.float)
    b3 = np.array([x['properties']['result']['B3'] for x in tmp['features']], dtype=np.float)
    b4 = np.array([x['properties']['result']['B4'] for x in tmp['features']], dtype=np.float)
    b5 = np.array([x['properties']['result']['B5'] for x in tmp['features']], dtype=np.float)
    b6 = np.array([x['properties']['result']['B6'] for x in tmp['features']], dtype=np.float)
    b7 = np.array([x['properties']['result']['B7'] for x in tmp['features']], dtype=np.float)

    ge_dates = np.array([datetime.strptime(x['id'][9::], '%Y%j') for x in tmp['features']])

    valid = np.where(np.isfinite(b2))

    # cut out invalid values
    b1 = b1[valid]
    b2 = b2[valid]
    b3 = b3[valid]
    b4 = b4[valid]
    b5 = b5[valid]
    b6 = b6[valid]
    b7 = b7[valid]
    ge_dates = ge_dates[valid]

    return((ge_dates, {'B1': b1,
                       'B2': b2,
                       'B3': b3,
                       'B4': b4,
                       'B5': b5,
                       'B6': b6,
                       'B7': b7}))


def extr_L8_array_reGE(minlon, minlat, maxlon, maxlat, year, month, day):

    ee.Initialize()

    def setresample(image):
        image = image.resample()
        return (image)

    def mask(image):
        # mask image
        immask = image.select('cfmask').eq(ee.Image(0))
        image = image.updateMask(immask)
        return(image)


    # load collection
    gee_l8_collection = ee.ImageCollection('LANDSAT/LC8_SR').map(setresample)

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])

    # filter
    doi = dt.date(year=year, month=month, day=day)
    sdate = doi - dt.timedelta(days=30)
    edate = doi + dt.timedelta(days=30)
    gee_l8_fltd = gee_l8_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # create a list of availalbel dates
    tmp = gee_l8_fltd.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.datetime.strptime(x[9::], '%Y%j').date() for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter collection for respective dates
    edate = date_selected + dt.timedelta(days=1)
    gee_l8_fltd = gee_l8_fltd.filterDate(date_selected.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # mosaic scenes
    gee_l8_fltd = gee_l8_fltd.map(mask)
    gee_l8_mosaic = ee.Image(gee_l8_fltd.mosaic().clip(roi))

    return(gee_l8_mosaic)


def extr_L8_array(minlon, minlat, maxlon, maxlat, year, month, day, workdir, sampling):

    ee.Initialize()

    def setresample(image):
        image = image.resample()
        return (image)

    def mask(image):
        # mask image
        immask = image.select('cfmask').eq(ee.Image(0))
        image = image.updateMask(immask)
        return(image)


    # load collection
    gee_l8_collection = ee.ImageCollection('LANDSAT/LC8_SR').map(setresample)

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])

    # filter
    doi = dt.date(year=year, month=month, day=day)
    sdate = doi - dt.timedelta(days=30)
    edate = doi + dt.timedelta(days=30)
    gee_l8_fltd = gee_l8_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # create a list of availalbel dates
    tmp = gee_l8_fltd.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.datetime.strptime(x[9::], '%Y%j').date() for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter collection for respective dates
    edate = date_selected + dt.timedelta(days=1)
    gee_l8_fltd = gee_l8_fltd.filterDate(date_selected.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # mosaic scenes
    gee_l8_fltd = gee_l8_fltd.map(mask)
    gee_l8_mosaic = ee.Image(gee_l8_fltd.mosaic().clip(roi))

    GEtodisk(gee_l8_mosaic.select(['B1','B2','B3','B4','B5','B6','B7']), 'e_l8', workdir, sampling, roi)
    #return(gee_l8_mosaic)


def extr_SIG0_LIA_ts_GEE(lon, lat,
                         bufferSize=20,
                         maskwinter=True,
                         lcmask=True,
                         globcover_mask=False,
                         trackflt=None,
                         masksnow=True,
                         varmask=False,
                         ssmcor=None,
                         dual_pol=True,
                         desc=False,
                         tempfilter=False):

    ee.Reset()
    ee.Initialize()

    # def setresample(image):
    #     image = image.resample()
    #     return (image)


    def computeLIA(image):

        srtm = ee.Image("USGS/SRTMGL1_003")
        srtm_slope = ee.Terrain.slope(srtm)
        srtm_aspect = ee.Terrain.aspect(srtm)

        inc = ee.Image(image).select('angle')#.resample()

        # azim_img = ee.Terrain.aspect(inc)
        # azim = azim_img.reduceRegion(ee.Reducer.mean(),
        #                              azim_img.geometry(), 1000).get('aspect')

        s = srtm_slope.multiply(ee.Image.constant(277).subtract(srtm_aspect).multiply(math.pi / 180).cos())
        lia = inc.subtract(ee.Image.constant(90).subtract(ee.Image.constant(90).subtract(s))).abs()

        return image.addBands(lia.select(['angle'], ['lia']).reproject(srtm.projection()))


    def miscMask(image):
        tmp = ee.Image(image)

        # srtm dem
        gee_srtm = ee.Image("USGS/SRTMGL1_003")
        gee_srtm_slope = ee.Terrain.slope(gee_srtm)

        # mask pixels
        vv = tmp.select('VV')
        if dual_pol == True:
            vh = tmp.select('VH')
            maskvh = vh.gte(-25).bitwiseAnd(vh.lt(0)) # was -25 and 0
        lia = tmp.select('lia')
        maskvv = vv.gte(-25).bitwiseAnd(vv.lt(0))
        masklia1 = lia.gt(20) # angle 10
        masklia2 = lia.lt(45) # angle 50
        masklia = masklia1.bitwiseAnd(masklia2)
        #maskslope = gee_srtm_slope.lt(20)

        if dual_pol == True:
            mask = maskvv.bitwiseAnd(maskvh)
        else:
            mask = maskvv
        mask = mask.bitwiseAnd(masklia)
        #mask = mask.bitwiseAnd(maskslope)
        tmp = tmp.updateMask(mask)

        return(tmp)


    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        #tmp = ee.Image(image).resample()
        tmp = ee.Image(image)

        # Conver to linear before averaging
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV').divide(10)))
        if dual_pol == True:
            tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VH').divide(10)))
            tmp = tmp.select(['constant', 'constant_1', 'angle', 'lia'], ['VV', 'VH', 'angle', 'lia'])
        else:
            tmp = tmp.select(['constant', 'angle', 'lia'], ['VV', 'angle', 'lia'])

        reduced_img_data = tmp.reduceRegion(ee.Reducer.median(), gee_roi, 10)
        totcount = ee.Image(1).reduceRegion(ee.Reducer.count(), gee_roi, 10)
        pcount = tmp.reduceRegion(ee.Reducer.count(), gee_roi, 10)
        return ee.Feature(None, {'result': reduced_img_data, 'count': pcount, 'totcount': totcount})


    def mask_lc(image):

        tmp = ee.Image(image)

        # load land cover info
        corine = ee.Image('users/felixgreifeneder/corine')

        # create lc mask
        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
            .bitwiseOr(corine.eq(valLClist[2])) \
            .bitwiseOr(corine.eq(valLClist[3])) \
            .bitwiseOr(corine.eq(valLClist[4])) \
            .bitwiseOr(corine.eq(valLClist[5])) \
            .bitwiseOr(corine.eq(valLClist[6])) \
            .bitwiseOr(corine.eq(valLClist[7])) \
            .bitwiseOr(corine.eq(valLClist[8])) \
            .bitwiseOr(corine.eq(valLClist[9])) \
            .bitwiseOr(corine.eq(valLClist[10])) \
            .bitwiseOr(corine.eq(valLClist[11]))

        tmp = tmp.updateMask(lcmask)

        return tmp


    def mask_lc_globcover(image):

        tmp = ee.Image(image)

        # load lc
        glbcvr = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3")

        valLClist = [11, 14, 20, 30, 120, 140, 150]

        lcmask = glbcvr.eq(valLClist[0]) \
                       .bitwiseOr(glbcvr.eq(valLClist[1])) \
                       .bitwiseOr(glbcvr.eq(valLClist[2])) \
                       .bitwiseOr(glbcvr.eq(valLClist[3])) \
                       .bitwiseOr(glbcvr.eq(valLClist[4])) \
                       .bitwiseOr(glbcvr.eq(valLClist[5])) \
                       .bitwiseOr(glbcvr.eq(valLClist[6]))

        tmp = tmp.updateMask(lcmask)

        return tmp


    def toln(image):

        tmp = ee.Image(image)

        # Convert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dual_pol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to ln
        out = vv.log()
        if dual_pol == True:
            out = out.addBands(vh.log())
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = out.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))


    def tolin(image):

        tmp = ee.Image(image)

        # Convert to linear
        vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Output
        out = vh.select(['constant'], ['VH'])

        return out.set('system:time_start', tmp.get('system:time_start'))


    def tolin_dual(image):

        tmp = ee.Image(image)
        if dual_pol == True:
            lin = ee.Image(10).pow(tmp.divide(10))#.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            lin = ee.Image(10).pow(tmp.divide(10))#.select(['constant'], ['VV'])

        return lin.set('system:time_start', tmp.get('system:time_start'))


    def applyvarmask(image):

        tmp = ee.Image(image)
        tmp = tmp.updateMask(varmask)

        return(tmp)


    def applysnowmask(image):

        tmp = ee.Image(image)
        sdiff = tmp.select('VH').subtract(snowref)
        wetsnowmap = sdiff.lte(-2.6).focal_mode(100, 'square', 'meters', 3)

        return (tmp.updateMask(wetsnowmap.eq(0)))


    def addASCAT(image):

        tmp = ee.Image(image)
        #img_date = ee.Date(tmp.get('system:time_start')).format('Y-M-d').getInfo()
        img_date = ee.Date(tmp.get('system:time_start'))
        ASCATtmp = ASCATcollection.filterDate(img_date.format('Y-M-d'))
        current_ssm = ee.ImageCollection(ASCATtmp).toList(10).get(0)
        #img_date_dt = dt.datetime.strptime(img_date, '%Y-%m-%d').date()

        # select best ASCAT
        #current_ssm = ssmcor[np.array([x.date() for x in ssmcor.index]) == img_date_dt].values[0]

        out_image = tmp.addBands(ee.Image(current_ssm))

        return(out_image)


    def s1_simplyfy_date(image):
        return(image.set('system:time_start', ee.Date(ee.Date(image.get('system:time_start')).format('Y-M-d'))))


    def applyCorrelationMask(image):
        mask = ssm_vv_cor.select('correlation').gt(0.1)
        return(image.updateMask(mask))


    def projectlia(image):
        tmp = ee.Image(image)
        trgtprj = tmp.select('VV').projection()
        tmp = tmp.addBands(tmp.select('angle').reproject(trgtprj), ['angle'], True)
        return(tmp)


    def todb(image):

        tmp = ee.Image(image)

        return ee.Image(10).multiply(tmp.log10()).set('system:time_start', tmp.get('system:time_start'))

    import timeit

    tic = timeit.default_timer()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')#.map(setresample)

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(gee_roi) \
        .filter(ee.Filter.eq('platform_number', 'A')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) #\ .filter(ee.Filter.eq('platform_number', 'A')) \
        #.filterDate('2016-01-01', opt_end='2017-06-01')
        #.filterDate('2014-01-01', opt_end='2015-12-31')
        #.filterDate('2015-01-1', opt_end='2015-12-31')
        #.filter(ee.Filter.dayOfYear(121,304)) # 1st of may to 31st of october

    # clip to roi
    # gee_s1_filtered = gee_s1_filtered.map(lambda x: x.clip(ee.Geometry.Point(lon, lat).buffer(1000)))

    if desc == False:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

    if dual_pol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if maskwinter == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121,304))

    if trackflt is not None:
        if isinstance(trackflt, list):
            gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.inList(ee.List(trackflt), 'relativeOrbitNumber_start'))
        else:
            gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', trackflt))

    # reproject angle
    # gee_s1_filtered = gee_s1_filtered.map(projectlia)

    if lcmask == True:
        gee_s1_filtered = gee_s1_filtered.map(mask_lc)
    if globcover_mask == True:
        gee_s1_filtered = gee_s1_filtered.map(mask_lc_globcover)

    gee_s1_filtered = gee_s1_filtered.map(computeLIA)

    # apply msicaleanous mask
    gee_s1_filtered = gee_s1_filtered.map(miscMask, True)

    if ssmcor is not None:
        ASCATlist = list()
        ssmcor = ssmcor.resample('D').mean().dropna()
        ssmcor = ssmcor.astype(np.float)
        for i in range(len(ssmcor)):
            ssm_img = ee.Image(ssmcor[i]).clip(gee_roi).float()
            ssm_img = ssm_img.set('system:time_start', ssmcor.index[i])
            ASCATlist.append(ssm_img)
        ASCATcollection = ee.ImageCollection(ASCATlist)

        # prepare the join
        s1_joined = gee_s1_filtered.map(s1_simplyfy_date)
        join_filter = ee.Filter.equals(leftField='system:time_start',rightField='system:time_start')
        simple_join = ee.Join.simple()
        s1_joined = simple_join.apply(s1_joined, ASCATcollection, join_filter)

        # create ssm ASCAT, image collection
        s1_plus_ASCAT = ee.ImageCollection(s1_joined.map(addASCAT, True))
        ssm_vv_cor = s1_plus_ASCAT.select(['VV', 'constant']).reduce(ee.Reducer.pearsonsCorrelation())
        gee_s1_filtered = gee_s1_filtered.map(applyCorrelationMask)

    # get the track numbers
    tmp = gee_s1_filtered.getInfo()
    track_series = np.array([x['properties']['relativeOrbitNumber_start'] for x in tmp['features']])
    available_tracks = np.unique(track_series)

    out_dict = {}
    for track_nr in available_tracks:

        #  filter for track
        if dual_pol == True:
            gee_s1_track_fltd = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr).select(['VV','VH','angle','lia'])
        else:
            gee_s1_track_fltd = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr).select(['VV','angle','lia'])

        if varmask == True:
            # compute temporal statistics
            gee_s1_ln = gee_s1_track_fltd.map(toln)
            # gee_s1_ln = gee_s1_ln.clip(roi)
            k2vv = ee.Image(gee_s1_ln.select('VV').reduce(ee.Reducer.stdDev()))
            if dual_pol == True:
                k2vh = ee.Image(gee_s1_ln.select('VH').reduce(ee.Reducer.stdDev()))
                varmask = k2vv.gt(0.4).And(k2vh.gt(0.4))
            else:
                varmask = k2vv.gt(0.4)
            gee_s1_track_fltd = gee_s1_track_fltd.map(applyvarmask)

        if tempfilter == True:
            # apply a temporal speckle filter
            radius = 3
            units = 'pixels'
            gee_s1_linear = gee_s1_filtered.map(tolin_dual)
            gee_s1_dspckld_vv = multitemporalDespeckle(gee_s1_linear.select('VV'), radius, units,
                                                       {'before': -12, 'after': 12, 'units': 'month'})
            gee_s1_dspckld_vv = gee_s1_dspckld_vv.map(todb).select(['constant'], ['VV'])
            if dual_pol == True:
                gee_s1_dspckld_vh = multitemporalDespeckle(gee_s1_linear.select('VH'), radius, units,
                                                           {'before': -12, 'after': 12, 'units': 'month'})
                gee_s1_dspckld_vh = gee_s1_dspckld_vh.map(todb).select(['constant'], ['VH'])
            if dual_pol == False:
                gee_s1_track_fltd = gee_s1_dspckld_vv.combine(gee_s1_track_fltd.select('angle')) \
                                                     .combine(gee_s1_track_fltd.select('lia'))
            else:
                gee_s1_track_fltd = gee_s1_dspckld_vv.combine(gee_s1_dspckld_vh) \
                                                     .combine(gee_s1_track_fltd.select('angle')) \
                                                     .combine(gee_s1_track_fltd.select('lia'))

        # apply wet snow mask
        if masksnow == True:
            gee_s1_lin = gee_s1_track_fltd.select('VH').map(tolin)
            snowref = ee.Image(10).multiply(gee_s1_lin.reduce(ee.Reducer.intervalMean(5,100)).log10())
            gee_s1_track_fltd = gee_s1_track_fltd.map(applysnowmask)

        gee_s1_mapped = gee_s1_track_fltd.map(createAvg)
        tmp = gee_s1_mapped.getInfo()
        # get vv
        vv_sig0 = 10*np.log10(np.array([x['properties']['result']['VV'] for x in tmp['features']], dtype=np.float))

        if dual_pol == True:
            # get vh
            vh_sig0 = 10*np.log10(np.array([x['properties']['result']['VH'] for x in tmp['features']], dtype=np.float))

        ge_dates = np.array([datetime.strptime(x['id'][17:32], '%Y%m%dT%H%M%S') for x in tmp['features']])

        # get lia
        lia = np.array([x['properties']['result']['lia'] for x in tmp['features']], dtype=np.float)

        # get val_count
        val_count = np.array([np.float(x['properties']['count']['VV']) / np.float(x['properties']['totcount']['constant']) for x in tmp['features']], dtype=np.float)

        if bufferSize <= 100:
            valid = np.where(val_count > 0.01)
        else:
            valid = np.where(val_count > 0.1)
        vv_sig0 = vv_sig0[valid]
        if dual_pol == True:
            vh_sig0 = vh_sig0[valid]
        lia = lia[valid]
        ge_dates = ge_dates[valid]

        if dual_pol == True:
            out_dict[str(int(track_nr))] =  (ge_dates, {'sig0': vv_sig0, 'sig02': vh_sig0, 'lia': lia})
        else:
            out_dict[str(int(track_nr))] = (ge_dates, {'sig0': vv_sig0, 'lia': lia})

    toc = timeit.default_timer()

    print (toc - tic)

    return (out_dict)


def extr_SIG0_LIA_ts_GEE_VV(lon, lat, bufferSize=20, maskwinter=True):

    ee.Reset()
    ee.Initialize()

    def setresample(image):
        image = image.resample()
        return (image)

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        tmp = ee.Image(image)

        # mask pixels
        lia = tmp.select('angle')
        masklia1 = lia.gt(10)
        masklia2 = lia.lt(50)
        masklia = masklia1.bitwiseAnd(masklia2)

        tmp = tmp.updateMask(masklia)

        # Conver to linear before averaging
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV').divide(10)))
        # tmp2 = ee.Image.cat([VVlin, VHlin, tmp.select('angle)')])
        tmp = tmp.select(['constant', 'angle'], ['VV', 'angle'])

        reduced_img_data = tmp.reduceRegion(ee.Reducer.median(), gee_roi, 10)
        return ee.Feature(None, {'result': reduced_img_data})

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD').map(setresample)

    # filter S1 collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(gee_roi) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        #.filterDate('2014-01-01', opt_end='2015-12-31')
        #.filter(ee.Filter.dayOfYear(121,304)) # 1st of may to 31st of october

    if maskwinter == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121,304))

    # apply lc mask
    #gee_s1_filtered.updateMask(lcmask)

    # .filterMetadata('relativeOrbitNumber_start', 'equals', track_nr) \
    # get the track numbers
    tmp = gee_s1_filtered.getInfo()
    track_series = np.array([x['properties']['relativeOrbitNumber_start'] for x in tmp['features']])
    available_tracks = np.unique(track_series)


    out_dict = {}
    for track_nr in available_tracks:

        #  filter for track
        gee_s1_track_fltd = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)

        gee_s1_mapped = gee_s1_track_fltd.map(createAvg)
        tmp = gee_s1_mapped.getInfo()
        # get vv
        vv_sig0 = 10*np.log10(np.array([x['properties']['result']['VV'] for x in tmp['features']], dtype=np.float))
        ge_dates = np.array([datetime.strptime(x['id'][17:32], '%Y%m%dT%H%M%S') for x in tmp['features']])

        # get lia
        lia = np.array([x['properties']['result']['angle'] for x in tmp['features']])

        out_dict[str(int(track_nr))] =  (ge_dates, {'sig0': vv_sig0, 'lia': lia})

    return (out_dict)


def extr_GEE_array(minlon, minlat, maxlon, maxlat, year, month, day, workdir, tempfilter=True, applylcmask=True, sampling=20,
                   dualpol=True, trackflt=None, maskwinter=True):
    def maskterrain(image):
        # srtm dem
        gee_srtm = ee.Image("USGS/SRTMGL1_003")
        gee_srtm_slope = ee.Terrain.slope(gee_srtm)

        # load land cover info
        corine = ee.Image('users/felixgreifeneder/corine')

        # create lc mask
        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
            .bitwiseOr(corine.eq(valLClist[2])) \
            .bitwiseOr(corine.eq(valLClist[3])) \
            .bitwiseOr(corine.eq(valLClist[4])) \
            .bitwiseOr(corine.eq(valLClist[5])) \
            .bitwiseOr(corine.eq(valLClist[6])) \
            .bitwiseOr(corine.eq(valLClist[7])) \
            .bitwiseOr(corine.eq(valLClist[8])) \
            .bitwiseOr(corine.eq(valLClist[9])) \
            .bitwiseOr(corine.eq(valLClist[10])) \
            .bitwiseOr(corine.eq(valLClist[11]))

        tmp = ee.Image(image)
        mask = gee_srtm_slope.lt(20)
        mask2 = tmp.lt(0).bitwiseAnd(tmp.gt(-25))
        # mask2 = tmp.gte(-25)
        if applylcmask == False:
            mask = mask.bitwiseAnd(mask2)
        else:
            mask = mask.bitwiseAnd(mask2).bitwiseAnd(lcmask)
        tmp = tmp.updateMask(mask)
        return (tmp)

    def setresample(image):
        image = image.resample()
        return (image)

    def toln(image):

        tmp = ee.Image(image)

        # Convert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to ln
        out = vv.log()
        if dualpol == True:
            out = out.addBands(vh.log())
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = out.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def tolin(image):

        tmp = ee.Image(image)

        # Covert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to
        if dualpol == True:
            out = vv.addBands(vh)
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = vv.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def todb(image):

        tmp = ee.Image(image)

        return ee.Image(10).multiply(tmp.log10()).set('system:time_start', tmp.get('system:time_start'))

    ee.Reset()
    ee.Initialize()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

    # for lc_id in range(1, len(valLClist)):
    #     tmpmask = corine.eq(valLClist[lc_id])
    #     lcmask = lcmask.bitwiseAnd(tmpmask)

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])

    # ASCENDING acquisitions
    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

    if dualpol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if trackflt is not None:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', trackflt))

    if maskwinter == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121, 304))

    #gee_s1_filtered = gee_s1_filtered.map(setresample)
    gee_s1_filtered = gee_s1_filtered.map(maskterrain)

    # create a list of availalbel dates
    tmp = gee_s1_filtered.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter imagecollection for respective date
    gee_s1_list = gee_s1_filtered.toList(2000)
    doi_indices = np.where(dates == date_selected)[0]
    gee_s1_drange = ee.ImageCollection(gee_s1_list.slice(doi_indices[0], doi_indices[-1] + 1))
    s1_sig0 = gee_s1_drange.mosaic()
    s1_sig0 = ee.Image(s1_sig0.copyProperties(gee_s1_drange.first()))

    # fetch image from image collection
    s1_lia = s1_sig0.select('angle').clip(roi)
    # get the track number
    s1_sig0_info = s1_sig0.getInfo()
    track_nr = s1_sig0_info['properties']['relativeOrbitNumber_start']

    # despeckle
    if tempfilter == True:
        radius = 7
        units = 'pixels'
        gee_s1_linear = gee_s1_filtered.map(tolin)
        gee_s1_dspckld_vv = multitemporalDespeckle(gee_s1_linear.select('VV'), radius, units,
                                                   {'before': -12, 'after': 12, 'units': 'month'})
        gee_s1_dspckld_vv = gee_s1_dspckld_vv.map(todb)
        gee_s1_list_vv = gee_s1_dspckld_vv.toList(2000)
        gee_s1_fltrd_vv = ee.ImageCollection(gee_s1_list_vv.slice(doi_indices[0], doi_indices[-1] + 1))
        s1_sig0_vv = gee_s1_fltrd_vv.mosaic()
        # s1_sig0_vv = ee.Image(gee_s1_list_vv.get(doi_index))

        if dualpol == True:
            gee_s1_dspckld_vh = multitemporalDespeckle(gee_s1_linear.select('VH'), radius, units,
                                                       {'before': -12, 'after': 12, 'units': 'month'})
            gee_s1_dspckld_vh = gee_s1_dspckld_vh.map(todb)
            gee_s1_list_vh = gee_s1_dspckld_vh.toList(2000)
            gee_s1_fltrd_vh = ee.ImageCollection(gee_s1_list_vh.slice(doi_indices[0], doi_indices[-1] + 1))
            s1_sig0_vh = gee_s1_fltrd_vh.mosaic()

        if dualpol == True:
            s1_sig0 = s1_sig0_vv.addBands(s1_sig0_vh).select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            s1_sig0 = s1_sig0_vv.select(['constant'], ['VV'])
            # s1_sig0_dsc = s1_sig0_vv_dsc.select(['constant'], ['VV'])

    # extract information
    # if applylcmask == True:
    #     s1_sig0 = s1_sig0.updateMask(lcmask)
    # s1_sig0 = s1_sig0.clip(roi)
    s1_sig0_vv = s1_sig0.select('VV')
    s1_sig0_vv = s1_sig0_vv.clip(roi)
    if dualpol == True:
        s1_sig0_vh = s1_sig0.select('VH')
        s1_sig0_vh = s1_sig0_vh.clip(roi)

    # compute temporal statistics
    gee_s1_filtered = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)
    gee_s1_ln = gee_s1_filtered.map(toln)
    # gee_s1_ln = gee_s1_ln.clip(roi)
    k1vv = ee.Image(gee_s1_ln.select('VV').mean()).clip(roi)
    k2vv = ee.Image(gee_s1_ln.select('VV').reduce(ee.Reducer.stdDev())).clip(roi)

    if dualpol == True:
        k1vh = ee.Image(gee_s1_ln.select('VH').mean()).clip(roi)
        k2vh = ee.Image(gee_s1_ln.select('VH').reduce(ee.Reducer.stdDev())).clip(roi)


    # export
    #s1_sig0_vv = s1_sig0_vv.reproject(s1_lia.projection(), scale=sampling)
    # s1_sig0diff = s1_sig0diff.reproject(s1_lia.projection(), scale=sampling)
    # lia_exp = ee.batch.Export.image.toDrive(image=s1_lia, description='lia',
    #                                         fileNamePrefix='s1lia'+str(date_selected), scale=sampling,
    #                                         region=roi.toGeoJSON()['coordinates'],
    #                                         maxPixels=1000000000000)
    # sig0diff_exp = ee.batch.Export.image.toDrive(image=s1_sig0diff, description='diff',
    #                                              fileNamePrefix='s1diff'+str(date_selected), scale=sampling)
    GEtodisk(s1_sig0_vv, 'a_sig0vv'+str(date_selected), workdir, sampling, roi)
    GEtodisk(k1vv, 'c_k1vv' + str(int(track_nr)), workdir, sampling, roi)
    GEtodisk(k2vv, 'e_k2vv' + str(int(track_nr)), workdir, sampling, roi)

    if dualpol == True:
        GEtodisk(s1_sig0_vh, 'b_sig0vh' + str(date_selected), workdir, sampling, roi)
        GEtodisk(k1vh, 'd_k1vh' + str(int(track_nr)), workdir, sampling, roi)
        GEtodisk(k2vh, 'f_k2vh' + str(int(track_nr)), workdir, sampling, roi)


def extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat, year, month, day, tempfilter=True, applylcmask=True, mask_globcover=False,
                        sampling=20, dualpol=True, trackflt=None, maskwinter=True, masksnow=True,
                        explicit_t_mask=None, ascending=True, maskLIA=True):

    def computeLIA(image):

        srtm = ee.Image("USGS/SRTMGL1_003")
        srtm_slope = ee.Terrain.slope(srtm)
        srtm_aspect = ee.Terrain.aspect(srtm)

        inc = ee.Image(image).select('angle')#.resample()

        # azim_img = ee.Terrain.aspect(inc)
        # azim = azim_img.reduceRegion(ee.Reducer.mean(),
        #                              azim_img.geometry(), 1000).get('aspect')

        s = srtm_slope.multiply(ee.Image.constant(277).subtract(srtm_aspect).multiply(math.pi / 180).cos())
        lia = inc.subtract(ee.Image.constant(90).subtract(ee.Image.constant(90).subtract(s))).abs()

        return image.addBands(lia.select(['angle'], ['lia']).reproject(srtm.projection()))


    def maskterrain(image):
        tmp = ee.Image(image)
        # srtm dem
        if maskLIA == False:
            gee_srtm = ee.Image("USGS/SRTMGL1_003")
            gee_srtm_slope = ee.Terrain.slope(gee_srtm)
            mask = gee_srtm_slope.lt(20)
        else:
            lia = tmp.select('lia')
            mask = lia.gt(20).bitwiseAnd(lia.lt(45))
        mask2 = tmp.lt(0).bitwiseAnd(tmp.gt(-25))
        #mask2 = tmp.gte(-25)
        mask = mask.bitwiseAnd(mask2)
        #mask = mask2
        tmp = tmp.updateMask(mask)
        return(tmp)


    def masklc(image):
        # load land cover info
        corine = ee.Image('users/felixgreifeneder/corine')

        # create lc mask
        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
            .bitwiseOr(corine.eq(valLClist[2])) \
            .bitwiseOr(corine.eq(valLClist[3])) \
            .bitwiseOr(corine.eq(valLClist[4])) \
            .bitwiseOr(corine.eq(valLClist[5])) \
            .bitwiseOr(corine.eq(valLClist[6])) \
            .bitwiseOr(corine.eq(valLClist[7])) \
            .bitwiseOr(corine.eq(valLClist[8])) \
            .bitwiseOr(corine.eq(valLClist[9])) \
            .bitwiseOr(corine.eq(valLClist[10])) \
            .bitwiseOr(corine.eq(valLClist[11]))

        tmp = ee.Image(image)

        tmp = tmp.updateMask(lcmask)
        return(tmp)


    def mask_lc_globcover(image):

        tmp = ee.Image(image)

        # load lc
        glbcvr = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3").select('landcover')

        valLClist = [11, 14, 20, 30, 120, 140, 150]

        lcmask = glbcvr.eq(valLClist[0]) \
            .bitwiseOr(glbcvr.eq(valLClist[1])) \
            .bitwiseOr(glbcvr.eq(valLClist[2])) \
            .bitwiseOr(glbcvr.eq(valLClist[3])) \
            .bitwiseOr(glbcvr.eq(valLClist[4])) \
            .bitwiseOr(glbcvr.eq(valLClist[5])) \
            .bitwiseOr(glbcvr.eq(valLClist[6]))

        tmp = tmp.updateMask(lcmask)

        return tmp


    def setresample(image):
        image = image.resample()
        return(image)


    def toln(image):

        tmp = ee.Image(image)

        # Convert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to ln
        out = vv.log()
        if dualpol == True:
            out = out.addBands(vh.log())
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = out.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))


    def tolin(image):

        tmp = ee.Image(image)

        # Covert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to
        if dualpol == True:
            out = vv.addBands(vh)
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = vv.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))


    def todb(image):

        tmp = ee.Image(image)

        return ee.Image(10).multiply(tmp.log10()).set('system:time_start', tmp.get('system:time_start'))


    def applysnowmask(image):

        tmp = ee.Image(image)
        sdiff = tmp.select('VH').subtract(snowref)
        wetsnowmap = sdiff.lte(-2.6).focal_mode(100, 'square', 'meters', 3)

        return(tmp.updateMask(wetsnowmap.eq(0)))


    def projectlia(image):
        tmp = ee.Image(image)
        trgtprj = tmp.select('VV').projection()
        tmp = tmp.addBands(tmp.select('angle').reproject(trgtprj), ['angle'], True)
        return (tmp)


    def apply_explicit_t_mask(image):

        t_mask = ee.Image('users/felixgreifeneder/' + explicit_t_mask)
        mask = t_mask.eq(0)
        return(image.updateMask(mask))

    ee.Reset()
    ee.Initialize()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])
    #roi = get_south_tyrol_roi()

    # ASCENDING acquisitions
    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('platform_number', 'A')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

    if ascending == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \

    if dualpol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if trackflt is not None:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', trackflt))

    if maskwinter == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121, 304))

    #gee_s1_filtered = gee_s1_filtered.map(projectlia)
    #s1_lia = gee_s1_filtered.map(computeLIA).select('angle', 'lia')
    # add LIA
    if maskLIA == True:
        gee_s1_filtered = gee_s1_filtered.map(computeLIA)
    s1_lia = gee_s1_filtered.select('angle')

    if applylcmask == True:
        gee_s1_filtered = gee_s1_filtered.map(masklc)
    if mask_globcover == True:
        gee_s1_filtered = gee_s1_filtered.map(mask_lc_globcover)

    gee_s1_filtered = gee_s1_filtered.map(setresample)

    if explicit_t_mask == None:
        gee_s1_filtered = gee_s1_filtered.map(maskterrain)
    else:
        gee_s1_filtered = gee_s1_filtered.map(apply_explicit_t_mask)


    # construct roi
    # roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
    #                            [maxlon, maxlat], [maxlon, minlat],
    #                            [minlon, minlat]], ee.Image(gee_s1_filtered.first()).select('VV').projection(), evenOdd=False)

    # apply wet snow mask
    if masksnow == True:
        gee_s1_linear_vh = gee_s1_filtered.map(tolin).select('VH')
        snowref = ee.Image(10).multiply(gee_s1_linear_vh.reduce(ee.Reducer.intervalMean(5, 100)).log10())
        gee_s1_filtered = gee_s1_filtered.map(applysnowmask)

    # create a list of availalbel dates
    tmp = gee_s1_filtered.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter imagecollection for respective date
    gee_s1_drange = gee_s1_filtered.filterDate(date_selected.strftime('%Y-%m-%d'), (date_selected + dt.timedelta(days=1)).strftime('%Y-%m-%d'))
    s1_lia_drange = s1_lia.filterDate(date_selected.strftime('%Y-%m-%d'), (date_selected + dt.timedelta(days=1)).strftime('%Y-%m-%d'))
    if gee_s1_drange.size().getInfo() > 1:
        s1_lia = s1_lia_drange.mosaic()
        s1_sig0 = gee_s1_drange.mosaic()
        s1_lia = ee.Image(s1_lia.copyProperties(s1_lia_drange.first()))
        s1_sig0 = ee.Image(s1_sig0.copyProperties(gee_s1_drange.first()))
    else:
        s1_sig0 = ee.Image(gee_s1_drange.first())
        s1_lia = ee.Image(s1_lia_drange.first())

    # fetch image from image collection
    targetprj = s1_sig0.select('VV').projection()
    # s1_lia = s1_sig0.select('angle').reproject(targetprj, 100).clip(roi)
    # s1_lia = s1_sig0.select('angle')
    # get the track number
    s1_sig0_info = s1_sig0.getInfo()
    track_nr = s1_sig0_info['properties']['relativeOrbitNumber_start']

    # only uses images of the same track
    gee_s1_filtered = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)

    # despeckle
    if tempfilter == True:
        radius = 7
        units = 'pixels'
        gee_s1_linear = gee_s1_filtered.map(tolin)
        gee_s1_dspckld_vv = multitemporalDespeckle(gee_s1_linear.select('VV'), radius, units,
                                                   {'before': -12, 'after': 12, 'units': 'month'})
        gee_s1_dspckld_vv = gee_s1_dspckld_vv.map(todb)
        #gee_s1_list_vv = gee_s1_dspckld_vv.toList(2000)
        #gee_s1_fltrd_vv = ee.ImageCollection(gee_s1_list_vv.slice(doi_indices[0], doi_indices[-1]+1))
        gee_s1_fltrd_vv = gee_s1_dspckld_vv.filterDate(date_selected.strftime('%Y-%m-%d'), (date_selected + dt.timedelta(days=1)).strftime('%Y-%m-%d'))
        s1_sig0_vv = gee_s1_fltrd_vv.mosaic()
        #s1_sig0_vv = ee.Image(gee_s1_list_vv.get(doi_index))

        if dualpol == True:
            gee_s1_dspckld_vh = multitemporalDespeckle(gee_s1_linear.select('VH'), radius, units,
                                                       {'before': -12, 'after': 12, 'units': 'month'})
            gee_s1_dspckld_vh = gee_s1_dspckld_vh.map(todb)
            #gee_s1_list_vh = gee_s1_dspckld_vh.toList(2000)
            #gee_s1_fltrd_vh = ee.ImageCollection(gee_s1_list_vh.slice(doi_indices[0], doi_indices[-1] + 1))
            gee_s1_fltrd_vh = gee_s1_dspckld_vh.filterDate(date_selected.strftime('%Y-%m-%d'), (date_selected + dt.timedelta(days=1)).strftime('%Y-%m-%d'))
            s1_sig0_vh = gee_s1_fltrd_vh.mosaic()

        if dualpol == True:
            s1_sig0 = s1_sig0_vv.addBands(s1_sig0_vh).select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            s1_sig0 = s1_sig0_vv.select(['constant'], ['VV'])
            # s1_sig0_dsc = s1_sig0_vv_dsc.select(['constant'], ['VV'])

    # apply the LIA mask
    #LIA = computeLIA(s1_lia, targetprj)
    #LIAmask = LIA.gt(20).bitwiseAnd(LIA.lt(45))
    #s1_lia = s1_lia.updateMask(LIAmask)
    #s1_sig0 = s1_sig0.updateMask(LIAmask)

    # extract information
    s1_sig0_vv = s1_sig0.select('VV')
    s1_sig0_vv = s1_sig0_vv.clip(roi)
    if dualpol == True:
        s1_sig0_vh = s1_sig0.select('VH')
        s1_sig0_vh = s1_sig0_vh.clip(roi)
    # s1_lia = s1_lia.reproject(s1_sig0_vv.projection()).clip(roi)
    # compute temporal statistics
    #gee_s1_filtered = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)

    gee_s1_ln = gee_s1_filtered.map(toln)
    gee_s1_lin = gee_s1_filtered.map(tolin)
    # gee_s1_ln = gee_s1_ln.clip(roi)
    k1vv = ee.Image(gee_s1_ln.select('VV').mean()).clip(roi)
    k2vv = ee.Image(gee_s1_ln.select('VV').reduce(ee.Reducer.stdDev())).clip(roi)
    mean_vv = ee.Image(gee_s1_lin.select('VV').mean()).clip(roi)
    std_vv = ee.Image(gee_s1_lin.select('VV').reduce(ee.Reducer.stdDev())).clip(roi)

    if dualpol == True:
        k1vh = ee.Image(gee_s1_ln.select('VH').mean()).clip(roi)
        k2vh = ee.Image(gee_s1_ln.select('VH').reduce(ee.Reducer.stdDev())).clip(roi)
        mean_vh = ee.Image(gee_s1_lin.select('VH').mean()).clip(roi)
        std_vh = ee.Image(gee_s1_lin.select('VH').reduce(ee.Reducer.stdDev())).clip(roi)

    #mask insensitive pixels
    # if dualpol == True:
    #     smask = k2vv.gt(0.4).And(k2vh.gt(0.4))
    # else:
    #     smask = k2vv.gt(0.4)
    #
    # s1_sig0_vv = s1_sig0_vv.updateMask(smask)
    # s1_lia = s1_lia.updateMask(smask)
    # k1vv = k1vv.updateMask(smask)
    # k2vv = k2vv.updateMask(smask)
    #
    # if dualpol == True:
    #     s1_sig0_vh = s1_sig0_vh.updateMask(smask)
    #     k1vh = k1vh.updateMask(smask)
    #     k2vh = k2vh.updateMask(smask)

    # export
    if dualpol == False:
        #s1_sig0_vv = s1_sig0_vv.reproject(s1_lia.projection())
        return(s1_sig0_vv, s1_lia, k1vv, k2vv, roi, date_selected)
    else:
        return(s1_lia,#.focal_median(sampling, 'square', 'meters'),
               s1_sig0_vv,#.focal_median(sampling, 'square', 'meters'),
               s1_sig0_vh,#.focal_median(sampling, 'square', 'meters'),
               k1vv,#.focal_median(sampling, 'square', 'meters'),
               k1vh,#.focal_median(sampling, 'square', 'meters'),
               k2vv,#.focal_median(sampling, 'square', 'meters'),
               k2vh,#.focal_median(sampling, 'square', 'meters'),
               roi,
               date_selected,
               mean_vv,#.focal_median(sampling, 'square', 'meters'),
               mean_vh,
               std_vv,
               std_vh)#.focal_median(sampling, 'square', 'meters'))


def get_s1_dates(minlon, minlat, maxlon, maxlat, tracknr=None, dualpol=True):

    ee.Initialize()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])
    # roi = get_south_tyrol_roi()

    # ASCENDING acquisitions
    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('platform_number', 'A')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) #\
        # .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

    if dualpol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if tracknr is not None:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', tracknr))

    # create a list of availalbel dates
    tmp = gee_s1_filtered.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

    return(dates)


# extract time series of SIG0 and LIA from SGRT database
def extr_SIG0_LIA_ts(dir_root, product_id, soft_id, product_name, src_res, lon, lat, xdim, ydim,
                     pol_name=None, grid=None, subgrid='EU', hour=None, sat_pass=None, monthmask=None):
    #initialise grid
    alpGrid = Equi7.Equi7Grid(src_res)

    #identify tile
    if grid is None:
        Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
    elif grid == 'Equi7':
        Equi7XY = [subgrid, lon, lat]
    TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1],Equi7XY[2]])
    TileExtent = Equi7.Equi7Tile(TileName).extent
    #load tile
    TOI = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id, soft_id=soft_id, product_name=product_name, ftile=TileName, src_res=src_res)

    # extract data
    x = int((Equi7XY[1] - TileExtent[0]) / src_res)
    y = int((TileExtent[3] - Equi7XY[2]) / src_res)

    # check if month mask is set
    if monthmask is None:
        monthmask = [1,2,3,4,5,6, 7, 8, 9,10,11,12]

    #extract data
    if pol_name is None:
        SIG0 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim)
        LIA = TOI.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1],:,:]
        SIG0 = (days, data)
        udates = np.unique(LIA[0], return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1],:,:]
        LIA = (days, data)

    elif len(pol_name) == 1:
        SIG0 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name.upper())
        LIA = TOI.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1],:,:]
        SIG0 = (days, data)
        udates = np.unique(LIA[0], return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1],:,:]
        LIA = (days, data)

    elif len(pol_name) == 2:
        SIG0 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim,
                           pol_name=pol_name[0].upper(), sat_pass=sat_pass)
        SIG02 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim,
                            pol_name=pol_name[1].upper(), sat_pass=sat_pass)
        LIA = TOI.read_ts("LIA__", x, y, xsize=xdim, ysize=ydim, sat_pass=sat_pass)

        # this is temporary: filter scenes based on time, TODO: change or remove
        if hour is not None:
            morning = np.where(np.array([SIG0[0][i].hour for i in range(len(SIG0[0]))]) == hour)[0]
            SIG0 = (np.array(SIG0[0])[morning], SIG0[1][morning])
            morning = np.where(np.array([SIG02[0][i].hour for i in range(len(SIG02[0]))]) == hour)[0]
            SIG02 = (np.array(SIG02[0])[morning], SIG02[1][morning])
            morning = np.where(np.array([LIA[0][i].hour for i in range(len(LIA[0]))]) == hour)[0]
            LIA = (np.array(LIA[0])[morning], LIA[1][morning])

        # filter months
        # TODO make an option
        summer = np.where(np.in1d(np.array([SIG0[0][i].month for i in range(len(SIG0[0]))]), monthmask))[0]
        SIG0 = (np.array(SIG0[0])[summer], SIG0[1][summer])
        summer = np.where(np.in1d(np.array([SIG02[0][i].month for i in range(len(SIG02[0]))]), monthmask))[0]
        SIG02 = (np.array(SIG02[0])[summer], SIG02[1][summer])
        summer = np.where(np.in1d(np.array([LIA[0][i].month for i in range(len(LIA[0]))]), monthmask))[0]
        LIA = (np.array(LIA[0])[summer], LIA[1][summer])

        # check if date dublicates exist
        # TODO average if two measurements in one day
        datedate = [SIG0[0][i].date() for i in range(len(SIG0[0]))]
        udates = np.unique(datedate, return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1],:,:]
        SIG0 = (days, data)
        datedate = [SIG02[0][i].date() for i in range(len(SIG02[0]))]
        udates = np.unique(datedate, return_index=True)
        days = np.array(SIG02[0])[udates[1]]
        data = np.array(SIG02[1])[udates[1],:,:]
        SIG02 = (days, data)
        datedate = [LIA[0][i].date() for i in range(len(LIA[0]))]
        udates = np.unique(datedate, return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1],:,:]
        LIA = (days, data)
    else:
        return None

    # format datelists and check if all dates are available for both SIG0 and LIA.
    # datelistSIG = []
    # datelistLIA = []
    # for i in range(len(SIG0[0])): datelistSIG.append(int(SIG0[0][i].toordinal()))
    # for i in range(len(LIA[0])): datelistLIA.append(int(LIA[0][i].toordinal()))
    datelistSIG = SIG0[0]
    if len(pol_name) == 2:
        datelistSIG2 = SIG02[0]
    else:
        datelistSIG2 = SIG02[0]

    datelistLIA = LIA[0]

    datelistFINAL = [x for x in datelistSIG if (x in datelistLIA) and (x in datelistSIG2)]

    SIG0out = [SIG0[1][x,:,:] for x in range(len(SIG0[0])) if datelistSIG[x] in datelistFINAL]
    LIAout = [LIA[1][x,:,:] for x in range(len(LIA[0])) if datelistLIA[x] in datelistFINAL]
    if len(pol_name) == 1:
        outtuple = (np.asarray(datelistFINAL), {'sig0': np.asarray(SIG0out), 'lia': np.asarray(LIAout)})
    elif len(pol_name) == 2:
        SIG0out2 = [SIG02[1][x,:,:] for x in range(len(SIG02[0])) if datelistSIG2[x] in datelistFINAL]
        outtuple = (np.asarray(datelistFINAL), {'sig0': np.asarray(SIG0out), 'sig02': np.asarray(SIG0out2), 'lia': np.asarray(LIAout)})

    #TOI = None

    return outtuple


def read_NORM_SIG0(dir_root, product_id, soft_id, product_name, src_res, lon, lat, xdim, ydim, pol_name=None, grid=None):
    # initialise grid
    alpGrid = Equi7.Equi7Grid(src_res)

    # identify tile
    if grid is None:
        Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
    elif grid == 'Equi7':
        Equi7XY = ['EU', lon, lat]
    TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1], Equi7XY[2]])
    TileExtent = Equi7.Equi7Tile(TileName).extent
    # load tile
    TOI = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id, soft_id=soft_id, product_name=product_name,
                            ftile=TileName, src_res=src_res)
    TOI_LIA = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id, soft_id='A0111', product_name='resampled',
                                ftile=TileName, src_res=src_res) # TODO allow to specify different versions

    # extract data
    x = int((Equi7XY[1] - TileExtent[0]) / src_res)
    y = int((TileExtent[3] - Equi7XY[2]) / src_res)


    # extract data
    if pol_name is None:
        SIG0 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim)
        LIA = TOI_LIA.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1], :, :]
        SIG0 = (days, data)

    elif len(pol_name) == 1:
        SIG0 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name.upper())
        LIA = TOI_LIA.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1], :, :]
        SIG0 = (days, data)

    elif len(pol_name) == 2:
        SIG0 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name[0].upper(), sat_pass='A')
        SIG02 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name[1].upper(), sat_pass='A')
        LIA = TOI_LIA.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1], :, :]
        SIG0 = (days, data)
        udates = np.unique(SIG02[0], return_index=True)
        days = np.array(SIG02[0])[udates[1]]
        data = np.array(SIG02[1])[udates[1], :, :]
        SIG02 = (days, data)
        udates = np.unique(LIA[0], return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1], :, :]
        LIA = (days, data)

    else:
        return None

    # format datelist and, in case of dual-pol, check if dates are available for both
    # polarisations
    datelistSIG = SIG0[0]
    if len(pol_name) == 2:
        datelistSIG2 = SIG02[0]
    else:
        datelistSIG2 = SIG0[0]

    datelistLIA = LIA[0]

    datelistFINAL = [x for x in datelistSIG if (x in datelistLIA) and (x in datelistSIG2)]

    SIG0out = [SIG0[1][x, :, :] for x in range(len(SIG0[0])) if datelistSIG[x] in datelistFINAL]
    LIAout = [LIA[1][x, :, :] for x in range(len(LIA[0])) if datelistLIA[x] in datelistFINAL]
    if len(pol_name) == 1:
        outtuple = (np.asarray(datelistFINAL), {'sig0': np.asarray(SIG0out), 'lia': np.asarray(LIAout)})
    elif len(pol_name) == 2:
        SIG0out2 = [SIG02[1][x, :, :] for x in range(len(SIG02[0])) if datelistSIG2[x] in datelistFINAL]
        outtuple = (np.asarray(datelistFINAL),
                    {'sig0': np.asarray(SIG0out), 'sig02': np.asarray(SIG0out2), 'lia': np.asarray(LIAout)})

    # TOI = None

    return outtuple

#extract value of any given raster at band (band) at lat, lon
def extr_raster_pixel_values(filename, bandnr, lat, lon, dtype):
    #initialisiation
    dataset = gdal.Open(filename, GA_ReadOnly)
    geotransform = dataset.GetGeoTransform()

    originX = geotransform[0]
    originY = geotransform[3]
    pixelSize = geotransform[1]

    #translate lat lon to image coordinates
    imX = int((lon - originX) / pixelSize)
    imY = int((originY - lat) / pixelSize)

    #retrieve data
    band = dataset.GetRasterBand(bandnr)
    if dtype == 1:
        bandArr = band.ReadAsArray().astype(np.byte)
    elif dtype == 2:
        bandArr = band.ReadAsArray().astype(np.int)
    elif dtype == 3:
        bandArr = band.ReadAsArray().astype(np.float32)
    elif dtype ==4:
        bandArr = band.ReadAsArray().astype(np.float64)
    else:
        print('unknown datatype')
        return(-9999)

    band = None
    dataset = None

    return(bandArr[imY, imX])


def get_south_tyrol_roi():

    roi = ee.Geometry.Polygon([[10.5029296875,46.49082901981415],
                                [10.607299804,46.468132992155],
                                [10.610046386,46.426499019253],
                                [11.011047363,46.417032314661],
                                [11.315917968,46.375359286114],
                                [11.571350097,46.424605809835],
                                [11.818542480,46.504063997400],
                                [11.994323730,46.530524288784],
                                [12.060241699,46.672056467344],
                                [12.205810546,46.606054127132],
                                [12.376098632,46.630578680594],
                                [12.483215332,46.677710064644],
                                [12.389831542,46.726683132784],
                                [12.359619140,46.764324496011],
                                [12.277221679,46.788777287932],
                                [12.301940917,46.835770679359],
                                [12.268981933,46.882723010671],
                                [12.164611816,46.899615799267],
                                [12.145385742,47.019588864382],
                                [12.238769531,47.066380283213],
                                [12.117919921,47.083215147741],
                                [11.758117675,46.961510504873],
                                [11.629028320,47.015843777908],
                                [11.486206054,47.012098428760],
                                [11.420288085,46.959635958619],
                                [11.329650878,46.987747256465],
                                [11.107177734,46.927758623434],
                                [11.011047363,46.773730730079],
                                [10.821533203,46.773730730079],
                                [10.747375488,46.824496010262],
                                [10.656738281,46.877090898744],
                                [10.483703613,46.862069043222],
                                [10.450744628,46.754916619281],
                                [10.384826660,46.687131412444],
                                [10.395812988,46.640008243515],
                                [10.442504882,46.638122462379],
                                [10.486450195,46.609827858351],
                                [10.442504882,46.524855311033]])

    return roi


def GEtodisk(geds, name, dir, sampling, roi, timeout=True):

    file_exp = ee.batch.Export.image.toDrive(image=geds, description='fileexp' + name,
                                             fileNamePrefix=name, scale=sampling, region=roi.getInfo()['coordinates'],
                                             maxPixels=1000000000000)

    file_exp.start()

    start = time.time()
    success = 1

    while (file_exp.active() == True):
        time.sleep(2)
        if timeout == True and (time.time()-start) > 4800:
            success = 0
            break
    else:
        print('Export completed')

    if success == 1:
        # initialise Google Drive
        drive_handler = gdrive()
        print('Downloading files ...')
        print(name)
        drive_handler.download_file(name + '.tif',
                                    dir + name + '.tif')
        drive_handler.delete_file(name + '.tif')
    else:
        file_exp.cancel()