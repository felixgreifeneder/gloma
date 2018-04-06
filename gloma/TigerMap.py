# routines for the mapping of soil moisture based on GEE

__author__ = 'usergre'

import sys

sys.path.extend(['/home/usergre/winpycharm/sgrt_run',
                 '/home/usergre/winpycharm/sgrt',
                 '/home/usergre/winpycharm/Python_SGRT_devel',
                 '/home/usergre/winpycharm/ascat'])

from sgrt_devels.extr_TS import extr_GEE_array
from sgrt_devels.extr_TS import GEtodisk
from sgrt_devels.extr_TS import extr_GEE_array_reGE
from sgrt_devels.extr_TS import extr_MODIS_MOD13Q1_array_reGE
from sgrt_devels.extr_TS import extr_L8_array
from sgrt_devels.extr_TS import get_s1_dates
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE
import pickle
import numpy as np
from SMC.TIGERio import geotiff
import os, subprocess
import ee
import pandas as pd
import matplotlib.pyplot as plt
from sgrt_devels.extr_TS import extr_GLDAS_ts_GEE
import pytesmo.time_series.anomaly as anomaly
from pytesmo.time_series.plotting import plot_clim_anom
from sklearn.linear_model import LinearRegression
from pytesmo.temporal_matching import df_match


def estimateSMC(modelpath,
                sig0vvpath,
                sig0vhpath,
                k1vvpath,
                k1vhpath,
                k2vvpath,
                k2vhpath,
                outpath,
                workdir,
                outname):

    # load SVR model
    MLmodel = pickle.load(open(modelpath, 'rb'))
    MLmodel = {'SVRmodel': MLmodel[0], 'scaler': MLmodel[1]}

    # load images
    #sig0 = geotiff(sig0path)
    #k1 = geotiff(k1path)
    #k2 = geotiff(k2path)
    #lia = geotiff(liapath)

    # create image stack using gdal_merge
    #im_stack = np.stack((sig0.data, lia.data, k1.data, k2.data), axis=2)
    sys.path.append('/usr/local/bin')
    gm = '/usr/local/bin/gdalbuildvrt'
    command = [gm,'-separate', workdir+'tmpstack.vrt', sig0vvpath,
               sig0vhpath, k1vvpath, k1vhpath, k2vvpath, k2vhpath]
    subprocess.call(command)

    # load image stack and create feature list
    im_stack = geotiff(workdir+'tmpstack.vrt')
    feature_list = im_stack.data.reshape((im_stack.rows*im_stack.cols, 6))
    invalid = np.any(np.isnan(feature_list), axis=1) | (feature_list[:, 1] == 0) | (feature_list[:, 5] == 0)
    nanpos = np.where(np.isnan(feature_list))
    feature_list[nanpos] = 0

    # predictSMC
    feature_list = MLmodel['scaler'].transform(feature_list)
    smc_vector = MLmodel['SVRmodel'].predict(feature_list)
    smc_vector[invalid] = -1

    # reshape smc_vector and write map to disk
    smc_map = smc_vector.reshape((im_stack.rows, im_stack.cols))
    smc = im_stack
    smc.data = smc_map
    smc.nbands = 1
    smc.write(outpath + outname + '.tif')
    #os.unlink(outpath+'tmpstack.vrt')


def estimateSMConline(modelpath,
                      ges1vv,
                      gek1vv,
                      gek2vv,
                      roi,
                      outpath,
                      outname,
                      sampling):

    # load SVR model
    MLmodel = pickle.load(open(modelpath, 'rb'))

    # create parameter images
    alpha = [ee.Image(MLmodel.SVRmodel.best_estimator_.dual_coef_[0][i]) for i in range(len(MLmodel.SVRmodel.best_estimator_.dual_coef_[0]))]
    gamma = ee.Image(-MLmodel.SVRmodel.best_estimator_.gamma)
    intercept = ee.Image(MLmodel.SVRmodel.best_estimator_.intercept_[0])

    # support vectors stack
    sup_vectors = MLmodel.SVRmodel.best_estimator_.support_vectors_
    n_vectors = sup_vectors.shape[0]

    tmp_a = ee.Image(sup_vectors[0,0])
    tmp_b = ee.Image(sup_vectors[0, 1])
    tmp_c = ee.Image(sup_vectors[0, 2])

    sup_image = ee.Image.cat(tmp_a, tmp_b, tmp_c).select(['constant', 'constant_1', 'constant_2'],
                                           ['VV', 'VV_1', 'VV_stdDev'])
    sup_list = [sup_image]

    for i in range(1, n_vectors):
        tmp_a = ee.Image(sup_vectors[i, 0])
        tmp_b = ee.Image(sup_vectors[i, 1])
        tmp_c = ee.Image(sup_vectors[i, 2])
        sup_image = ee.Image.cat(tmp_a, tmp_b, tmp_c).select(['constant', 'constant_1', 'constant_2'],
                                               ['VV', 'VV_1', 'VV_stdDev'])
        sup_list.append(sup_image)

    input_image = ee.Image([ges1vv, gek1vv, gek2vv])
    ipt_img_mask = input_image.mask().reduce(ee.Reducer.allNonZero())

    S1mask = ges1vv.mask()
    zeromask = input_image.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())

    combined_mask = S1mask.And(zeromask).And(ipt_img_mask)

    input_image = input_image.updateMask(ee.Image([combined_mask, combined_mask,
                                                   combined_mask]))

    # scale the estimation image
    scaling_std_img = ee.Image([ee.Image(MLmodel.scaler.scale_[0].astype(np.float)),
                                ee.Image(MLmodel.scaler.scale_[1].astype(np.float)),
                                ee.Image(MLmodel.scaler.scale_[2].astype(np.float))])

    scaling_std_img = scaling_std_img.select(['constant', 'constant_1', 'constant_2'],
                                             ['VV', 'VV_1', 'VV_stdDev'])

    scaling_mean_img = ee.Image([ee.Image(MLmodel.scaler.mean_[0].astype(np.float)),
                                 ee.Image(MLmodel.scaler.mean_[1].astype(np.float)),
                                 ee.Image(MLmodel.scaler.mean_[2].astype(np.float))])

    scaling_mean_img = scaling_mean_img.select(['constant', 'constant_1', 'constant_2'],
                                               ['VV', 'VV_1', 'VV_stdDev'])

    input_image_scaled = input_image.subtract(scaling_mean_img).divide(scaling_std_img)

    k_x1x2 = [sup_list[i].subtract(input_image_scaled) \
                  .pow(ee.Image(2)) \
                  .reduce(ee.Reducer.sum()) \
                  .sqrt() \
                  .pow(ee.Image(2)) \
                  .multiply(ee.Image(gamma)) \
                  .exp() for i in range(n_vectors)]

    alpha_times_k = [ee.Image(alpha[i].multiply(k_x1x2[i])) for i in range(n_vectors)]

    alpha_times_k = ee.ImageCollection(alpha_times_k)
    alpha_times_k_sum = alpha_times_k.reduce(ee.Reducer.sum())

    estimated_smc = alpha_times_k_sum.add(intercept).round().int8()

    GEtodisk(estimated_smc, outname, outpath, sampling, roi)


def estimateSMConline_dualpol(modelpath,
                      s1,
                      outpath,
                      outname,
                      sampling):

    # load SVR model
    MLmodel = pickle.load(open(modelpath, 'rb'))
    MLmodel = {'SVRmodel': MLmodel[0], 'scaler': MLmodel[1]}

    # create parameter images
    alpha = [ee.Image(MLmodel['SVRmodel'].best_estimator_.dual_coef_[0][i]) for i in range(len(MLmodel['SVRmodel'].best_estimator_.dual_coef_[0]))]
    # alpha = ee.ImageCollection(alpha)
    # alpha = ee.Image(MLmodel.SVRmodel.best_estimator_.dual_coef_ )
    gamma = ee.Image(-MLmodel['SVRmodel'].best_estimator_.gamma)
    intercept = ee.Image(MLmodel['SVRmodel'].best_estimator_.intercept_[0])

    # support vectors stack
    sup_vectors = MLmodel['SVRmodel'].best_estimator_.support_vectors_
    n_vectors = sup_vectors.shape[0]

    tmp_a = ee.Image(sup_vectors[0,0])
    tmp_b = ee.Image(sup_vectors[0, 1])
    tmp_c = ee.Image(sup_vectors[0, 2])
    tmp_d = ee.Image(sup_vectors[0, 3])
    tmp_e = ee.Image(sup_vectors[0, 4])
    tmp_f = ee.Image(sup_vectors[0, 5])

    sup_image = ee.Image.cat(tmp_a, tmp_b, tmp_c, tmp_d, tmp_e, tmp_f).select(['constant', 'constant_1', 'constant_2',
                                                                               'constant_3', 'constant_4', 'constant_5'],
                                                                               ['VV', 'VH',
                                                                                'VV_1', 'VH_1',
                                                                                'VV_stdDev', 'VH_stdDev'])
    sup_list = [sup_image]

    for i in range(1, n_vectors):
        tmp_a = ee.Image(sup_vectors[i, 0])
        tmp_b = ee.Image(sup_vectors[i, 1])
        tmp_c = ee.Image(sup_vectors[i, 2])
        tmp_d = ee.Image(sup_vectors[i, 3])
        tmp_e = ee.Image(sup_vectors[i, 4])
        tmp_f = ee.Image(sup_vectors[i, 5])

        sup_image = ee.Image.cat(tmp_a, tmp_b, tmp_c, tmp_d, tmp_e, tmp_f).select(['constant', 'constant_1', 'constant_2',
                                                                               'constant_3', 'constant_4', 'constant_5'],
                                                                               ['VV', 'VH',
                                                                                'VV_1', 'VH_1',
                                                                                'VV_stdDev', 'VH_stdDev'])

        sup_list.append(sup_image)

    #vector_collection = ee.ImageCollection(sup_list)

    # create estimation stack
    vv = s1[1]
    vh = s1[2]
    k1_vv = s1[3]
    k1_vh = s1[4]
    k2_vv = s1[5]
    k2_vh = s1[6]

    input_image = ee.Image([vv, vh, k1_vv, k1_vh, k2_vv, k2_vh])
    #input_image = ee.Image([vv.subtract(k1_vv), vh.subtract(k2_vh)])
    ipt_img_mask = input_image.mask().reduce(ee.Reducer.allNonZero())
    S1mask = vv.mask()
    zeromask = input_image.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())
    combined_mask = S1mask.And(zeromask).And(ipt_img_mask)

    #input_image = input_image.updateMask(combined_mask)
    input_image = input_image.updateMask(ee.Image([combined_mask, combined_mask,
                                                   combined_mask, combined_mask,
                                                   combined_mask, combined_mask]))

    # scale the estimation image
    scaling_std_img = ee.Image([ee.Image(MLmodel['scaler'].scale_[0].astype(np.float)),
                                ee.Image(MLmodel['scaler'].scale_[1].astype(np.float)),
                                ee.Image(MLmodel['scaler'].scale_[2].astype(np.float)),
                                ee.Image(MLmodel['scaler'].scale_[3].astype(np.float)),
                                ee.Image(MLmodel['scaler'].scale_[4].astype(np.float)),
                                ee.Image(MLmodel['scaler'].scale_[5].astype(np.float))])

    scaling_std_img = scaling_std_img.select(['constant', 'constant_1', 'constant_2',
                                              'constant_3', 'constant_4', 'constant_5'],
                                              ['VV', 'VH',
                                               'VV_1', 'VH_1',
                                               'VV_stdDev', 'VH_stdDev'])


    scaling_mean_img = ee.Image([ee.Image(MLmodel['scaler'].center_[0].astype(np.float)),
                                 ee.Image(MLmodel['scaler'].center_[1].astype(np.float)),
                                 ee.Image(MLmodel['scaler'].center_[2].astype(np.float)),
                                 ee.Image(MLmodel['scaler'].center_[3].astype(np.float)),
                                 ee.Image(MLmodel['scaler'].center_[4].astype(np.float)),
                                 ee.Image(MLmodel['scaler'].center_[5].astype(np.float))])

    scaling_mean_img = scaling_mean_img.select(['constant', 'constant_1', 'constant_2',
                                              'constant_3', 'constant_4', 'constant_5'],
                                              ['VV', 'VH',
                                               'VV_1', 'VH_1',
                                               'VV_stdDev', 'VH_stdDev'])

    input_image_scaled = input_image.subtract(scaling_mean_img).divide(scaling_std_img)

    k_x1x2 = [sup_list[i].subtract(input_image_scaled) \
                  .pow(ee.Image(2)) \
                  .reduce(ee.Reducer.sum()) \
                  .sqrt() \
                  .pow(ee.Image(2)) \
                  .multiply(ee.Image(gamma)) \
                  .exp() for i in range(n_vectors)]

    alpha_times_k = [ee.Image(alpha[i].multiply(k_x1x2[i])) for i in range(n_vectors)]

    print(n_vectors)

    if n_vectors > 2000:

        sub_sum_list = list()

        for indx in range(0,n_vectors,100):

            start = indx
            if (indx + 99) > (n_vectors - 1):
                end = n_vectors - 1
            else:
                end = indx + 99

            sub_sum = ee.ImageCollection(alpha_times_k[start:end]).reduce(ee.Reducer.sum())

            sub_sum_list.append(sub_sum)

        alpha_times_k_sum = ee.ImageCollection(sub_sum_list).reduce(ee.Reducer.sum())

    else:

        alpha_times_k_sum = ee.ImageCollection(alpha_times_k).reduce(ee.Reducer.sum())
        #alpha_times_k_sum = alpha_times_k.reduce(ee.Reducer.sum())

    #print(alpha_times_k_sum.getInfo())

    estimated_smc = alpha_times_k_sum.add(intercept).multiply(100).round().int8()

    GEtodisk(estimated_smc, outname, outpath, sampling, s1[7])


def estimateSMConline_dualpol_2step(modelpath,
                      s1,
                      lc,
                      gldas,
                      topo,
                      outpath,
                      outname,
                      sampling):

    # load SVR model
    MLmodel_tuple = pickle.load(open(modelpath, 'rb'))
    MLmodel1 = {'SVRmodel': MLmodel_tuple[0], 'scaler': MLmodel_tuple[1]}
    MLmodel2 = {'SVRmodel': MLmodel_tuple[2], 'scaler': MLmodel_tuple[3]}

    # create parameter images
    alpha1 = [ee.Image(MLmodel1['SVRmodel'].best_estimator_.dual_coef_[0][i]) for i in range(len(MLmodel1['SVRmodel'].best_estimator_.dual_coef_[0]))]
    gamma1 = ee.Image(-MLmodel1['SVRmodel'].best_estimator_.gamma)
    intercept1 = ee.Image(MLmodel1['SVRmodel'].best_estimator_.intercept_[0])

    # support vectors stack
    sup_vectors1 = MLmodel1['SVRmodel'].best_estimator_.support_vectors_
    n_vectors1 = sup_vectors1.shape[0]
    n_features1 = 8

    tmp_list = [ee.Image(sup_vectors1[0, i]) for i in range(n_features1)]

    sup_image1 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2',
                                               'constant_3', 'constant_4', 'constant_5',
                                               'constant_6', 'constant_7'],
                                               ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                'lc', 'aspect', 'slope', 'height'])
    sup_list1 = [sup_image1]

    for i in range(1, n_vectors1):
        tmp_list = [ee.Image(sup_vectors1[i, j]) for j in range(n_features1)]

        sup_image1 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2',
                                               'constant_3', 'constant_4', 'constant_5',
                                               'constant_6', 'constant_7'],
                                               ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                'lc', 'aspect', 'slope', 'height'])
        sup_list1.append(sup_image1)

    # create estimation stack
    vv = s1[1]
    k1_vv = s1[3].rename(['VVk1'])
    k1_vh = s1[4].rename(['VHk1'])
    k2_vv = s1[5].rename(['VVk2'])
    k2_vh = s1[6].rename(['VHk2'])
    lia = s1[0].rename(['lia'])
    aspect = topo[2].rename(['aspect'])
    slope = topo[1].rename(['slope'])
    height = topo[0].rename(['height'])
    gldas_img = gldas[0].rename(['gldas'])
    gldas_mean = gldas[1].rename(['gldas_mean'])


    input_image1 = ee.Image([k1_vv.toFloat(),
                             k1_vh.toFloat(),
                             k2_vv.toFloat(),
                             k2_vh.toFloat(),
                             lc.toFloat(),
                             #lia.toFloat(),
                             aspect.toFloat(),
                             slope.toFloat(),
                             height.toFloat()])#,
                             #gldas_mean.toFloat()])
    ipt_img_mask1 = input_image1.mask().reduce(ee.Reducer.allNonZero())
    S1mask = vv.mask()
    zeromask = input_image1.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())
    combined_mask = S1mask.And(zeromask).And(ipt_img_mask1)

    input_image1 = input_image1.updateMask(ee.Image(combined_mask))

    # scale the estimation image
    scaling_std_img1 = ee.Image([ee.Image(MLmodel1['scaler'].scale_[i].astype(np.float)) for i in range(n_features1)])

    scaling_std_img1 = scaling_std_img1.select(['constant', 'constant_1', 'constant_2',
                                               'constant_3', 'constant_4', 'constant_5',
                                               'constant_6', 'constant_7'],
                                               ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                'lc', 'aspect', 'slope', 'height'])


    scaling_mean_img1 = ee.Image([ee.Image(MLmodel1['scaler'].center_[i].astype(np.float)) for i in range(n_features1)])

    scaling_mean_img1 = scaling_mean_img1.select(['constant', 'constant_1', 'constant_2',
                                               'constant_3', 'constant_4', 'constant_5',
                                               'constant_6', 'constant_7'],
                                               ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                'lc', 'aspect', 'slope', 'height'])

    input_image_scaled1 = input_image1.subtract(scaling_mean_img1).divide(scaling_std_img1)

    k_x1x2_1 = [sup_list1[i].subtract(input_image_scaled1) \
                  .pow(ee.Image(2)) \
                  .reduce(ee.Reducer.sum()) \
                  .sqrt() \
                  .pow(ee.Image(2)) \
                  .multiply(ee.Image(gamma1)) \
                  .exp() for i in range(n_vectors1)]

    alpha_times_k1 = [ee.Image(alpha1[i].multiply(k_x1x2_1[i])) for i in range(n_vectors1)]

    print(n_vectors1)

    alpha_times_k_sum_1 = ee.ImageCollection(alpha_times_k1).reduce(ee.Reducer.sum())
    #alpha_times_k_sum = alpha_times_k.reduce(ee.Reducer.sum())

    #print(alpha_times_k_sum.getInfo())

    estimated_smc_average = alpha_times_k_sum_1.add(intercept1)


    # estimate relative smc

    # create parameter images
    alpha2 = [ee.Image(MLmodel2['SVRmodel'].best_estimator_.dual_coef_[0][i]) for i in
              range(len(MLmodel2['SVRmodel'].best_estimator_.dual_coef_[0]))]
    gamma2 = ee.Image(-MLmodel2['SVRmodel'].best_estimator_.gamma)
    intercept2 = ee.Image(MLmodel2['SVRmodel'].best_estimator_.intercept_[0])

    # support vectors stack
    sup_vectors2 = MLmodel2['SVRmodel'].best_estimator_.support_vectors_
    n_vectors2 = sup_vectors2.shape[0]
    n_features2 = 3

    tmp_list = [ee.Image(sup_vectors2[0, i]) for i in range(n_features2)]

    sup_image2 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2'],
                                               ['relVV', 'relVH', 'gldas'])
    sup_list2 = [sup_image2]

    for i in range(1, n_vectors2):
        tmp_list = [ee.Image(sup_vectors2[i, j]) for j in range(n_features2)]

        sup_image2 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2'],
                                                   ['relVV', 'relVH', 'gldas'])
        sup_list2.append(sup_image2)

    # create estimation stack
    vv = s1[1]
    vh = s1[2]
    vv_mean = s1[9]
    vh_mean = s1[10]
    vv_std = s1[11]
    vh_std = s1[12]

    vv_lin = ee.Image(10).pow(vv.divide(10)).rename(['relVV'])
    vh_lin = ee.Image(10).pow(vh.divide(10)).rename(['relVH'])

    input_image2 = ee.Image([vv_lin.subtract(vv_mean).divide(vv_std).toFloat(),
                             vh_lin.subtract(vh_mean).divide(vh_std).toFloat(),
                             gldas_img.subtract(gldas_mean).rename(['gldas']).toFloat()])
    ipt_img_mask2 = input_image2.mask().reduce(ee.Reducer.allNonZero())
    S1mask = vv.mask()
    zeromask = input_image2.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())
    combined_mask = S1mask.And(zeromask).And(ipt_img_mask2)

    input_image2 = input_image2.updateMask(ee.Image(combined_mask))

    # scale the estimation image
    scaling_std_img2 = ee.Image([ee.Image(MLmodel2['scaler'].scale_[i].astype(np.float)) for i in range(n_features2)])

    scaling_std_img2 = scaling_std_img2.select(['constant', 'constant_1', 'constant_2'],
                                                   ['relVV', 'relVH', 'gldas'])

    scaling_mean_img2 = ee.Image([ee.Image(MLmodel2['scaler'].center_[i].astype(np.float)) for i in range(n_features2)])

    scaling_mean_img2 = scaling_mean_img2.select(['constant', 'constant_1', 'constant_2'],
                                                   ['relVV', 'relVH', 'gldas'])

    input_image_scaled2 = input_image2.subtract(scaling_mean_img2).divide(scaling_std_img2)

    k_x1x2_2 = [sup_list2[i].subtract(input_image_scaled2) \
                    .pow(ee.Image(2)) \
                    .reduce(ee.Reducer.sum()) \
                    .sqrt() \
                    .pow(ee.Image(2)) \
                    .multiply(ee.Image(gamma2)) \
                    .exp() for i in range(n_vectors2)]

    alpha_times_k2 = [ee.Image(alpha2[i].multiply(k_x1x2_2[i])) for i in range(n_vectors2)]

    print(n_vectors2)

    alpha_times_k_sum_2 = ee.ImageCollection(alpha_times_k2).reduce(ee.Reducer.sum())

    estimated_smc_relative = alpha_times_k_sum_2.add(intercept2)


    estimated_smc = estimated_smc_average.add(estimated_smc_relative).multiply(100).round().int8()



    GEtodisk(estimated_smc, outname, outpath, sampling, s1[7])


def estimateSMConline_linear_dualpol(modelpath,
                      s1,
                      outpath,
                      outname,
                      sampling):

    # load SVR model
    MLmodel = pickle.load(open(modelpath, 'rb'))
    MLmodel = {'LinearModel': MLmodel[0], 'scaler': MLmodel[1]}

    # create parameter images
    coef = [ee.Image(MLmodel['LinearModel'].coef_[i]) for i in range(len(MLmodel['LinearModel'].coef_))]
    intercept = ee.Image(MLmodel['LinearModel'].intercept_)

    # create estimation stack
    vv = s1[1]
    vh = s1[2]
    k1_vv = s1[3]
    k1_vh = s1[4]
    k2_vv = s1[5]
    k2_vh = s1[6]

    input_image = ee.Image([vv, vh, k1_vv, k1_vh, k2_vv, k2_vh])

    ipt_img_mask = input_image.mask().reduce(ee.Reducer.allNonZero())
    S1mask = vv.mask()
    zeromask = input_image.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())
    combined_mask = S1mask.And(zeromask).And(ipt_img_mask)

    input_image = input_image.updateMask(combined_mask)
    input_image = [input_image.slice(i) for i in range(len(coef))]

    # scale the estimation image
    scaling_std_img = [ee.Image(MLmodel['scaler'].scale_[0].astype(np.float)),
                       ee.Image(MLmodel['scaler'].scale_[1].astype(np.float)),
                       ee.Image(MLmodel['scaler'].scale_[2].astype(np.float)),
                       ee.Image(MLmodel['scaler'].scale_[3].astype(np.float)),
                       ee.Image(MLmodel['scaler'].scale_[4].astype(np.float)),
                       ee.Image(MLmodel['scaler'].scale_[5].astype(np.float))]

    scaling_mean_img = [ee.Image(MLmodel['scaler'].center_[0].astype(np.float)),
                        ee.Image(MLmodel['scaler'].center_[1].astype(np.float)),
                        ee.Image(MLmodel['scaler'].center_[2].astype(np.float)),
                        ee.Image(MLmodel['scaler'].center_[3].astype(np.float)),
                        ee.Image(MLmodel['scaler'].center_[4].astype(np.float)),
                        ee.Image(MLmodel['scaler'].center_[5].astype(np.float))]

    input_image_scaled = [input_image[i].subtract(scaling_mean_img[i]).divide(scaling_std_img[i]) for i in range(len(coef))]

    estimated_smc = ee.Image([input_image_scaled[i].multiply(coef[i]) for i in range(len(coef))]).reduce(ee.Reducer.sum()).add(intercept)

    estimated_smc = estimated_smc.round().int8()

    GEtodisk(estimated_smc, outname, outpath, sampling, s1[7])


def get_gldas(date, minlon, maxlon, minlat, maxlat):

    ee.Initialize()
    doi = ee.Date(date)
    roi = ee.Geometry.Polygon([[minlon, maxlat], [maxlon, maxlat], [maxlon, minlat], [minlon, minlat], [minlon, maxlat]])

    gldas_mean = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
                   .select('SoilMoi0_10cm_inst') \
                   .filterDate('2014-10-01', '2018-01-22').reduce(ee.Reducer.mean())

    gldas_mean = ee.Image(gldas_mean).resample().clip(roi)

    gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
              .select('SoilMoi0_10cm_inst') \
              .filterDate(doi, doi.advance(3, 'hour'))

    gldas_img = ee.Image(gldas.first()).resample().clip(roi)

    try:
        return (gldas_img, gldas_mean)
    except:
        return None


def get_globcover(minlon, maxlon, minlat, maxlat):

    # get lc
    ee.Initialize()
    roi = ee.Geometry.Polygon(
        [[minlon, maxlat], [maxlon, maxlat], [maxlon, minlat], [minlon, minlat], [minlon, maxlat]])
    globcover_image = ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3")
    land_cover = globcover_image.select('landcover').clip(roi)
    return land_cover


def get_terrain(minlon, maxlon, minlat, maxlat):
    # get elevation
    ee.Initialize()
    roi = ee.Geometry.Polygon(
        [[minlon, maxlat], [maxlon, maxlat], [maxlon, minlat], [minlon, minlat], [minlon, maxlat]])
    elev = ee.Image("CGIAR/SRTM90_V4").select('elevation').clip(roi)
    aspe = ee.Terrain.aspect(ee.Image("CGIAR/SRTM90_V4")).select('aspect').clip(roi)
    slop = ee.Terrain.slope(ee.Image("CGIAR/SRTM90_V4")).select('slope').clip(roi)

    return (elev, aspe, slop)


def createMaponline(minlon, maxlon, minlat, maxlat, outpath, sampling, year=None, month=None, day=None, tracknr=None, overwrite=False):

    if year is not None:
        # extract GE images
        try:
            images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                         year, month, day,
                                         tempfilter=True,
                                         applylcmask=False,
                                         mask_globcover=True,
                                         sampling=sampling,
                                         dualpol=False,
                                         trackflt=tracknr,
                                         maskwinter=False,
                                         masksnow=False)

        except:
            print(str(year) + '-' + str(month) + '-' + str(day) + ' no MODIS image available')
            return

        #GEtodisk(images[0], 's1img', '/mnt/SAT/Workspaces/GrF/Processing/ESA_TIGER/GEE/', 100, images[4])

        outname = 'SMCmap_' + str(images[5].year) + \
                      '_' + str(images[5].month) + '_' + str(images[5].day)

        estimateSMConline('/mnt/SAT/Workspaces/GrF/Processing/ESA_TIGER/GEE/SVR_Model_Python_S1.p',
                          images[0],
                          images[2],
                          images[3],
                          images[4],
                          outpath,
                          outname,
                          sampling)
    else:
        # get list of S1 dates
        dates = get_s1_dates(minlon, minlat, maxlon, maxlat, tracknr=tracknr, dualpol=False)
        dates = np.unique(dates)

        for dateI in dates:
            try:
                images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                             dateI.year, dateI.month, dateI.day,
                                             tempfilter=True,
                                             applylcmask=False,
                                             mask_globcover=True,
                                             sampling=sampling,
                                             dualpol=False,
                                             trackflt=tracknr,
                                             maskwinter=False,
                                             masksnow=False)

                # imageMODIS = extr_MODIS_MOD13Q1_array_reGE(minlon, minlat, maxlon, maxlat,
                #                                            dateI.year, dateI.month, dateI.day)
                # except:
                #     print(str(dateI.year) + '-' + str(dateI.month) + '-' + str(dateI.day) + ' no MODIS image available')
                #     continue

                # GEtodisk(images[0], 's1img', '/mnt/SAT/Workspaces/GrF/Processing/ESA_TIGER/GEE/', 100, images[4])

                outname = 'SMCmap_' + str(dateI.year) + '_' + str(dateI.month) + '_' + str(dateI.day)# + '_MODIS_' + str(
                   # imageMODIS[1].year) + '_' + str(imageMODIS[1].month) + '_' + str(imageMODIS[1].day)

                if overwrite == False:
                    if os.path.exists(outpath + outname + '.tif'):
                        continue

                estimateSMConline('/mnt/SAT/Workspaces/GrF/Processing/ESA_TIGER/GEE/SVR_Model_Python_S1.p',
                                  images[0],
                                  images[2],
                                  images[3],
                                  images[4],
                                  outpath,
                                  outname,
                                  sampling)
            except:
                print(str(dateI.year) + '-' + str(dateI.month) + '-' + str(dateI.day) + ' Error producing map')
                continue


def createMaponline_dualpol(minlon, maxlon, minlat, maxlat, outpath, sampling, year=None, month=None, day=None, tracknr=None, overwrite=False):

    if year is not None:
        # extract GE images
        #try:
        images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                     year, month, day,
                                     tempfilter=True,
                                     applylcmask=True,
                                     sampling=sampling,
                                     dualpol=True,
                                     trackflt=tracknr,
                                     maskwinter=False,
                                     masksnow=True)


        #GEtodisk(images[0], 's1img', '/mnt/SAT/Workspaces/GrF/Processing/ESA_TIGER/GEE/', 100, images[4])

        outname = 'SMCmap_' + str(images[8].year) + '_' + str(images[8].month) + '_' + str(images[8].day)

        estimateSMConline_dualpol('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global/mlmodelNoneSVR.p',
                                  images,
                                  outpath,
                                  outname,
                                  sampling)
        #except:
        #    print(str(year) + '-' + str(month) + '-' + str(day) + ' Error producing map')
        #    return
    else:
        # get list of S1 dates
        dates = get_s1_dates(minlon, minlat, maxlon, maxlat, tracknr=tracknr, dualpol=True)
        dates = np.unique(dates)

        for dateI in dates:
            try:
                images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                             dateI.year, dateI.month, dateI.day,
                                             tempfilter=True,
                                             applylcmask=True,
                                             sampling=sampling,
                                             dualpol=True,
                                             trackflt=tracknr,
                                             maskwinter=False,
                                             masksnow=True)

                outname = 'SMCmap_' + str(dateI.year) + '_' + str(dateI.month) + '_' + str(dateI.day)

                if overwrite == False:
                    if os.path.exists(outpath + outname + '.tif'):
                        continue

                estimateSMConline_dualpol('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global/mlmodelNoneSVR.p',
                                  images,
                                  outpath,
                                  outname,
                                  sampling)
            except:
                print(str(dateI.year) + '-' + str(dateI.month) + '-' + str(dateI.day) + ' Error producing map')
                continue


def createMaponline_2step(minlon, maxlon, minlat, maxlat, outpath, sampling, year=None, month=None, day=None, tracknr=None, overwrite=False, ex_t_mask=None):

    if year is not None:

        images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                     year, month, day,
                                     tempfilter=True,
                                     applylcmask=True,
                                     sampling=sampling,
                                     dualpol=True,
                                     trackflt=tracknr,
                                     maskwinter=False,
                                     masksnow=True,
                                     explicit_t_mask=ex_t_mask)

        gldas_img = get_gldas(str(year) + '-' + str(month) + '-' + str(day), minlon, minlat, maxlon, maxlat)
        lc = get_globcover(minlon, minlat, maxlon, maxlat)
        topo = get_terrain(minlon, minlat, maxlon, maxlat)

        outname = 'SMCmap_' + str(images[8].year) + '_' + str(images[8].month) + '_' + str(images[8].day)

        estimateSMConline_dualpol_2step('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global005_highdbtollerance/mlmodelNonewliawlcwaspectSVR2stepKernelopt.p', #mlmodelNonewliawlcwaspectSVR2stepKernelopt.p
                                        images,
                                        lc,
                                        gldas_img,
                                        topo,
                                        outpath,
                                        outname,
                                        sampling)

    else:

        # get list of S1 dates
        dates = get_s1_dates(minlon, minlat, maxlon, maxlat, tracknr=tracknr, dualpol=True)
        dates = np.unique(dates)

        for dateI in dates:
            images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                         dateI.year, dateI.month, dateI.day,
                                         tempfilter=True,
                                         applylcmask=True,
                                         sampling=sampling,
                                         dualpol=True,
                                         trackflt=tracknr,
                                         maskwinter=False,
                                         masksnow=True,
                                         explicit_t_mask=ex_t_mask)

            gldas_img = get_gldas(str(dateI.year) + '-' + str(dateI.month) + '-' + str(dateI.day), minlon, minlat, maxlon, maxlat)
            lc = get_globcover(minlon, minlat, maxlon, maxlat)
            topo = get_terrain(minlon, minlat, maxlon, maxlat)

            outname = 'SMCmap_' + str(images[8].year) + '_' + str(images[8].month) + '_' + str(images[8].day)

            if overwrite == False and os.path.exists(outpath + outname + '.tif'):
                print(outname + ' already done')
                continue

            estimateSMConline_dualpol_2step(
                '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global005_highdbtollerance/mlmodelNonewliawlcwaspectSVR2stepKernelopt.p',
                images,
                lc,
                gldas_img,
                topo,
                outpath,
                outname,
                sampling)


def ECOcreateMaponline_2step(minlon, maxlon, minlat, maxlat, outpath, sampling, year=None, month=None, day=None, tracknr=None, overwrite=False, ex_t_mask=None):

    if year is not None:

        images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                     year, month, day,
                                     tempfilter=True,
                                     applylcmask=False,
                                     mask_globcover=True,
                                     sampling=sampling,
                                     dualpol=True,
                                     trackflt=tracknr,
                                     maskwinter=False,
                                     masksnow=False,
                                     explicit_t_mask=ex_t_mask,
                                     ascending=False,
                                     maskLIA=False)

        gldas_img = get_gldas(str(year) + '-' + str(month) + '-' + str(day), minlon, maxlon, minlat, maxlat)
        lc = get_globcover(minlon, maxlon, minlat, maxlat)
        topo = get_terrain(minlon, maxlon, minlat, maxlat)

        outname = 'SMCmap_' + str(images[8].year) + '_' + str(images[8].month) + '_' + str(images[8].day)

        ECOestimateSMConline_dualpol_2step('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global005/mlmodelNonedbwliawlcwaspectSVR2step.p', #mlmodelNonewliawlcwaspectSVR2stepKernelopt.p
                                        images,
                                        lc,
                                        gldas_img,
                                        topo,
                                        outpath,
                                        outname,
                                        sampling)

    else:

        # get list of S1 dates
        dates = get_s1_dates(minlon, minlat, maxlon, maxlat, tracknr=tracknr, dualpol=True)
        dates = np.unique(dates)
        dates = np.array([x for x in dates if x.year == 2017])

        for dateI in dates:
            images = extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat,
                                         dateI.year, dateI.month, dateI.day,
                                         tempfilter=True,
                                         applylcmask=False,
                                         mask_globcover=True,
                                         sampling=sampling,
                                         dualpol=True,
                                         trackflt=tracknr,
                                         maskwinter=False,
                                         masksnow=False,
                                         explicit_t_mask=ex_t_mask,
                                         ascending=False,
                                         maskLIA=False)

            gldas_img = get_gldas(str(dateI.year) + '-' + str(dateI.month) + '-' + str(dateI.day), minlon, maxlon, minlat, maxlat)
            lc = get_globcover(minlon, maxlon, minlat, maxlat)
            topo = get_terrain(minlon, maxlon, minlat, maxlat)

            outname = 'SMCmap_' + str(images[8].year) + '_' + str(images[8].month) + '_' + str(images[8].day)

            if overwrite == False and os.path.exists(outpath + outname + '.tif'):
                print(outname + ' already done')
                continue

            ECOestimateSMConline_dualpol_2step(
                '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/gee_global005/mlmodelNonedbwliawlcwaspectSVR2step.p',
                images,
                lc,
                gldas_img,
                topo,
                outpath,
                outname,
                sampling)


def ECOestimateSMConline_dualpol_2step(modelpath,
                      s1,
                      lc,
                      gldas,
                      topo,
                      outpath,
                      outname,
                      sampling):

    # load SVR model
    MLmodel_tuple = pickle.load(open(modelpath, 'rb'))
    MLmodel1 = {'SVRmodel': MLmodel_tuple[0], 'scaler': MLmodel_tuple[1]}
    MLmodel2 = {'SVRmodel': MLmodel_tuple[2], 'scaler': MLmodel_tuple[3]}

    # create parameter images
    alpha1 = [ee.Image(MLmodel1['SVRmodel'].best_estimator_.dual_coef_[0][i]) for i in range(len(MLmodel1['SVRmodel'].best_estimator_.dual_coef_[0]))]
    gamma1 = ee.Image(-MLmodel1['SVRmodel'].best_estimator_.gamma)
    intercept1 = ee.Image(MLmodel1['SVRmodel'].best_estimator_.intercept_[0])

    # support vectors stack
    sup_vectors1 = MLmodel1['SVRmodel'].best_estimator_.support_vectors_
    n_vectors1 = sup_vectors1.shape[0]
    n_features1 = 8

    tmp_list = [ee.Image(sup_vectors1[0, i]) for i in range(n_features1)]

    sup_image1 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2',
                                                'constant_3', 'constant_4', 'constant_5',
                                                'constant_6', 'constant_7'],
                                               ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                'lc', 'lia', 'aspect', 'gldas_mean'])
    sup_list1 = [sup_image1]

    for i in range(1, n_vectors1):
        tmp_list = [ee.Image(sup_vectors1[i, j]) for j in range(n_features1)]

        sup_image1 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2',
                                                    'constant_3', 'constant_4', 'constant_5',
                                                    'constant_6', 'constant_7'],
                                                   ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                    'lc', 'lia', 'aspect', 'gldas_mean'])
        sup_list1.append(sup_image1)

    # create estimation stack
    vv = s1[1]
    k1_vv = s1[3].rename(['VVk1'])
    k1_vh = s1[4].rename(['VHk1'])
    k2_vv = s1[5].rename(['VVk2'])
    k2_vh = s1[6].rename(['VHk2'])
    lia = s1[0].rename(['lia'])
    aspect = topo[2].rename(['aspect'])
    slope = topo[1].rename(['slope'])
    height = topo[0].rename(['height'])
    gldas_img = gldas[0].rename(['gldas'])
    gldas_mean = gldas[1].rename(['gldas_mean'])

    #lc = lc.reproject(vv.projection())
    #gldas_img = gldas_img.reproject(vv.projection())
    #gldas_mean = gldas_mean.reproject(vv.projection())

    input_image1 = ee.Image([k1_vv.toFloat(),
                             k1_vh.toFloat(),
                             k2_vv.toFloat(),
                             k2_vh.toFloat(),
                             lc.toFloat(),
                             lia.toFloat(),
                             aspect.toFloat(),
                             #slope.toFloat(),
                             #height.toFloat()])#,
                             gldas_mean.toFloat()])
    ipt_img_mask1 = input_image1.mask().reduce(ee.Reducer.allNonZero())
    S1mask = vv.mask()
    zeromask = input_image1.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())
    combined_mask = S1mask.And(zeromask).And(ipt_img_mask1)

    input_image1 = input_image1.updateMask(ee.Image(combined_mask))

    # scale the estimation image
    scaling_std_img1 = ee.Image([ee.Image(MLmodel1['scaler'].scale_[i].astype(np.float)) for i in range(n_features1)])

    scaling_std_img1 = scaling_std_img1.select(['constant', 'constant_1', 'constant_2',
                                                'constant_3', 'constant_4', 'constant_5',
                                                'constant_6', 'constant_7'],
                                               ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                'lc', 'lia', 'aspect', 'gldas_mean'])


    scaling_mean_img1 = ee.Image([ee.Image(MLmodel1['scaler'].center_[i].astype(np.float)) for i in range(n_features1)])

    scaling_mean_img1 = scaling_mean_img1.select(['constant', 'constant_1', 'constant_2',
                                                  'constant_3', 'constant_4', 'constant_5',
                                                  'constant_6', 'constant_7'],
                                                 ['VVk1', 'VHk1', 'VVk2', 'VHk2',
                                                  'lc', 'lia', 'aspect', 'gldas_mean'])

    input_image_scaled1 = input_image1.subtract(scaling_mean_img1).divide(scaling_std_img1)

    k_x1x2_1 = [sup_list1[i].subtract(input_image_scaled1) \
                  .pow(ee.Image(2)) \
                  .reduce(ee.Reducer.sum()) \
                  .sqrt() \
                  .pow(ee.Image(2)) \
                  .multiply(ee.Image(gamma1)) \
                  .exp() for i in range(n_vectors1)]

    alpha_times_k1 = [ee.Image(alpha1[i].multiply(k_x1x2_1[i])) for i in range(n_vectors1)]

    print(n_vectors1)

    alpha_times_k_sum_1 = ee.ImageCollection(alpha_times_k1).reduce(ee.Reducer.sum())
    #alpha_times_k_sum = alpha_times_k.reduce(ee.Reducer.sum())

    #print(alpha_times_k_sum.getInfo())

    estimated_smc_average = alpha_times_k_sum_1.add(intercept1)


    # estimate relative smc

    # create parameter images
    alpha2 = [ee.Image(MLmodel2['SVRmodel'].best_estimator_.dual_coef_[0][i]) for i in
              range(len(MLmodel2['SVRmodel'].best_estimator_.dual_coef_[0]))]
    gamma2 = ee.Image(-MLmodel2['SVRmodel'].best_estimator_.gamma)
    intercept2 = ee.Image(MLmodel2['SVRmodel'].best_estimator_.intercept_[0])

    # support vectors stack
    sup_vectors2 = MLmodel2['SVRmodel'].best_estimator_.support_vectors_
    n_vectors2 = sup_vectors2.shape[0]
    n_features2 = 3

    tmp_list = [ee.Image(sup_vectors2[0, i]) for i in range(n_features2)]

    sup_image2 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2'],
                                               ['relVV', 'relVH', 'gldas'])
    sup_list2 = [sup_image2]

    for i in range(1, n_vectors2):
        tmp_list = [ee.Image(sup_vectors2[i, j]) for j in range(n_features2)]

        sup_image2 = ee.Image.cat(tmp_list).select(['constant', 'constant_1', 'constant_2'],
                                                   ['relVV', 'relVH', 'gldas'])
        sup_list2.append(sup_image2)

    # create estimation stack
    vv = s1[1]
    vh = s1[2]
    vv_mean = s1[9]
    vh_mean = s1[10]
    vv_std = s1[11]
    vh_std = s1[12]

    vv_lin = ee.Image(10).pow(vv.divide(10)).rename(['relVV'])
    vh_lin = ee.Image(10).pow(vh.divide(10)).rename(['relVH'])

    input_image2 = ee.Image([vv_lin.subtract(vv_mean).toFloat(),
                             vh_lin.subtract(vh_mean).toFloat(),
                             gldas_img.subtract(gldas_mean).rename(['gldas']).toFloat()])
    ipt_img_mask2 = input_image2.mask().reduce(ee.Reducer.allNonZero())
    S1mask = vv.mask()
    zeromask = input_image2.neq(ee.Image(0)).reduce(ee.Reducer.allNonZero())
    combined_mask = S1mask.And(zeromask).And(ipt_img_mask2)

    input_image2 = input_image2.updateMask(ee.Image(combined_mask))

    # scale the estimation image
    scaling_std_img2 = ee.Image([ee.Image(MLmodel2['scaler'].scale_[i].astype(np.float)) for i in range(n_features2)])

    scaling_std_img2 = scaling_std_img2.select(['constant', 'constant_1', 'constant_2'],
                                                   ['relVV', 'relVH', 'gldas'])

    scaling_mean_img2 = ee.Image([ee.Image(MLmodel2['scaler'].center_[i].astype(np.float)) for i in range(n_features2)])

    scaling_mean_img2 = scaling_mean_img2.select(['constant', 'constant_1', 'constant_2'],
                                                   ['relVV', 'relVH', 'gldas'])

    input_image_scaled2 = input_image2.subtract(scaling_mean_img2).divide(scaling_std_img2)

    k_x1x2_2 = [sup_list2[i].subtract(input_image_scaled2) \
                    .pow(ee.Image(2)) \
                    .reduce(ee.Reducer.sum()) \
                    .sqrt() \
                    .pow(ee.Image(2)) \
                    .multiply(ee.Image(gamma2)) \
                    .exp() for i in range(n_vectors2)]

    alpha_times_k2 = [ee.Image(alpha2[i].multiply(k_x1x2_2[i])) for i in range(n_vectors2)]

    print(n_vectors2)

    alpha_times_k_sum_2 = ee.ImageCollection(alpha_times_k2).reduce(ee.Reducer.sum())

    estimated_smc_relative = alpha_times_k_sum_2.add(intercept2)


    estimated_smc = estimated_smc_average.add(estimated_smc_relative).multiply(100).round().int8()

    # mask negative values
    estimated_smc = estimated_smc.updateMask(estimated_smc.gt(0))


    GEtodisk(estimated_smc, outname, outpath, sampling, s1[7], timeout=False)


def createMap(minlon, maxlon, minlat, maxlat, year, month, day, workdir, outpath, sampling, tracknr=None):

    # download S1 data
    extr_GEE_array(minlon, minlat, maxlon, maxlat,
                   year, month, day,
                   workdir,
                   tempfilter=True,
                   applylcmask=True,
                   sampling=sampling,
                   dualpol=True,
                   trackflt=tracknr,
                   maskwinter=True)

    list_files = os.listdir(workdir)
    list_files.sort()

    estimateSMC('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_subset_CE/mlmodel.p',
                workdir + list_files[0],
                workdir + list_files[1],
                workdir + list_files[2],
                workdir + list_files[3],
                workdir + list_files[4],
                workdir + list_files[5],
                outpath,
                workdir,
                str(year)+ str(month) + str(day))

    # for thefile in os.listdir(workdir):
    #     file_path = workdir + thefile
    #     try:
    #         if os.path.isfile(file_path):
    #             os.unlink(file_path)
    #     except Exception as e:
    #         print(e)


def getTimeSeries(lon, lat, outpath, sampling, tracknr, modelpath, create_plot=True, save_as_txt=True, calc_anomalies=False, name=None):

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

def getKenyaRoi():

    roi = ee.Geometry.Polygon([[37.6611328125,-3.5134210456400323],[39.17724609375,-4.696879026871413],[39.6826171875,-3.710782004348708],[39.9462890625,-2.8772079526533365],[40.80322265625,-1.9332268264771106],[41.66015625,-1.7355743631421197],[41.0009765625,-0.856901647439813],[41.06689453125,2.8113711933311403],[41.90185546875,3.951940856157594],[41.1767578125,3.9300201571114752],[40.75927734375,4.324501493019203],[39.92431640625,3.886177033699361],[39.55078125,3.4037578795775882],[38.671875,3.6230713262356864],[38.14453125,3.5572827265412794],[36.93603515625,4.390228926463396],[36.14501953125,4.390228926463396],[35.3759765625,5.00339434502215],[34.56298828125,4.740675384778373],[33.99169921875,4.258768357307995],[34.43115234375,3.7765593098768635],[34.4091796875,3.228271011252635],[34.892578125,2.6796866158037598],[35.1123046875,1.7355743631421197],[34.8486328125,1.3182430568620136],[34.43115234375,0.9447814006874024],[33.9697265625,0.15380840901698828],[33.99169921875,-0.9667509997666298],[37.59521484375,-3.008869978848142]])

    return(roi)