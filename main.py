import os
import sys
import numpy as np
import pandas as pd
import rasterio as rio
import time
import tensorflow as tf
import re
import xarray as xr
import rioxarray
import glob

from skimage import io as skio
from skimage import util as skutil

# Import modules from lib/ directory
from lib.STpconvLayer import STpconv
from lib.STpconvUnet import STpconvUnet
from lib.DataGenerator import DataGenerator

import gc
import sys
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

# uncomment for computation time measurements
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)


def smooth_transitions(existing, predicted, mask, sigma=2):
    """
    Smooths the transitions between existing and predicted values using Gaussian blurring.

    :param existing: Original image with existing values.
    :param predicted: Image with predicted values.
    :param mask: Binary mask indicating missing values (0 for missing, 1 for existing).
    :param sigma: Standard deviation for Gaussian kernel.
    :return: Smoothed image.
    """
    combined = existing.copy()
    combined[mask == 0] = predicted[mask == 0]

    blurred = gaussian_filter(combined, sigma=sigma)
    blurred[predicted == pred_const] = -3.4e+38
    smoothed = existing * mask + blurred * (1 - mask)
    return smoothed


def predict_single_overlap(input_dt, sub_itert, n):
    temp = input_dt.values.transpose(1, 2, 0)
    temp = temp/5
    temp = np.expand_dims(temp, axis=0)  # append "sample" dimension
    temp = np.expand_dims(temp, axis=4)  # append "channel" dimension
    temp[temp < 0] = 0
    X = temp.copy()

    mask = temp.copy()
    mask[mask != 0] = 1

    # 2nd mask is not needed here, so we simply use the same mask
    pred = model.predict([X, mask, mask])
    pred = np.clip(pred, a_min=0, a_max=1.0)  # set the upper limit to 5
    # pred[pred == pred_const] = -3.4e+38 / 5
    # pred[X > 0] = X[X > 0]
    pred = smooth_transitions(X, pred, mask, sigma=2)
    pred = pred*5

    # create output file and copy spatial reference from input
    # input_dt.values[:] = pred[0,:,:,:,0].transpose(2,0,1)
    pred = pred[0, :, :, :, 0].transpose(2, 0, 1)
    sub_itert.values[(input_dt.values < 0) & (pred > 0)] = n+1
    input_dt.values[input_dt.values < 0] = pred[input_dt.values < 0]

    return input_dt, sub_itert

# def most_frequent_value_2d(array):
#     # Flatten the array and remove NaN values
#     flat_array = array[~np.isnan(array)]

#     # Check if the array is empty after removing NaN values
#     if flat_array.size == 0:
#         return None, 0

#     # Get unique values and their counts excluding 5
#     unique_values, counts = np.unique(flat_array[flat_array!=5], return_counts=True)

#     # Find the index of the maximum count
#     max_count_index = np.argmax(counts)

#     return unique_values[max_count_index], counts[max_count_index]


# List of TIFF file paths to concatenate
# Define a custom sorting key function


def sort_by_number(file_name):
    # Extract the second number from the file name
    # Assuming the file names are like "prefix_number1_number2_suffix"
    file_number = int(re.split(r'[_.]', file_name)[3])
    return file_number


print("Using TensorFlow version", tf.__version__)
training_len = 36

DATA_PATH_IN = "/rs1/researchers/z/zqu5/abloom/data/processed_goes/2025/"
DATA_PATH_OUT = "/rs1/researchers/z/zqu5/abloom/data/april_only_aod/"

if os.path.exists(DATA_PATH_OUT):
    # if len(os.listdir(DATA_PATH_OUT)) > 0:
    # sys.exit("Output directory exists and is not empty")
    print('Output directory exists')
else:
    os.makedirs(DATA_PATH_OUT)

model2 = STpconvUnet.load("model_architecture_MERRA2_36_test22norm500",
                          weights_name="out_MERRA2_36_test22norm500/epoch_05.h5")
# model3 = STpconvUnet.load("model_architecture_MERRA2_"+str(training_len)+"_test522norm", weights_name = "out_MERRA2_"+str(training_len)+"_test522norm/epoch_18.h5")
# model4 = STpconvUnet.load("model_architecture_MERRA2_"+str(training_len)+"_test5522norm", weights_name = "out_MERRA2_"+str(training_len)+"_test5522norm/epoch_30.h5")

X = np.zeros([1, 500, 800, 36, 1])
mask = X > 0
pred_const_list = []
pred = model2.predict([X, mask, mask])
pred_const_list.append(pred[0, 0, 0, 0, 0])
# pred = model3.predict([X,mask, mask])
# pred_const_list.append(pred[0,0,0,0,0])
# pred = model4.predict([X,mask, mask])
# pred_const_list.append(pred[0,0,0,0,0])

water_mask = np.zeros((4, 5, 500, 800))
# for r in range(4):
#     for c in range(5):
#         water_mask[r, c, :, :] = rioxarray.open_rasterio(
#             'water_mask_1400_3200/water_mask_'+str(r)+'_'+str(c)+'.tif').values[0]

day_index = range(12, 8760, 24)

for i in range(int(sys.argv[1])*30, (int(sys.argv[1])+1)*30):
    tif_paths = [
        f"{DATA_PATH_IN+'GOES16_0.02'}_{j}.tif" for j in range(day_index[i], day_index[i]+36)]
    arrays = [rioxarray.open_rasterio(
        tif_path).fillna(-3.4e+38) for tif_path in tif_paths]
    input_data = xr.concat(arrays, dim='band')
    # remove AOD lower than MERRA2 min
    input_data = input_data.where(input_data >= 0.007162074, other=-3.4e+38)
    # remove AOD higher than 5 updated on May 21, 2024
    input_data = input_data.where(input_data < 5, other=-3.4e+38)
    input_data['band'] = np.arange(1, 37)
    input_data['x'] = input_data.x.round(2)
    input_data['y'] = input_data.y.round(2)
    iter_times = input_data.copy()
    iter_times = iter_times.where(iter_times <= 0, 0)

    missing_ratio = pd.DataFrame(columns=[0, 1, 2, 3, 4])
    for r in range(4):
        for c in range(5):
            sub_tmp = input_data[:, 300 *
                                 r:(500+300*r), 600*c:(800+600*c)].copy()
            missing_ratio.loc[r, c] = np.count_nonzero(
                sub_tmp == -3.4e+38)/36/500/800
    missing_ratio.to_csv(DATA_PATH_OUT+'missing_ratio_'+str(i)+'.csv')
    df_b = missing_ratio.stack().reset_index()
    df_b.columns = ['Row Index', 'Column Name', 'Sorted Values']
    df_b.sort_values('Sorted Values', inplace=True)
    df_b.reset_index(drop=True, inplace=True)

    start = time.process_time()
    n = 0
    while df_b['Sorted Values'][19] > 0:
        print(n)
        for k in range(20):
            r = df_b['Row Index'][k]
            c = df_b['Column Name'][k]
            if df_b['Sorted Values'][k] == 0:
                continue

            sub_input = input_data[:, 300*r:(500+300*r), 600*c:(800+600*c)]
            sub_itert = iter_times[:, 300*r:(500+300*r), 600*c:(800+600*c)]

            model = model2
            pred_const = pred_const_list[0]

            sub_input, sub_itert = predict_single_overlap(
                sub_input, sub_itert, n)
            # input_data.where(input_data > 0)[40].plot(vmin=0,vmax=1)
            # plt.show()
            # iter_times.plot()
            # plt.show()
            gc.collect()

        missing_ratio = pd.DataFrame(columns=[0, 1, 2, 3, 4])
        for r in range(4):
            for c in range(5):
                sub_tmp = input_data[:, 300 *
                                     r:(500+300*r), 600*c:(800+600*c)].copy()
                # sub_tmp = sub_tmp * water_mask[r, c]
                missing_ratio.loc[r, c] = round(
                    np.count_nonzero(sub_tmp[12:] < 0)/24/500/800, 10)

        df_b = missing_ratio.stack().reset_index()
        df_b.columns = ['Row Index', 'Column Name', 'Sorted Values']
        df_b.sort_values('Sorted Values', inplace=True)
        df_b.reset_index(drop=True, inplace=True)

        print(df_b)
        n = n + 1

    t = time.process_time() - start
    (input_data[12:]).rio.to_raster(DATA_PATH_OUT +
                                    "PRED_" + str(training_len) + 'h_' + str(i) + '.tif')
    (iter_times[12:]).rio.to_raster(DATA_PATH_OUT +
                                    "Iter_times_" + str(training_len) + 'h_' + str(i) + '.tif')
    print("DAY"+str(i) + " DONE.")
