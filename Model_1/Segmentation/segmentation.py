###################################################################
## Import DL packages setting up GPU if used
import os
import tensorflow as tf
import keras.backend as K
import sys
sys.path.append('/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/src/Model_1/Segmentation')
from TheDuneAI import ContourPilot as cp 

#If you have an available GPU and tensorflow-gpu >=1.15.0, CUDA >= 10.0.130, CuDNN installed you can try setting gpu=True
GPU_compute = False
if GPU_compute:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ['CUDA_VISIBLE_DEVICES']='0'          #Choose GPU device ID
    #Check availableGPUs
    print(K.tensorflow_backend._get_available_gpus())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # évite l'allocation totale
        print("[INFO] GPU memory growth activé.")
    except RuntimeError as e:
        print(f"[ERREUR] Impossible de configurer GPU: {e}")

###################################################################
##Initialize model inputs

model_path = '/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Segmentation/model_files'
path_to_test_data = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/NRRD/converted_nrrds/'
save_path = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/Processed'

#initialize the model
model = cp(model_path,path_to_test_data,save_path,verbosity=True)    #set verbosity =True to see what is going on

###################################################################
## Estimated segmentation time per patient:
##      with GPU(RTX2080TI): 2-3 sec
##      with CPU(Core i5-7200U ): 170 -180 sec
## Estimated processing time per patient depends on the multiple parameters such as: CPU/GPU usage, Hardware (HDD/SSD),
## and length of the CT scan (whole body scan CT/ thorax CT) 
## The estimated processing time per pat. range is: 25 sec - 280 sec.
###################################################################
##Starting the segmentation process

model.segment()

