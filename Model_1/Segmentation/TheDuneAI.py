# -*- coding: utf-8 -*-
"""
@author: s.primakov
"""
import SimpleITK as sitk
import numpy as np
import keras
import cv2
import time
import os

#Import pipeline functions
import lung_extraction_funcs_13_09 as le
import Generator_v1 as Generator_v1

from tqdm import tqdm

class ContourPilot:
    def __init__(self, model_path, data_path, output_path='./',verbosity=False,pat_dict = None):
        self.verbosity = verbosity
        self.model1 = None
        self.__load_model__(model_path)
        if pat_dict:
            self.Patient_dict = pat_dict
        else:
            self.Patient_dict = le.parse_dataset(data_path,img_only=True)
        self.Patients_gen = Generator_v1.Patient_data_generator(self.Patient_dict, predict=True, batch_size=1, image_size=512,shuffle=True,  
                                      use_window = True, window_params=[1500,-600],resample_int_val = True, resampling_step =25,   #25
                                      extract_lungs=True, size_eval=False,verbosity=verbosity,reshape = True, img_only=True)
        self.Output_path = output_path
        
    def __load_model__(self, model_path):
        json_file = open(os.path.join(model_path,'model_v7.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model1 = keras.models.model_from_json(loaded_model_json)
        self.model1.load_weights(os.path.join(model_path,'weights_v7.hdf5'))

    def __generate_segmentation__(self,img,params,thr=0.99):
        temp_pred_arr = np.zeros_like(img)
        if self.verbosity:
            print('Segmentation started')
            st = time.time()
        for j in range(len(img)):  
            predicted_slice = self.model1.predict(img[j,...].reshape(-1,512,512,1)).reshape(512,512)
            temp_pred_arr[j,...] = 1*(predicted_slice> thr)
        if self.verbosity:
            print('Segmentation is finished')
            print('time spent: %s sec.'%(time.time()-st))
        predicted_arr_temp = le.max_connected_volume_extraction(temp_pred_arr)
        temporary_mask = np.zeros(params['normalized_shape'],np.uint8)
        
        if params['crop_type']:
            temporary_mask[params['z_st']:params['z_end'],...] = predicted_arr_temp[:,params['xy_st']:params['xy_end'],params['xy_st']:params['xy_end']]
        else:
            temporary_mask[params['z_st']:params['z_end'],params['xy_st']:params['xy_end'],params['xy_st']:params['xy_end']] = predicted_arr_temp

        if temporary_mask.shape != params['original_shape']:
            predicted_array  = np.array(1*(le.resize_3d_img(temporary_mask,params['original_shape'],cv2.INTER_NEAREST)>0.5),np.int8)
        else:
            predicted_array  = np.array(temporary_mask,np.int8)

        return predicted_array
    
    """ def segment(self):
        if self.model1 and self.Patient_dict and self.Output_path:
            count=0
            for img,_,filename,params in tqdm(self.Patients_gen, desc='Progress'):

                filename=filename[0]
                params=params[0]
                img = np.squeeze(img)

                if img.size == 0:  # Skip empty images
                    print(f"Warning: Empty image encountered for file {filename}")
                    continue
                
                # Generate the segmentation
                try:
                    predicted_array = self.__generate_segmentation__(img, params)
                except Exception as e:
                    print(f"Error during segmentation of {filename}: {str(e)}")
                    continue

                predicted_array = self.__generate_segmentation__(img,params)

                folder_name = os.path.basename(os.path.dirname(filename))
                path_out = os.path.join(self.Output_path, f"{folder_name}_(DL)")

                if not os.path.exists(path_out):
                    os.makedirs(path_out)
                                
                # Guardar la máscara generada
                generated_img = sitk.GetImageFromArray(predicted_array)
                generated_img.SetSpacing(params['original_spacing'])
                generated_img.SetOrigin(params['img_origin'])
                sitk.WriteImage(generated_img, os.path.join(path_out, 'DL_mask.nrrd'))

                # Guardar la imagen original
                temp_data = sitk.ReadImage(filename)
                sitk.WriteImage(temp_data, os.path.join(path_out, 'image.nrrd'))

                if count == len(self.Patients_gen):
                    return 0

                count+=1
 """
    def segment(self):
        if not (self.model1 and self.Patient_dict and self.Output_path):
            print("[Error] Model, Patient_dict, or Output_path is not set.")
            return

        count = 0

        for sample in tqdm(self.Patients_gen, desc='Progress'):
            try:
                img, _, filename, params = sample

                filename = filename[0]
                params = params[0]
                img = np.squeeze(img)

                if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                    print(f"[Warning] Skipping empty or invalid image: {filename}")
                    continue

                # Generate the segmentation
                predicted_array = self.__generate_segmentation__(img, params)

                # Prepare output path
                folder_name = os.path.basename(os.path.dirname(filename))
                path_out = os.path.join(self.Output_path, f"{folder_name}_(DL)")
                os.makedirs(path_out, exist_ok=True)

                # Save mask
                generated_img = sitk.GetImageFromArray(predicted_array)
                generated_img.SetSpacing(params['original_spacing'])
                generated_img.SetOrigin(params['img_origin'])
                sitk.WriteImage(generated_img, os.path.join(path_out, 'DL_mask.nrrd'))

                # Save original image
                temp_data = sitk.ReadImage(filename)
                sitk.WriteImage(temp_data, os.path.join(path_out, 'image.nrrd'))

                count += 1

            except Exception as e:
                print(f"[Error] Problem with file {filename if 'filename' in locals() else 'unknown'}: {e}")
                continue

        print(f"[Done] Processed {count} patient(s).")
