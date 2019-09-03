import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from keras.models import *
from keras.layers import *
from sklearn.utils import shuffle
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from Train_Seg_FCN import getImageArr, getSegmentationArr, give_color_to_seg_img, FCN8_custom, DSC, post_processing
from helper import sliding_window
import PIL
import cv2 as cv

cwd = os.getcwd()
# Task: Draw the ground truth on output images of this script. Add these images to manuscript.
# Solve this problem: The output of combining binary masks together is not the same with binary mask image output of combining slice images. 
# However, the binary mask of slices are correct. Look at post_processing method

# This script works on slices images that are generated from padded tile images.  This script loads the trained FCN model and save the segmentation results for slices.
# Then, build the results for tile images by combining the output slices. 

def extract_data_adds(img_folder):
	# The function extracts the image addresses, and image files.
    img_files = os.listdir(img_folder)    # list of images in subfolder
    img_adds = []  # a list of address of images
    
    img_folders = [f for f in img_files]
    
    if(len(img_folders) == 0):
        input('Warning: Train directory is empty')
    #j = 0 
    for i in range(len(img_files)):   #append images and masks to the list of files
        img_adds += [os.path.join(img_folder,img_files[i])]
           
    return img_adds, img_files

def extract_data(X_add, input_width, input_height):
    # the function receives two list of image and mask addresses and return all the images and masks in two 3-D array; X, Y
    X = [] # list of images
    for img_add in X_add:
            if img_add.endswith('.jpg'): X.append( getImageArr(img_add , input_width , input_height))
    X = np.array(X) 
    return X
	
# this function combines 2d sub binary masks and make a big binary mask for tile image.
# the args argument contains 64 slices of a tile. 
def combine_matrix(args, slice_dim = (224,224), output_dim = (1792,1792)):  
    n=len(args)
    rows,cols=slice_dim
    a=np.zeros(output_dim)
    m=0
    for i in range(int(math.sqrt(n))):
        for j in range(int(math.sqrt(n))):
            a[i*rows:(i+1)*rows,j*cols:(j+1)*cols]=args[m]
            #a[i*rows:(i+1)*rows,j*cols:(j+1)*cols]=args[m]
            m+=1

    return a

def combine_mask_images(output_save_path, tile_img, mask_img_slices_dir, gt_num, windowSize=(224, 224), stepSize = 224):
    
    #clone = tile_img.copy()
    counter = 0 
    img_list_v = []
    
    for y in range(0, tile_img.shape[0], stepSize):
        img_list_h = []
        for x in range(0, tile_img.shape[1], stepSize):
            if y + windowSize[0] <= tile_img.shape[0] and x + windowSize[1] <= tile_img.shape[1] : 
                
                dst_filename = os.path.join(mask_img_slices_dir, "SL_"+ gt_num + "_" + str(counter) +  "_x_" + str(x) +"_y_" + str(y) +  ".jpg")  
                #print("dst_filename: " + str(dst_filename))
                #img_list_h += [PIL.Image.open(dst_filename)]
                img_list_h.append(PIL.Image.open(dst_filename))
                counter +=1
        
        # combine images horizontally 
        #print("counter: " + str(counter))
        if len(img_list_h) > 0 :
            
            #print("len of img_list_h before hstack: " + str(len(img_list_h)))
            row_img_list = img_list_h
            row_img = np.hstack(np.asarray(i) for i in row_img_list) 
            row_img = PIL.Image.fromarray( row_img)
            #print("len of img_list_h after hstack: " + str(len(img_list_h)))
            #img_list_v += [row_img]
            img_list_v.append(row_img)
        #print("y: " + str(y) + " , img_list_h:" + str(len(img_list_h))
        #print("len of img_list_v : " + str(len(img_list_v)))
        
                
    #print("len of img_list_v: " + str(len(img_list_v)))
    tile_img_list = img_list_v
    tile_img = np.vstack(np.asarray(i) for i in tile_img_list)
    tile_img_pil = PIL.Image.fromarray( tile_img)
    #the size of this image is 1792 x 1792. 
    tile_img_pil.save(output_save_path+'tile'+gt_num+'_1792.png')
    return tile_img_pil
    #png.from_array(tile_img, 'L').save(output_save_path+'tile'+gt_num+'_1792.png')
        

if __name__ == "__main__":
    
    dir_data = "crater_data/slices/org_noverlap/"   # Directory where the data is saved
    dir_data_tiles = "crater_data/tiles/" # Directory of tile images
    sub_img_mask_save_dir = "results/output_sub_img_masks_noverlap/" # output directory for saving sub segmented images.
    sub_bin_mask_save_dir = "results/output_sub_bin_masks_noverlap/" # output directory for saving sub binary mask of segmented images
    tile_img_mask_save_dir = "results/output_img_masks_noverlap/" # output directory for saving tile segmented image
    tile_bin_mask_save_dir = "results/output_bin_masks_noverlap/" # output directory for saving tile binary mask
	
    n_classes= 2    # number of classes in the image, 2 as foreground and background
    input_height , input_width = 224 , 224
    
    model = FCN8_custom(n_classes, input_height, input_width)
    
    model.load_weights("models/seg-fcn-model.h5")
    print("Model loaded .")
    # go through all the tile folders
    #gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]
    gt_list = ["1_24"]
    
    for gt_num in gt_list:
        
        print("Working on tile: tile" +gt_num )
        # big tile image mask
        #tile_img_mask = np.zeros([1700,1700,3],dtype=np.uint8)

        # list of small image masks
        img_mask_list = []
		
        tile_dir_data = dir_data + 'tile' + gt_num + '/'
        tile_img_refelected_path = dir_data_tiles + 'tile' + gt_num + '_reflected.png' # input: directory of reflected (padded) tile image
        tile_sub_img_mask_save_dir = sub_img_mask_save_dir + 'tile' + gt_num + '/'
        tile_sub_bin_mask_save_dir = sub_bin_mask_save_dir + 'tile' + gt_num + '/'
        # 
        tile_img_refelected = cv.imread(tile_img_refelected_path)
        X_add, X_files = extract_data_adds(tile_dir_data)  # Extract a list of addresses: image addresses.
        
        X = extract_data(X_add, input_width, input_height)    # one arrays of images are extracted.
   
        y_pred = model.predict(X)
        y_predi = np.argmax(y_pred, axis=3)     # predicted class number of every pixel in the image, shape = sample_no x height x width
        y_predi_post = post_processing(y_predi)
        count = 0 
    
        # the next for loop shows the testing image, ground truth and segmented area.
        for i in range(X.shape[0]):
            count += 1
            img_filename = X_files[i]
            binmask_filename = img_filename[:-3] + "csv"
            img_is  = (X[i] + 1)*(255.0/2)
            seg = y_predi_post[i] # segmented image after post processing
            
            # save segmented image on output_sub_masks_noverlap folder. We need tiles folders inside this folder.
            mpimg.imsave(tile_sub_img_mask_save_dir + img_filename,give_color_to_seg_img(seg,n_classes) ,format='png', dpi=400)
            img_mask_list += [give_color_to_seg_img(seg,n_classes)]
            # save binary matrix segmented image on output_sub_matrix_noverlap folder. We need tiles folders inside this folder.
            # example of saving a numpy array: np.savetxt("foo.csv", a, fmt='%d')
            # example of loading a numpy array: b = np.loadtxt("foo.csv", dtype=int)
            np.savetxt(tile_sub_bin_mask_save_dir + binmask_filename, y_predi_post[i], fmt='%d')
			            
        print("Save segmented image and binary masks for tile" + str(gt_num) + " are done. " +str(count * 2) + " files are saved in results directory.")
	
	    # convert the tiles segmented image and binary seg usbmatrixes into one big image and one big binary mask.
        tile_img = combine_mask_images(tile_img_mask_save_dir, tile_img_refelected, tile_sub_img_mask_save_dir, gt_num)
        print("The segmented output saved.")
	    # save the binary mask matrix into csv format.
        # big tile binary mask
        #print("len of y_predi_post: " + str(y_predi_post))
        tile_binary_mask = combine_matrix(y_predi_post)
        #the output of combine_matrix is a 1792 x 1792 because the input slices contains padding
        #we need to convert the image into 1700 x 1700 matrix. 
        #tile_img_2 = tile_img[:1700,:1700]
        #np.savetxt(tile_bin_mask_save_dir + "bin_mask_tile" + gt_num + ".csv", tile_img[:,:,:3], fmt='%d')
        #np.savetxt(tile_bin_mask_save_dir + "bin_mask_tile" + gt_num + ".csv", tile_binary_mask[:1700,:1700], fmt='%d') 
        np.savetxt(tile_bin_mask_save_dir + "bin_mask_tile" + gt_num + ".csv", tile_binary_mask, fmt='%d')
        print("The binary mask of segmented output saved.")
        
        res = cv.bitwise_and(tile_img,tile_img,mask = tile_binary_mask)
    
    
    print("end of main")
    
    
