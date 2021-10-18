#import numpy as np
#from matplotlib import pyplot as plt
from patchify import patchify
import cv2

large_image_stack = cv2.imread('images/d1-b4-001.jpg')
large_mask_stack = cv2.imread('masks/d1-b4-001.png')

patches_img = patchify(large_image_stack, (256, 256,3), step=256)  # Step=256 for 256 patches means no overlap

for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, 0, :, :, :]
        cv2.imwrite('patches/images/' + 'image_' + '_' + str(i) + str(j) + '.png', single_patch_img)

patches_mask = patchify(large_mask_stack, (256, 256, 3), step=256)  #Step=256 for 256 patches means no overlap
    

for i in range(patches_mask.shape[0]):
    for j in range(patches_mask.shape[1]):
        single_patch_mask = patches_mask[i,j,0,:,:,:]
        cv2.imwrite('patches/masks/' + 'mask_' + '_' + str(i)+str(j)+ ".png", single_patch_mask)
        single_patch_mask = single_patch_mask / 255.
            

