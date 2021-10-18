import os
from patchify import patchify
import cv2

image_directory = 'images/'
mask_directory = 'masks/'

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory+image_name)
        patches_img = patchify(image, (256, 256,3), step=256)
        for j in range(patches_img.shape[0]):
            for k in range(patches_img.shape[1]):
                single_patch_img = patches_img[j, k, 0, :, :, :]
                cv2.imwrite('patches/2_images/' + 'image_' + '_' + str(i) + '_' +str(j) + str(k) + '.png', single_patch_img)

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if image_name.split('.')[1] == 'png':
        image = cv2.imread(mask_directory+image_name)
        patches_mask = patchify(image, (256, 256, 3), step=256)
        for j in range(patches_mask.shape[0]):
            for k in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[j,k,0,:,:,:]
                cv2.imwrite('patches/2_masks/' + 'mask_' + '_' + str(i) + '_' + str(j)+str(k)+ ".png", single_patch_mask)
                single_patch_mask = single_patch_mask / 255.
            

