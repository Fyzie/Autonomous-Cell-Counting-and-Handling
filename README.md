# Cell Confluency Using Image Segmentation

## Project Details
Currently employed methods:  
1- U-Net CNN (Supervised Learning)  
2- K-Means Clustering (Unsupervised Learning)  

Language: Python  

Platform IDEs:  
1- Anaconda (Environment Setup)  
2- Pycharm (Script development)  
3- Spyder (Data viewer & visualization)  

Requirements and Libraries:  
1- Tensorflow Installation Setup [[Reference](https://www.youtube.com/watch?v=hHWkvEcDBO0)]  
2- Libraries: opencv-python, matplotlib, numpy, keras, pillow, sklearn, imutils  

## Current Progress
### 1. U-Net CNN
<details open>
<summary>Methodology</summary>
<br>
 
**Step 1-** Data annotation. Cells are labelled to distinguish between the background and the cell confluency.  
  
**Step 2-** (Optional) Data/ Image patching. If your data resolutions/ dimensions are high, this step is recommended for GPU support. [[Reference 1](https://github.com/Fyzie/Cell-Confluency/blob/main/U-Net%20CNN/patch_images_masks_for_large_image.py)] [[Reference 2](https://github.com/bnsreenu/python_for_microscopists/blob/master/Tips_Tricks_5_extracting_patches_from_large_images_and_masks_for_semantic_segm.py)]
  
**Step 3-** U-Net model development. [[Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/204-207simple_unet_model.py)]  
  
**Step 4-** Data Training. [[Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/204_train_simple_unet_for_mitochondria.py)]
  
**Step 5-** Model Inference and Prediction. [[Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/204_train_simple_unet_for_mitochondria.py)]
  
**Step 6-** Model Application. [[Reference 1](https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py)] [[Reference 2](https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_patchify.py)]

[[Youtube Reference](https://www.youtube.com/watch?v=csFGTLT6_WQ&t=1295s)]
 
</details>

<details open>
<summary>Arising Problems</summary>
<br>

A. GPU exhaustion leads to the needs for:  
  (i) batch size reduction  
  (ii) dimensions reduction  
  (iii) data training reduction
  
B. Bad model development leads to:  
  (i) error prediction and inference  

</details>

### 2. K-Means Clustering

<details open>
<summary>Methodology</summary>
<br>

**Step 1-** Image Normalization.
  
**Step 2-** Image Flattening.
  
**Step 3-** K-Means Clustering. Identify number of clusters accordingly to available distinct regions.
  
**Step 4-** Labeled Image Restructuring.
  
**Step 5-** Data Reshaping.
  
[[Reference](https://github.com/Fyzie/Cell-Confluency/blob/main/K-means/Kmeans_sklearn_for_image_segmentation.py)]
  
</details>

<details open>
<summary>Arising Problems</summary>
<br>
  
A. Bad clustering may due to:  
  (i) irregular brightness of image datasets (trying to normalize them in Step 1) 
  
</details>

## Image Segmentation
Image segmentation is a process of partitioning images into multiple distinguished regions containing each pixel with similar features/ attributes.




