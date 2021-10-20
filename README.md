# Cell Confluency Using Image Segmentation

## Project Details
Currently employed methods:  
1- U-Net CNN (Supervised Learning)  
2- K-Means Clustering (Unsupervised Learning)  

Language: Python  

Platform IDEs:  
1- Anaconda (Environment Setup) [[Installation](https://www.anaconda.com/products/individual#Downloads)] 
2- Pycharm (Script development) [[Installation](https://www.jetbrains.com/pycharm/download/)]
3- Spyder (Data viewer & visualization) [[Installation](https://www.spyder-ide.org/)] [[Anaconda Envs Preferences](https://medium.com/@apremgeorge/using-conda-python-environments-with-spyder-ide-and-jupyter-notebooks-in-windows-4e0a905aaac5)]  [[Compatibility](https://docs.spyder-ide.org/5/troubleshooting/common-illnesses.html)]  

Requirements and Libraries:  
1- Data Annotation Software/ Website; Suggested: [[LabelMe](https://github.com/wkentaro/labelme)] [[Apeer](https://www.apeer.com/app)]; Discoverable: [[Tools](https://github.com/taivop/awesome-data-annotation)]  
2- Tensorflow Installation Setup [[Reference](https://www.youtube.com/watch?v=hHWkvEcDBO0)]  
3- Libraries: opencv-python, matplotlib, numpy, keras, pillow, sklearn, imutils  

How to Install LabelMe:  
Install [Anaconda](https://www.anaconda.com/products/individual#Downloads), then in an Anaconda Prompt run:
```
conda create --name=labelme python=3.6
conda activate labelme
pip install labelme
``` 

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

[[Youtube Reference](https://www.youtube.com/channel/UC34rW-HtPJulxr5wp2Xa04w)]
 
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

**Step 1-** Image Normalization. To manipulate brightness and contrast of the image datasets.
  
**Step 2-** Image Flattening.
  
**Step 3-** K-Means Clustering. Identify number of clusters accordingly to available distinct regions.
  
**Step 4-** Labeled Image Restructuring.
  
**Step 5-** Image Reshaping.
  
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




