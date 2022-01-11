# Cell Confluency Using Image Segmentation

## Project Details
Currently employed methods:  
1- U-Net CNN (Supervised Learning)  
2- K-Means Clustering (Unsupervised Learning)  
3- Tiny-YOLO (Supervised Learning, Object Detection)

Language: Python  

Platform IDEs:  
1- Anaconda (Environment Setup) [[Installation](https://www.anaconda.com/products/individual#Downloads)]  
2- Pycharm (Script development) [[Installation](https://www.jetbrains.com/pycharm/download/)]  
3- Spyder (Data viewer & visualization) [[Installation](https://www.spyder-ide.org/)] [[Anaconda Envs Preferences](https://medium.com/@apremgeorge/using-conda-python-environments-with-spyder-ide-and-jupyter-notebooks-in-windows-4e0a905aaac5)]  [[Compatibility](https://docs.spyder-ide.org/5/troubleshooting/common-illnesses.html)]  
4- Google Colaboratory  

Requirements and Libraries:  
1- Data Annotation Tools for Method 1; Suggested: [[LabelMe](https://github.com/wkentaro/labelme)] [[Apeer](https://www.apeer.com/app)]; Discoverable: [[Tools](https://github.com/taivop/awesome-data-annotation)]  
2- Data Annotation Tool for Method 3-YOLO [[OpenLabelling](https://github.com/Cartucho/OpenLabeling)]  
3- Tensorflow Installation Setup [[Reference](https://www.youtube.com/watch?v=hHWkvEcDBO0)]  
4- Libraries: opencv-python, matplotlib, numpy, keras, pillow, sklearn, imutils  

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
 
**Step 1-** Data annotation. Cells are labelled to distinguish between the background and the cell confluency. [[Reference](https://medium.com/ching-i/segmentation-label-%E6%A8%99%E8%A8%BB%E6%95%99%E5%AD%B8-26b8179d661)]  
  
**Step 2-** (Optional) Data/ Image patching. If your data resolutions/ dimensions are high, this step is recommended for GPU support. [[Reference 1](https://github.com/Fyzie/Cell-Confluency/blob/main/U-Net%20CNN/patch_images_masks_for_large_image.py)] [[Reference 2](https://github.com/bnsreenu/python_for_microscopists/blob/master/Tips_Tricks_5_extracting_patches_from_large_images_and_masks_for_semantic_segm.py)]
  
**Step 3-** U-Net model development. [[Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/204-207simple_unet_model.py)]  
  
**Step 4-** Data Training. [[Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/204_train_simple_unet_for_mitochondria.py)]
  
**Step 5-** Model Inference and Prediction. [[Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/204_train_simple_unet_for_mitochondria.py)]
  
**Step 6-** Model Application. [[Reference 1](https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py)] [[Reference 2](https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_patchify.py)]

Youtube Reference: [[DigitalSreeni](https://www.youtube.com/channel/UC34rW-HtPJulxr5wp2Xa04w)] [[Apeer_micro](https://www.youtube.com/channel/UCVrG0AsRMb0pPcxzX75SusA/featured)]
 
</details>

<details open>
<summary>Results</summary>
<br>
  
Patched images:
 
   <img src="https://user-images.githubusercontent.com/76240694/148942361-26a31557-35c9-4d41-b70c-7202c2d2e017.png" width="600">
 
   <img src="https://user-images.githubusercontent.com/76240694/148942459-9b846d42-f7ec-4fab-b328-a6bef275516d.png" width="600">  
   
Full image:  
   <img src="https://user-images.githubusercontent.com/76240694/148942598-cff6f7c2-e11d-4dcb-9cb2-105599b33468.jpeg" width="600">

</details>

<summary>Arising Problems</summary>
<br>

**GPU exhaustion leads to the need for:**  
  (i) batch size reduction  
  (ii) dimensions reduction  
  <!--(iii) data training reduction-->
  
<!--**Improper model development leads to: ** 
  (i) error prediction and inference  -->

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
<summary>Results</summary>
<br>
  
  <img src="https://user-images.githubusercontent.com/76240694/138198216-0d487c0b-f31b-4b77-b9e2-0f8e385c927b.png" width="600">
 
</details>

<details open>
<summary>Arising Problems</summary>
<br>
  
**Improper clustering may due to:**  
  (i) irregular brightness of image datasets (trying to normalize them in Step 1) 
  (ii) random cluster number and cluster centers
  
</details>

### 3. Tiny-YOLO

## Research Paper References

1. [MC-Unet: Multi-scale Convolution Unet for Bladder Cancer Cell Segmentation in Phase-Contrast Microscopy Images](https://ieeexplore.ieee.org/document/8983121)

2. [An Engineered Approach to Stem Cell Culture: Automating the Decision Process for Real-Time Adaptive Subculture of Stem Cells](https://www.researchgate.net/publication/51824254_An_Engineered_Approach_to_Stem_Cell_Culture_Automating_the_Decision_Process_for_Real-Time_Adaptive_Subculture_of_Stem_Cells)

3. [CNNs for Segmenting Confluent Cellular Culture](http://cs229.stanford.edu/proj2016/report/BrunoBeltranNalinRatnayeke-CNNsForSegmentingConfluentCellularCulture-report.pdf)

4. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)


## Image Segmentation
Image segmentation is a process of partitioning images into multiple distinguished regions containing each pixel with similar features/ attributes.




