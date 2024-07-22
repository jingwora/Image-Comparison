# Image Comparison


## 1. Image Similarity
- MSE
- SSIM
- SIFT
- ORB
- Perceptual Hash
- Color Histograms
- HOG
- MobileNetV2
- Siamese Networks
- ViT
- CLIP
- GPT-4 Vision


| Categories        | Methods        | Description                                             | Pros                                                                 | Cons                                                                | Use Case                                          | Tools                    |
|----------------|-------------------|---------------------------------------------------------|---------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------|--------------------------|
| Pixel-based    | MSE               | Mean Squared Error measures the average squared difference between pixels of two images | Simple and fast                                                     | Sensitive to minor changes and noise                               | Image comparison in controlled environments      | OpenCV                   |
| Pixel-based    | SSIM              | Structural Similarity Index measures the perceived quality of digital images | Considers structural information                                    | Computationally intensive for large images                         | Quality assessment in image compression          | scikit-image             |
| Feature-based  | SIFT              | Scale-Invariant Feature Transform detects and describes local features in images | Scale and rotation invariant                                        | Patented, slower than other methods                                | Object recognition, image stitching              | OpenCV                   |
| Feature-based  | ORB               | Oriented FAST and Rotated BRIEF is a fast and efficient feature detector and descriptor | Fast and free from patents                                          | Less accurate than SIFT and SURF                                   | Real-time applications, augmented reality        | OpenCV                   |
| Hash-based     | Perceptual Hash   | Generates a hash value based on the visual content of an image, robust to minor changes | Efficient and robust to minor changes                               | May not be accurate for visually similar but different images      | Image deduplication, image retrieval             | ImageHash                |
| Histogram-Based| Color Histograms  | Represents the distribution of colors in an image, useful for color-based image retrieval | Simple and effective for color-based image retrieval                | Not robust to changes in lighting conditions                       | Image retrieval based on color similarity        | OpenCV                   |
| Histogram-Based| HOG               | Histogram of Oriented Gradients detects object shapes by counting occurrences of gradient orientation | Effective for detecting objects in images                           | High computational cost, sensitive to image rotation               | Object detection, particularly in surveillance   | scikit-image, OpenCV     |
| DL             | MobileNetV2       | A lightweight deep learning model designed for efficient image classification on mobile devices | Lightweight and fast, suitable for mobile devices                   | May lack accuracy for complex tasks                                | Real-time image classification on mobile devices | TensorFlow, Keras        |
| DL             | Siamese Networks  | Uses two identical neural networks to find similarities between images | Effective for finding similarities between images                   | Requires large amounts of training data                            | Face verification, signature verification        | PyTorch, TensorFlow      |
| DL             | ViT               | Vision Transformer uses transformer models for image classification tasks | High accuracy for image classification tasks                        | Requires a lot of computational power and data                     | Large-scale image classification tasks           | Hugging Face Transformers|
| LLM            | CLIP              | Connects images and text by learning visual and textual representations | Connects images and text effectively                                | May produce inaccurate results for nuanced queries                 | Multimodal tasks, image and text retrieval       | OpenAI, Hugging Face     |
| LLM            | GPT-4 Vision      | An advanced AI model that understands and generates images and text | Advanced vision-language understanding                              | High computational cost, limited availability                      | Complex vision-language tasks, AI research       | OpenAI                   |

### Experiments

<img src="https://github.com/jingwora/Image-Comparison/blob/main/data/car_diagram/11.brake_base.jpg?raw=true" width="250"/> <img src="https://github.com/jingwora/Image-Comparison/blob/main/data/car_diagram/12.brake_senser1.jpg?raw=true" width="250"/><img src="https://github.com/jingwora/Image-Comparison/blob/main/data/car_diagram/13.brake_senser2.jpg?raw=true" width="250"/><img src="https://github.com/jingwora/Image-Comparison/blob/main/data/car_diagram/14.brake_senser3.jpg?raw=true" width="250"/><img src="https://github.com/jingwora/Image-Comparison/blob/main/data/car_diagram/21.wheel_base.jpg?raw=true" width="250"/>

| image1              | image2               | Pixel-based | Pixel-based | Feature-based | Feature-based | Hash-based        | Histogram-Based | Histogram-Based | DL           | DL              | DL     | LLM  | LLM          | LLM           |
|---------------------|----------------------|-------------|-------------|---------------|---------------|-------------------|-----------------|-----------------|--------------|-----------------|--------|------|--------------|---------------|
|                     |                      | MSE         | SSIM        | SIFT          | ORB           | Perceptual Hash   | Color Histograms| HOG             | MobileNetV2  | Siamese Networks| ViT    | CLIP | GPT-4o Vision| GPT-4o Vision |
|                     |                      |      MSE       | similarity  | good_matches  | good_matches  | hash_difference   | similarity      | distance        | distance     | distance        | distance | distance | similarity_score | explanation  |
| 11.brake_base.jpg   | 11.brake_base.jpg    | 60358.35328 | 0.005899    | 949           | 0             | 0                 | 1               | 0               | 0            | 0               | 0      | 0    | 100          | The two images are exactly the same in terms o... |
| 11.brake_base.jpg   | 12.brake_senser1.jpg | 60359.21833 | 0.005857    | 946           | 0             | 2                 | 0.999999        | 2.35027         | 5.544902     | 0.036028        | 4.777678 | 1.06747 | 95          | The two images are almost identical, with the ... |
| 11.brake_base.jpg   | 13.brake_senser2.jpg | 60360.55165 | 0.005793    | 946           | 0             | 4                 | 0.999992        | 3.15887         | 7.307299     | 0.064102        | 5.038806 | 1.40149 | 90          | The two images are very similar in terms of st... |
| 11.brake_base.jpg   | 14.brake_senser3.jpg | 60361.97542 | 0.00573     | 942           | 0             | 4                 | 0.999982        | 3.8962          | 9.489249     | 0.073777        | 6.066045 | 1.501479 | 85          | The two images are very similar in terms of st... |
| 11.brake_base.jpg   | 21.wheel_base.jpg    | 60379.06965 | 0.004981    | 234           | 87            | 34                | 0.99963         | 9.09234         | 22.866598    | 0.186927        | 19.555229 | 6.161696 | 20          | The two images are both technical diagrams rel... |
| run time (sec.)     |                      | 4.29        | 8.36        | 5.61          | 0.8           | 0.1               | 0.11            | 0.04            | 3.33         | 2.16            | 16.22   | 19.63 | 21.36        |               |




## 2. Image-Change-Detection

- Pixel Difference
- Structural Similarity Index (SSIM)
- Binary Thresholding
- Contour Detection and Bounding Boxes
- Histogram Comparison
- Keypoint Detection (e.g., SIFT, ORB)
- Convolutional Neural Networks (CNNs)
- Vision Transformers (ViT)
- YOLOv9
- U-Net


| Categories       | Methods                                  | Description                                                                 | Pros                                                   | Cons                                                     | Use Case                                                      | Tools                    |
|---------------|----------------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------|---------------------------------------------------------------|--------------------------|
| Basic         | Pixel Difference                       | Compares pixel values directly to find differences.                         | Simple to implement.                                    | Sensitive to minor changes like noise or compression artifacts. | Quick comparison for similar images.                           | OpenCV                   |
| Structural    | Structural Similarity Index (SSIM)     | Measures structural similarity between images to find differences.          | More robust to minor changes, considers structural information. | Computationally intensive, can be slow on large images.         | Quality assessment, detecting subtle changes.                  | scikit-image, OpenCV     |
| Thresholding  | Binary Thresholding                    | Converts difference image to binary image to highlight significant changes. | Effective for highlighting large differences.           | Loses fine-grained details.                                  | Highlighting major differences, preprocessing for contour detection. | OpenCV                   |
| Contour       | Contour Detection and Bounding Boxes   | Finds contours in thresholded image and draws bounding boxes around them.   | Provides clear visual indication of differences.        | May miss small or subtle differences.                        | Highlighting differences for visual inspection.                | OpenCV                   |
| Histogram     | Histogram Comparison                   | Compares image histograms to measure overall similarity.                    | Fast, not affected by small spatial shifts.             | Doesn't provide spatial information about differences.        | Quick similarity check, global changes detection.               | OpenCV                   |
| Feature-based | Keypoint Detection (e.g., SIFT, ORB)   | Detects and matches keypoints between images to find differences.           | Robust to scaling, rotation, and partial occlusion.     | Computationally expensive, requires feature-rich images.       | Object recognition, tracking, comparing complex scenes.        | OpenCV                   |
| Deep Learning | Convolutional Neural Networks (CNNs)   | Uses CNNs to learn and detect differences between images.                   | Highly accurate, can handle complex differences.        | Requires large datasets and computational resources.          | Advanced applications like medical imaging, anomaly detection. | TensorFlow, PyTorch      |
| Transformer   | Vision Transformers (ViT)              | Uses self-attention mechanisms to process images and detect differences.    | Captures long-range dependencies, robust to various image conditions. | Computationally intensive, complex to implement.               | Advanced image recognition tasks, large-scale datasets.        | PyTorch, TensorFlow      |
| Real-time     | YOLOv9                                 | Uses a single convolutional neural network for real-time object detection.  | High speed and accuracy, suitable for various object sizes. | Requires large labeled datasets, sensitive to objective functions. | Real-time detection in autonomous vehicles, surveillance.       | Darknet, OpenCV          |
| Segmentation  | U-Net                                  | Uses an encoder-decoder architecture for precise segmentation and difference detection. | Highly accurate for pixel-level tasks, robust for various medical and satellite imagery. | Computationally expensive, requires large datasets and training time. | Medical image analysis, satellite image processing, fine-grained segmentation. | TensorFlow, PyTorch      |

- https://github.com/likyoo/open-cd
- https://github.com/Z-Zheng/ChangeStar
- https://github.com/aigzhusmart/AdaptFormer
- https://www.kaggle.com/code/aninda/change-detection-nb
- https://paperswithcode.com/paper/tinycd-a-not-so-deep-learning-model-for
- https://paperswithcode.com/task/change-detection
  
### Experiments

- absolute_differences
<img src="https://github.com/jingwora/Image-Comparison/blob/main/assets/absolute_differences.png?raw=true" width="2000"/>

- binary_thresholding_differences
<img src="https://github.com/jingwora/Image-Comparison/blob/main/assets/binary_thresholding_differences.png?raw=true" width="2000"/>

- contour_differences
<img src="https://github.com/jingwora/Image-Comparison/blob/main/assets/contour_differences.png?raw=true" width="2000"/>

- histogram_differences
<img src="https://github.com/jingwora/Image-Comparison/blob/main/assets/histogram_differences.png?raw=true" width="2000"/>

- sift_differences
<img src="https://github.com/jingwora/Image-Comparison/blob/main/assets/sift_differences.png?raw=true" width="2000"/>

- ssim_differences
<img src="https://github.com/jingwora/Image-Comparison/blob/main/assets/ssim_differences.png?raw=true" width="2000"/>

### Datasets:

- https://www.kaggle.com/datasets/amerii/spacenet-7-change-detection-chips-and-masks
- https://www.kaggle.com/datasets/kmader/aerial-change-detection-in-video-games
- https://www.kaggle.com/datasets/dingmama/remote-sensing-image-change-detection-dataset
- 
### Reference:

- [Object-Detection](https://jingwora.github.io/contents/articles/Object-Detection.html)

- Spot the difference: Detection of Topological Changes via Geometric Alignment
- https://github.com/SteffenCzolbe/TopologicalChangeDetection

- visualize-differences-between-two-images-with-opencv-python
- https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
