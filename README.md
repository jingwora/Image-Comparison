# Image Comparison


## 1. Image Similarity
| Segment        | Techniques Brief        | Pros                                                                 | Cons                                                                | Use Case                                          | Tools                    |
|----------------|-------------------------|---------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------|--------------------------|
| Pixel-based    | MSE                     | Simple and fast                                                     | Sensitive to minor changes and noise                               | Image comparison in controlled environments      | OpenCV                   |
| Pixel-based    | SSIM                    | Considers structural information                                    | Computationally intensive for large images                         | Quality assessment in image compression          | scikit-image             |
| Feature-based  | SIFT                    | Scale and rotation invariant                                        | Patented, slower than other methods                                | Object recognition, image stitching              | OpenCV                   |
| Feature-based  | ORB                     | Fast and free from patents                                          | Less accurate than SIFT and SURF                                   | Real-time applications, augmented reality        | OpenCV                   |
| Hash-based     | Perceptual Hash         | Efficient and robust to minor changes                               | May not be accurate for visually similar but different images      | Image deduplication, image retrieval             | ImageHash                |
| Histogram-Based| Color Histograms        | Simple and effective for color-based image retrieval                | Not robust to changes in lighting conditions                       | Image retrieval based on color similarity        | OpenCV                   |
| Histogram-Based| HOG                     | Effective for detecting objects in images                           | High computational cost, sensitive to image rotation               | Object detection, particularly in surveillance   | scikit-image, OpenCV     |
| DL             | MobileNetV2             | Lightweight and fast, suitable for mobile devices                   | May lack accuracy for complex tasks                                | Real-time image classification on mobile devices | TensorFlow, Keras        |
| DL             | Siamese Networks        | Effective for finding similarities between images                   | Requires large amounts of training data                            | Face verification, signature verification        | PyTorch, TensorFlow      |
| DL             | ViT                     | High accuracy for image classification tasks                        | Requires a lot of computational power and data                     | Large-scale image classification tasks           | Hugging Face Transformers|
| LLM            | CLIP                    | Connects images and text effectively                                | May produce inaccurate results for nuanced queries                 | Multimodal tasks, image and text retrieval       | OpenAI, Hugging Face     |
| LLM            | GPT-4 Vision            | Advanced vision-language understanding                              | High computational cost, limited availability                      | Complex vision-language tasks, AI research       | OpenAI                   |


## 2. Image-Change-Detection

- [Object-Detection](https://jingwora.github.io/contents/articles/Object-Detection.html)

### Reference:
- Spot the difference: Detection of Topological Changes via Geometric Alignment
- https://github.com/SteffenCzolbe/TopologicalChangeDetection

- visualize-differences-between-two-images-with-opencv-python
- https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
