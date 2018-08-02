# Face-Verification
Repository for Face Detection and Verification Systems

***
## Tasks
+ [x] Face Detection
    - [x] Haar Cascade Classifier
    - [x] LBP Classifier (not actually implemented)
+ [ ] Face Alignment
    - [ ] TBD
+ [x] Face Verification
    - [x] Base CNN model building and training
    - [x] Face verification metric measure
+ [ ] Liveness Verification
    - [ ] Eye-blink
    - [ ] TBD
+ [ ] Interface or API
    - [ ] TBD

***
## Demo

#### 1. Face Detection --> [OpenCV Haar Feature-based Cascade Classifiers](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)
* Sample Input Images <br/>
<img src="./results/input_images.png" alt="Sample" style="width: 600px;"/>

* Frontal Face Detection <br/>
<img src="./results/face_detection.png" alt="Sample" style="width: 600px;"/>

* Frontal Face Crop <br/>
<img src="./results/cropped_faces.png" alt="Sample" style="width: 600px;"/>

* Check [Jupyter Notebook](https://github.com/JifuZhao/face-verification/blob/master/5.%20FaceNet%20Application%20Demo.ipynb) for details.

#### 2. Base CNN Model for Face Verification
+ Training Dataset: [VGGFace2 Dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
+ Triplet Loss <br/>
![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cbg_white%20L%28a%2C%20p%2C%20n%29%20%3D%20max%20%5C%7B%20%7C%7Cf%28a%29-f%28p%29%7C%7C_2%5E2%20-%7C%7Cf%28a%29-f%28n%29%7C%7C_2%5E2%20&plus;%20%5Calpha%20%2C%200%20%5C%7D)
    * a: anchor image
    * p: positive image
    * n: negative image
    * f(x): CNN model to encode the input image
    * $\alpha$: margin for triplet <br/>
    <img src="./results/triplet_loss.png" alt="Sample" style="width: 500px;"/>


#### 3. Face Verification



#### 4. Performance


***
### Useful Links:
* [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
* [Deep face recognition with Keras, Dlib and OpenCV](https://krasserm.github.io/2018/02/07/deep-face-recognition/)
* [Face detection with OpenCV and deep learning](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
* [Face recognition using Tensorflow](https://github.com/davidsandberg/facenet)
* [The world's simplest facial recognition api for Python and the command line](https://github.com/ageitgey/face_recognition)


***
#### Note
Limited by computation resources, a relatively small CNN model is used, and the model is only trained for 1,000 epochs. For better performance, please refer to other pre-trained models.


Copyright @ Jifu Zhao 2018
