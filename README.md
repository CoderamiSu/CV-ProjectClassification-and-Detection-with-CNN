# CV-ProjectClassification-and-Detection-with-CNN

The goal of this project is to develop a digit detection and recognition system which takes in a single image and returns any sequence of digits visible in that image. For example, if the input image contains a home address 123 Main Street, the algorithm should return “123”. The sequences of numbers may have varying scales, orientations, and fonts, and may be arbitrarily positioned in a noisy image. This is acomplished by first using the Maximally Stable Extremal Regions (MSER) algorithm (Matas et al, 2002) to extract regions of interest (ROI). Convolutional Neural Network (CNN) is used subsequently to recognize the digits in the ROI. The CNN model is developed based on pretrained weights of VGG16. The Street View House Numbers (SVHN) Dataset (ufldl.stanford.edu/housenumbers/) is used for training.

tensorflow version is 2.5.0rc1. Please use the cv_proj.yml to set up the environment.
Run run.py to produce images in graded_images folder.
ModelTraining.py is the code used for training the CNN model.

## Reference
J. Matas, O. Chum, M.U., Pajdla, T.: Robust wide baseline stereo from maximally stable extremal regions. Proc. of British Machine Vision Conference (2002) 384-396