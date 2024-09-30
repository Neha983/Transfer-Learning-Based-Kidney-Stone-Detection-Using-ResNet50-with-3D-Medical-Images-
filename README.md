# Transfer-Learning-Based-Kidney-Stone-Detection-Using-ResNet50-with-3D-Medical-Images 

The kidneys are essential for human health because they maintain vital functions such as blood filtration and 
electrolyte balance. Detecting kidney stones is a significant medical imperative, requiring precise diagnosis to 
ensure timely intervention. Healthcare personnel must manually review medical photographs as part of
traditional diagnostic procedures, which can be laborious and prone to human error. A novel method for the
automated detection of kidney stones based on the analysis of coronal computed tomography (CT) images has 
been proposed. This method leverages recent advancements in artificial intelligence to achieve accurate 
identification of the occurrence or non-occurrence of kidney stones. In our proposed system, we address the 
task of kidney stone detection through transfer learning (TL) with ResNet50 using medical images. The model 
demonstrate to address limited medical data challenges, superior accuracy and sensitivity in detecting kidney 
stones compared to traditional methods. Moreover, it exhibit robust ability to generalize, even in the presence 
of variations and noise in medical images.
Keywords: Kidney Stone, Resnet50, Deep Learning, Transfer Learning.

# Abstract 
TL is knowledge transfer from one block to another block in order to improve the efficiency of the learning model. In kidney stone classification, there is a huge need for an effective classifier in order to differentiate structural and intensity similarities between pixels. These difficulties can be solved by using TL models. Further, the performance of cross residual network(ResNet-50) models is mainly based on the hyper parameters in convolutional layers. These parameters values are adjusted correctly to get improved performance. In this work, the pre-trained cross-residual network models of DenseNet169, MobileNetv2 and Google Net are combined by ensemble method to classify the kidney stones.
 The proposed approach is divided into four stages: Preprocessing stage, training and optimization stage, classification stage and performance analysis stage as shown in Fig. 1.
 ![image](https://github.com/user-attachments/assets/99e82adf-d9db-4381-a06b-9f6653b0b4c8)

The preprocessing stage involves data collection and argumentation processes like flipping around the x-axis, right/left mirroring, salt noise creation and image rotation etc. The TL model hyper parameters are tuned for optimization. By performing the ensemble method, theclassification results are combined to classify the stone types. Finally, the performance of the proposed model is analyzed for various parameters.

# ResNet50 Model

ResNet50 is a deep convolutional neural network (CNN) architecture that was developed by Microsoft Research in 2015. It is a variant of the popular ResNet architecture, which stands for “Residual Network.” The “50” in the name refers to the number of layers in the network, which is 50 layers deep.

ResNet50 is a powerful image classification model that can be trained on large datasets and achieve state-of-the-art results. One of its key innovations is the use of residual connections, which allow the network to learn a set of residual functions that map the input to the desired output. These residual connections enable the network to learn much deeper architectures than was previously possible, without suffering from the problem of vanishing gradients.

The architecture of ResNet50 is divided into four main parts: the convolutional layers, the identity block, the convolutional block, and the fully connected layers. The convolutional layers are responsible for extracting features from the input image, while the identity block and convolutional block are responsible for processing and transforming these features. Finally, the fully connected layers are used to make the final classification.

The convolutional layers in ResNet50 consist of several convolutional layers followed by batch normalization and ReLU activation. These layers are responsible for extracting features from the input image, such as edges, textures, and shapes. The convolutional layers are followed by max pooling layers, which reduce the spatial dimensions of the feature maps while preserving the most important features.

The identity block and convolutional block are the key building blocks of ResNet50. The identity block is a simple block that passes the input through a series of convolutional layers and adds the input back to the output. This allows the network to learn residual functions that map the input to the desired output. The convolutional block is similar to the identity block, but with the addition of a 1x1 convolutional layer that is used to reduce the number of filters before the 3x3 convolutional layer.


The final part of ResNet50 is the fully connected layers. These layers are responsible for making the final classification. The output of the final fully connected layer is fed into a softmax activation function to produce the final class probabilities.

![image](https://github.com/user-attachments/assets/b537b80f-5f7d-49a0-9ded-420be81dcc28)
![image](https://github.com/user-attachments/assets/12844896-c5d4-461d-9f6f-2e4db136d81b)

# Coding Language
# Python: 
# Version: 3.7.0(Supports for this project)
Python is a general-purpose language, which means it’s designed to be used in a range of applications, including data science, software and web development, automation, and generally getting stuff done.
# How to Install Python on Windows and Mac :
Go to the official site to download and install python using Google Chrome or any other web browser. OR Click on the following link: https://www.python.org


# Modules Used in Project  :-
# Tensorflow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google. 
TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache 2.0 open-source license on November 9, 2015.
# Numpy
Numpy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays.
It is the fundamental package for scientific computing with Python. It contains various features including these important ones:
	A powerful N-dimensional array object
	Sophisticated (broadcasting) functions
	Tools for integrating C/C++ and Fortran code
	Useful linear algebra, Fourier transform, and random number capabilities
Besides its obvious scientific uses, Numpy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined using Numpy which allows Numpy to seamlessly and speedily integrate with a wide variety of databases.
# Pandas
Pandas is an open-source Python Library providing high-performance data manipulation and analysis tool using its powerful data structures. Python was majorly used for data munging and preparation. It had very little contribution towards data analysis. Pandas solved this problem. Using Pandas, we can accomplish five typical steps in the processing and analysis of data, regardless of the origin of data load, prepare, manipulate, model, and analyze. Python with Pandas is used in a wide range of fields including academic and commercial domains including finance, economics, Statistics, analytics, etc.
# Matplotlib
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter Notebook, web application servers, and four graphical user interface toolkits.Matplotlib tries to make easy things easy and hard things possible. You can generate plots, histograms, power spectra, bar charts, error charts, scatter plots, etc., with just a few lines of code. For examples, see the sample plots and thumbnail gallery.
For simple plotting the pyplot module provides a MATLAB-like interface, particularly when combined with IPython. For the power user, you have full control of line styles, font properties, axes properties, etc, via an object oriented interface or via a set of functions familiar to MATLAB users.
# Scikit – learn
Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python. It is licensed under a permissive simplified BSD license and is distributed under many Linux distributions, encouraging academic and commercial use. 
# Python
Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code readability, notably using significant whitespace. 
Python features a dynamic type system and automatic memory management. It supports multiple programming paradigms, including object-oriented, imperative, functional and procedural, and has a large and comprehensive standard library. 

# Modules Needed to Execute the code in CMD
pip install numpy==1.18.1

pip install matplotlib==3.1.3 

pip install pandas==0.25.3 

pip install opencv-python==4.2.0.32

pip install keras==2.3.1 

pip install tensorflow==1.14.0 

pip install h5py==2.10.0 

pip install protobuf==3.16.0

pip install pillow==7.0.0

pip install sklearn-genetic==0.2

pip install SwarmPackagePy

pip install sklearn

pip install scikit-learn==0.22.2.post1

Pip install sklearn-extensions==0.0.2

Pip install pyswarms==1.1.0

pip install nltk

pip install pyaes

pip install pbkdf2

Pip install opencv-contrib-python==4.3.0.36

pip install django==2.1.7

pip install pymysql==0.9.3

pip uninstall crypto

pip uninstall pycryptodome

pip install pycryptodome

pip install pycryptodome==3.10.1

pip install   cryptography==2.9.2

# Note 
If you face any trouble while implementing those modules try adding these lines before every module
python -m pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --upgrade pip

# Base Papers
* https://ieeexplore.ieee.org/abstract/document/9985723
* https://www.nature.com/articles/s41598-022-15634-4.pdf
*  https://www.taylorfrancis.com/chapters/edit/10.1201/9781032665535-47/transfer-learning-based-kidney-stone-detection-patients-using-resnet50-medical-images-47-naresh-jahnavi-reddy-prem-kumar-ch-nikhil-chandu


# REFERENCES
[1] K. K. Shung, “High frequency ultrasonic imaging,” Journal of Medical Ultrasound, vol. 17, no. 1, pp. 25–30, 
2009. 

[2] J. S. Jose, “An effificient diagnosis of kidney images using association rule,” International Journal of Computer 
Technology Electronic Engineering, vol. 12, no. 2, pp. 14–20, 2012. 

[3] C. Cortes and V. Vapnik, “Support-vector networks,” Journal of Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.

[4] K. He, X. Zhang, S. Ren and J. Sun, “Deep residual learning for image recognition,” in IEEE Conf. on Computer 
Vision and Pattern Recognition, Las Vegas, USA, pp. 770–778, 2016. 

[5] A. Soni and A. Rai, “Kidney stone recognition and extraction using directional emboss & SVM from computed 
tomography images,” in 2020 Third Int. Conf. on Multimedia Processing, Communication & Information 
Technology (MPCIT), Shivamogga, India, pp. 172–183, 2020. 

[6] K. Viswanath and R. Gunasundari, “Analysis and implementation of kidney stone detection by reaction diffusion 
level set segmentation using xilinx system generator on FPGA,” VLSI Design, vol. 5, no. 3, pp. 573–581, 2159.

[7] Q. Yuan, H. Zhang and T. Deng, “Role of artifificial intelligence in kidney disease,” International Journal of 
Medical Sciences, vol. 17, no. 7, pp. 970–984, 2009 . 

[8] A. Martinez, D. Trinh and J. Hubert, “Towards an automated classifification method for ureteroscopic kidney stone
