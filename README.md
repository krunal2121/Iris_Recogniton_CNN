<h1>Project Report: Iris Recognition using Convolutional Neural Networks</h1>

<h2>Introduction</h2>

The goal of this project is to build a convolutional neural network (CNN) for iris recognition. The CNN takes iris images as input and predicts the identity of the person whose iris is captured in the image. Iris recognition is a popular biometric technology that is used for security and identification purposes.

The dataset used in this project is the CASIA Iris Lamp dataset, which consists of iris images from 411 individuals. The dataset is split into left and right iris images, resulting in a total of 822 classes.

<h2>Methodology</h2>

<h3>Data Preprocessing</h3>

The first step in building the CNN is to preprocess the data. The images in the dataset are resized to 30x30 pixels and converted to grayscale. The images are then converted to numpy arrays and stored in the data variable. The corresponding labels for each image are stored in the labels variable.

The train_test_split function from the sklearn library is used to split the data into training and testing sets. The training set contains 80% of the data and the testing set contains the remaining 20%.

The labels are one-hot encoded using the to_categorical function from the keras.utils.np_utils library. This is done to convert the labels from integer format to binary format, which is required for training the CNN.

<h3>Model Architecture</h3>
The CNN architecture used in this project consists of several layers:

Conv2D layer with 32 filters and a kernel size of 5x5, with a ReLU activation function
Conv2D layer with 32 filters and a kernel size of 5x5, with a ReLU activation function
MaxPool2D layer with a pool size of 2x2
Dropout layer with a rate of 0.25
Conv2D layer with 64 filters and a kernel size of 3x3, with a ReLU activation function
Conv2D layer with 64 filters and a kernel size of 3x3, with a ReLU activation function
MaxPool2D layer with a pool size of 2x2
Dropout layer with a rate of 0.25
Flatten layer
Dense layer with 256 units and a ReLU activation function
Dropout layer with a rate of 0.5
Dense layer with 412 units and a softmax activation function
The CNN is compiled using the compile function from the keras.models library. The loss function used is categorical crossentropy and the optimizer used is Adam. The performance metric used is accuracy.

<h3>Model Training</h3>
The CNN is trained using the fit function from the keras.models library. The training data is passed to the fit function along with the number of epochs (in this case, 10). The validation data is also passed to the fit function to evaluate the performance of the model on the testing set.

The trained model is saved to a file called "my_model1.h5" using the save function from the keras.models library.

<h3>Model Evaluation</h3>
The trained model is used to make predictions on the testing set using the predict function from the keras.models library. The predicted classes are obtained using the argmax function from the numpy library.

The accuracy of the model on the testing set is calculated using the evaluate function from the keras.models library.
