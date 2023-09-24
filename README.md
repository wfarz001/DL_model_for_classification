# DL_model_for_classification
TASK:1:
Design a Convolutional Neural Network (CNN) to classify the benchmark handwritten digit dataset.





	
Training Accuracy	99.24%
Testing Accuracy	98.14%



TASK. 2:
Using the same design in Task 1 to classify the benchmark dataset: CIFAR10.

Training Accuracy	72.34 %
Testing Accuracy	63.99 %



TASK 3:
Transfer learning for CIFAR10. Download the pre-trained VGG16 model.

(a)
 
(b)
 

(c)
Data Size	Training Accuracy	Testing Accuracy
100	16%	5%
1K	11.2%	8.9%
10K	97.7%	67.5%
50K	91.59%	79.44%



TASK:4:
Transfer learning for a real medical dataset.

(a)
Resizing the images with the following code:
 


(b)
Performance with TASK:1 model architecture:
Sensitivity	0.83
Specificity	0.62
ROC Score	0.725

 

(c)
Features are created using VGG16 model feature extractor and the training and testing features are stored in csv files and provided with the code in the zip folder.
 

 


(d)  VGG16 features with SVM Classifier
Sensitivity	0.60
Specificity	0.86
ROC Score	0.893

 

(e)  VGG16 Model
Batch Size: 2
Epoch: 50
Optimizer: Adam
Metrics: Accuracy
Sensitivity	0.02
Specificity	1.00      
ROC Score	0.511

 

TASK:5:
From Task-1 and Task-2 it is observed that, with increment of training data, the model performance improves. For instance, in Task-1, the training data size is 60,000 and in TASK-2 training data size is 50,000. Therefore, with the same model, the training and testing accuracy is TASK-1 is better than TASK-2.

In Task-3, it is observed, with the increment of training data, the testing accuracy improves and testing accuracy is the highest with 50K training data.

In Task-4, in terms of ROC score, the SVM classifier performs the best with extracted features with VGG16 model. It might be concluded that with small dataset, the machine learning model performs better than deep learning model because in case of deep learning model a large amount of data is required to train the model. In this case, the training part consists of (42+42) =84 images whereas the testing data is consisting of (162+42) =204 cases, therefore, there is class imbalance in testing data which indicates real scenario. Therefore, in terms of medical data, where there is a possibility of different data distribution in testing data, machine learning model might perform better than deep learning model. However, if the goal is prioritizing the sensitivity, correctly classifying cancer images here, the performance of CNN model is better than machine learning model. Therefore, a trade-off needs to consider while deciding upon machine learning model and deep learning model.
Reference: 

Examples for MNIST dataset:

Matlab example can be found here:
https://www.mathworks.com/help/deeplearning/examples/create-simple-deep-learning-network-for-classification.html
Keras example can be found here:
https://keras.io/examples/mnist_cnn/
Pytorch example can be found here:
https://nextjournal.com/gkoehler/pytorch-mnist
