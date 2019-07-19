Question 1.
Develop code for training and testing an SVM classifier with nonlinear kernel. You are welcome to use either formulation described in the textbook (chapter 7). You cannot use an SVM library to complete this assignment. You can use quadratic programming library if you like. Using your implementation of the SVM classifier, compare multi-class classification performance of two different voting schemes:
i.	“one versus the rest” 
ii.	“one versus one” 
Be sure to specify your voting scheme using a method described in the book. To analyze accuracy, you will find it helpful to produce and analyze the multiclass confusion matrix in addition to examining the overall error rate.

Solution

I have implemented the code for SVM in python.  It will take care of finding support vectors and margins of the supplied points. I have used polynomial kernel with degree 2 and 6. Also, I have tried using Gaussian kernel with sigma 0.5.
