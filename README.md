a) Develop code for training and testing an SVM classifier with nonlinear kernel. You are welcome to use either formulation described in the textbook (chapter 7). You cannot use an SVM library to complete this assignment. You can use quadratic programming library if you like. Using your implementation of the SVM classifier, compare multi-class classification performance of two different voting schemes:
i. “one versus the rest” ii. “one versus one”
Be sure to specify your voting scheme using a method described in the book . To analyze accuracy, you will find it helpful to produce and analyze the multiclass confusion matrix in addition to examining the overall error rate.  

(b) (25 points) Use the same “one versus one” classifiers from the previous problem in a DAGSVM approach. A paper describing the approach, DAGSVM.pdf, is attached. Compare multi- class classification performance with the other two voting schemes.

(c) (c) (10 points) A baseline implementation of the DAGSVM with 6th degree polyno- mial kernels achieves 95% accuracy on the test set. See if you can do better than this baseline, using the DAGSVM approach. baseline-CM.pdf contains the confusion matrix of the baseline implementation:
