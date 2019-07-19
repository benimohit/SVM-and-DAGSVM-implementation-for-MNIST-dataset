#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:22:59 2019

@author: mohitbeniwal
"""
from collections import Counter
from scipy.io import loadmat
import numpy as np # linear algebra
from numpy import linalg
import cvxopt
from sklearn.metrics import confusion_matrix

#polynomial kernel with degree 2 by default
def polynomial_kernel(x, y, d=2):
    return (1 + np.dot(x, y)) ** d

# ploynomial kernel with degree 6 by default
def polynomial_kernel_dsix(x, y, d=6):
    return (1 + np.dot(x, y)) ** d

# gaussian kernel with sigma 0.5 by default
def gaussian_kernel(x, y, sigma=0.5):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

# SVM_classifier implementation
class SVM_classifier(object):
    
    def __init__(self, kernel=polynomial_kernel):
        self.kernel = kernel
        
    def createKernal(self, X):
        m = X.shape[0]
        # Gram matrix
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i,j] = self.kernel(X[i], X[j])     
        return K

    def fit(self, K, X, y):
        m = X.shape[0]
        # Solving  min 1/2 x^T P x + q^T x
        G = cvxopt.matrix(-np.eye(m))
        A = cvxopt.matrix(y, (1,m))
        
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(-np.ones((m,1)))
        b = cvxopt.matrix(0.0)
        h = cvxopt.matrix(np.zeros(m))
        
        cvxopt.solvers.options['show_progress'] = False
       
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alphas = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.a = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]        
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        # Weight vector
        self.w = None 
        
    def predict(self, X,scheme_vote):
        w = np.zeros(len(X)) # wx
        for i in range(len(X)):
            summ = 0
            # alphas, support vectors, sv labels
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                summ += a * sv_y * self.kernel(X[i], sv)
            w[i] = summ
        # 1 represents 'one vs one' and 2 represents 'one vs rest'
        out =w+self.b
        if scheme_vote == 1:
           return np.sign(out)
        else:
           return (out)
        
def get_one_vs_one_models(data, kernel_fun):
    models = [[SVM_classifier(kernel_fun) for j in range(10)] for i in range(10)]
    
    for i in range(10):
        for j in range(10):
            if i != j and i<j:
                X_train = data.get('train_samples')
                y_train = data.get('train_samples_labels')
                y_train = y_train.ravel()
                X_train = X_train[(y_train==i)|(y_train==j)]
                y_train = y_train[(y_train==i)|(y_train==j)]
                #label each digit 1 and  -1
                y_train = 1.0*(y_train==i)-1.0*(y_train==j)   
                
                ovo = SVM_classifier(kernel_fun)
                kernal = ovo.createKernal(X_train)
                ovo.fit(kernal,X_train, y_train)
                models[i][j] = ovo
                
    return(models)
                
              
def one_vs_one(data, svm_models):
    X_test = data.get('test_samples')
    y_test = data.get('test_samples_labels')
    y_pred = [[[None] for j in range(10)] for i in range(10)]
    y_test = y_test.ravel()
    
    for i in range(10):
        for j in range(10):
            if i != j and i<j:
                preds = svm_models[i][j].predict(X_test,1)
                pred_label = [i if x == 1 else j for x in preds] 
                y_pred[i][j] = pred_label
                
 
    predicted_labels = []*len(y_test)
    
    for k in range(len(y_test)):
        label_list = []
        for i in range(10):
            for j in range(10):
                if i != j and i<j:
                    label_list.append(y_pred[i][j][k])
         # Returns the highest occurring item            
        label = Counter(label_list).most_common(1)[0][0] 
        predicted_labels.append(label)
    
    correct = np.sum(predicted_labels == y_test)
    print(str(correct)+" out of "+str(len(predicted_labels))+ " predictions are correct") 
    model_acc = correct/len(predicted_labels)
    conf_mat = confusion_matrix(y_test,predicted_labels)
    print('\nSVM Trained Classifier Accuracy for One VS One: ', model_acc)
    print('\nConfusion Matrix for One VS One: \n',conf_mat)
    return(predicted_labels)
    

def dag_one_vs_one(data, svm):
    
    X_test = data.get('test_samples')
    y_test = data.get('test_samples_labels')
    y_test = y_test.ravel()
                
    predicted_labels = []*len(y_test)
                
    for i in range(len(X_test)):
        nums = np.arange(0,10)
        while len(nums) > 1:
            one = nums[0]
            last = nums[len(nums) - 1]
            tt = svm[one][last].predict([X_test[i]],1)
            
            if tt > 0:
                nums = nums[0:len(nums)-1]
            else:
                nums = nums[1:len(nums)]
        predicted_labels.append(nums[0])
    
    correct = np.sum(predicted_labels == y_test)
    #print(predicted_labels)
    #print(y_test)
    print(str(correct)+" out of "+str(len(predicted_labels))+ " predictions are correct") 
    model_acc = correct/len(predicted_labels)
    conf_mat = confusion_matrix(y_test,predicted_labels)
    print('\nDAG SVM Trained Classifier Accuracy for One VS One: ', model_acc)
    print('\nConfusion Matrix Dag SVM for One VS One: \n',conf_mat)
    return(predicted_labels)

def get_one_vs_rest_models(data, kernel_fun): 
    predict = [SVM_classifier(kernel_fun) for j in range(10)]
    X_train = data.get('train_samples')
    ovr = SVM_classifier(kernel_fun)
    K = ovr.createKernal(X_train)
    for i in range(10):
        y_train = data.get('train_samples_labels')
        y_train = y_train.ravel()
         #label each digit 1 and  -1
        y_train = 1.0*(y_train==i)-1.0*(y_train!=i)  
        ovr = SVM_classifier(kernel_fun)
        ovr.fit(K,X_train, y_train)
        predict[i] = ovr
        
    return(predict)
    
def one_vs_rest(data, svm):
    X_test = data.get('test_samples')
    y_test = data.get('test_samples_labels')
    y_test = y_test.ravel()
    y_predict = [[None]* len(y_test) ]*10
    for j in range(10):
        y_predict[j] = svm[j].predict(X_test,2)
    
    predicted_labels = []*len(y_test)
    
    for j in range(len(y_test)): 
        listC = []
        for i in range(10):
            listC.append(y_predict[i][j])
        predicted_labels.append(np.argmax(listC))
        

    correct = np.sum(predicted_labels == y_test)
    print(str(correct)+" out of "+str(len(predicted_labels))+ " predictions are correct") 
    model_acc = correct/len(predicted_labels)
    conf_mat = confusion_matrix(y_test,predicted_labels)
    print('\nSVM Trained Classifier Accuracy for One VS rest: ', model_acc)
    print('\nConfusion Matrix for One VS rest: \n',conf_mat)
    return(predicted_labels)
             

               
if __name__ == "__main__":
    
    data = loadmat('MNIST_data.mat')
    kernel_fun=polynomial_kernel
    #kernel_fun=polynomial_kernel_dsix
    #kernel_fun=gaussian_kernel
    
    # get one vs one models
    one_vs_one_models= get_one_vs_one_models(data,kernel_fun)
    # one vs one prediction
    one_vs_one(data, one_vs_one_models)
    # get models for one vs rest
    one_vs_rest_models = get_one_vs_rest_models(data,kernel_fun)
    # predict for one vs rest
    one_vs_rest(data, one_vs_rest_models)
    
    # dag svm one vs one prediction (uses same modles as one vs one svm)
    dag_one_vs_one(data, one_vs_one_models)
    
