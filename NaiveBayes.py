#!/usr/bin/env python 
#!-*- coding: utf-8 -*-

"""
Naive Bayes Classifier

Naive Bayes methods are a set of supervised learning algorithms 
based on applying Bayes' theorem with the "naive" assumption of 
independence between every pair of features. 
"""

import numpy as np

__author__ = "Irshad Ahmad Bhat"
__version__ = "1.0"
__email__ = "irshad.bhat@research.iiit.ac.in"

class multinomialNB():

    """
    Likelihood is computed by observing how often a feature value appears with 
    a given tag divided by the no. of times that tag appears in the training data
    """
	
    def fit(self, X, y):
	
	"""
	Compute various parameters for Naive Bayes binary calssifier 
	like how many times each feature value occurs with each tag to
	compute priors, prior counts and probabilities of labels and 
	total count of unique feature values (used in smoothing) etc.
	"""

	self.training_data = np.asarray(X)
        self.training_labels = np.asarray(y)

	unique_labels = np.unique(self.training_labels)
	unique_feats = np.unique(self.training_data)   # array of unique feature values in training-data
	label_count = dict()	# tag-count dictionary -- number of times each label occurs in the training-data

	# initialize parameters
	self.feats_count = len(unique_feats)
	self.feat_tag_cmat = np.zeros((len(unique_labels), self.feats_count))
	self.tag_id = {tag:i for i,tag in enumerate(unique_labels)}
	self.feat_id = {feat:i for i,feat in enumerate(unique_feats)}

	# populate feature-tag count matrix 
	for vec,lbl in zip(self.training_data, self.training_labels):
	    label_count.setdefault(lbl,0)
	    label_count[lbl] += 1
	    for x in vec:
		self.feat_tag_cmat[self.tag_id[lbl]][self.feat_id[x]] += 1

	# set prior probability and counts of labels
	self.prior_count = label_count
	self.prior_prob = {tag:np.log(label_count[tag]/float(len(self.training_labels))) \
			    for tag in unique_labels}

    def laplacian(self, val, tag):
	
	"""Returns conditional-probability (likelihod) with Laplacian smoothing"""
	if val in self.feat_id:
	    return (self.feat_tag_cmat[self.tag_id[tag]][self.feat_id[val]] + 1.0) / \
		(self.prior_count[tag] + self.feats_count) 
	else:
	    return 1.0 / (self.prior_count[tag] + self.feats_count) 
	
    def predict(self, testing_data):

	"""Returns a list of predicted labels using Naive-Bayes binary classifier"""
	labels = []
	testing_data = np.asarray(testing_data)

	if len(testing_data.shape) == 1 or testing_data.shape[1] == 1:
	    testing_data = testing_data.reshape(1,len(testing_data))

	for i,vec in enumerate(testing_data):
	    # initialize smoothed log probabilities for each tag
	    smoothed_lp = {tag:0.0 for tag in self.tag_id}	
	    for val in vec:
		for tag in self.tag_id:
		    # compute smoothed conditional probability
		    sl_prob = self.laplacian(val,tag)    
		    smoothed_lp[tag] += np.log(sl_prob)  
	    # Multiply priors
	    for tag in self.tag_id:
		smoothed_lp[tag] += self.prior_prob[tag]
	    labels.append(max(smoothed_lp.items(), key=lambda x:x[1])[0])

	return labels
    

class gaussianNB():

    """Likelihood is computed using Gaussian distribution formula"""

    def __init__(self): #, X, y):

	self.prior_prob = dict()    # prior probability of labels 
	self.variance = dict()	    # variances of data of different classes
	self.mean = dict()	    # means of data of different classes
    
    def get_mean(self, data):

	"""Return population mean"""	
	return np.sum(data, axis=0) / len(data)

    def get_variance(self, data, mean):
    
	"""Return population variance"""
	return np.sum((data-mean)**2, axis=0) / len(data)

    def fit(self, X, y):
	
	"""Compute mean and variance for each feature-column in data of different tags"""
	self.training_data = np.asarray(X)
        self.training_labels = np.asarray(y)
	self.unique_labels = np.unique(self.training_labels)

	# compute mean and variance of each feature column
	dim = len(self.training_data)	
	for lbl in self.unique_labels:
	    data = self.training_data[self.training_labels == lbl]
	    self.prior_prob[lbl] = np.log(len(data)/float(dim))
	    self.mean[lbl] = self.get_mean(data)
	    self.variance[lbl] = self.get_variance(data, self.mean[lbl])

    def gaussian(self, val, tag, i):

        """Returns conditional-probability (likelihod) using Gaussian distribution"""
        return (1/np.sqrt(2*np.pi*self.variance[tag][i])) * \
		np.exp(-0.5 * (val - self.mean[tag][i])**2 / self.variance[tag][i])

    def predict(self, testing_data):
	
	"""
	Returns a list of predicted labels using Naive-Bayes binary classifier where 
	conditional probability (likelihood) of each feature-value is computed using
	Gaussian Distribution formula.
	"""

	labels = list()
	testing_data = np.asarray(testing_data)

	if len(testing_data.shape) == 1 or testing_data.shape[1] == 1:
	    testing_data = testing_data.reshape(1,len(testing_data))

	for i,vec in enumerate(testing_data):
	    # initialize gaussian log probabilities for each tag
	    gaussian_lp = {tag:0.0 for tag in self.unique_labels}
	    for j,val in enumerate(vec):
		for tag in self.unique_labels:
		    # compute conditional probability
		    gs_prob = self.gaussian(val, tag, j)
		    if gs_prob:	    # filter zero probabilities
			gaussian_lp[tag] += np.log(gs_prob) 
	    # multiply priors
	    for tag in self.unique_labels:
		gaussian_lp[tag] += self.prior_prob[tag]
	    labels.append(max(gaussian_lp.items(), key=lambda x:x[1])[0])

	return labels	
