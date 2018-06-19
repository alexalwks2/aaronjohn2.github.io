---
layout: post
comments: true
title:  "Flower Recognition - Using CNN  on Mobile Device"
date:   2018-18-06 19:30:27
categories: deep-learning, swift, CoreML-tools, python, Caffe-Model, CNN, MLModel, IOS, mobile, AI
---

In this project I converted a pre-trained Caffe Model into MLModel format which was later used in my Xcode project.
The Caffe Model that I used had already been trained on thousands upon thousands of images. This pre-trained model was converted to a .MLModel so that it can be seen and used by my swift files in order to build this app.

Apple has released open source tools in python that allow you to convert any pre-trained model that had been trained using Caffe, Keras, scikit learn, and the list goes on.

Brief Description of app:

* This app allows you to take a picture of any flower and it will try to recognize what that flower is called.

* The data set that our model has been trained on is called "Oxford 102 Flower Dataset."
A bunch of researchers from Oxford did a very time consuming task by labeling a whole bunch of images of flowers with the correct classification of the flower.

* The reason why it is called Oxford 102, is because they have 102 categories for the flowers.
In their dataset they've got anywhere between 40 to 200 images for each and every category.
These images could vary and have been taken at different angles and at different stages of life cycle of the flower.

* Therefore if you train a machine learning model on it, it should be able to classify with relative accuracy as to what flower is in the image that you have taken.

* A man by the name Jimmie Goode(https://github.com/jimgoo/caffe-oxford102) used this dataset to train a Convolutional Neural Network(CNN). He worked on the training of CNN's with Caffe to classify images in the Oxford 102 category flower dataset. This Caffe model is what I used in this project to build the app.

You can get my [Flower/Plant recognition iOS app](//this is a comment.. leave app link in here, once it is published) in the App Store.
It's pretty cool to see Caffe finally working on my iPhone to solve a real world problem, even without Internet connection (This is due to having an MLModel pre-loaded onto the device and it does not require an API.)