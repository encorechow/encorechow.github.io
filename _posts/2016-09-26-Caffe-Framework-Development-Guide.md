---
layout: post
title: Caffe Framework Development Guide
excerpt: "This post intends to give some insights for developers who are doing their research or work on Caffe. All of contents are only based on my experience of developing new layer in Caffe. I can't say it is going to one hundred percent work for everyone, but it can be a helpful reference to check out the bugs and issues that you may face."
categories: [Caffe]
comments: true
image:
  feature: https://hd.unsplash.com/photo-1422207134147-65fb81f59e38
  credit: Padurariu Alexandru
  creditlink: https://unsplash.com/collections/206470/caffe?photo=k0SwnevO_wk
---
*A CAFFE A DAY, KEEPS THE GRUMPY AWAY.*

# Introduction

Since deep learning became extremely prevalent today, more and more developers and researchers are digging into different deep learning framework. Comparing with old-school machine learning algorithm, deep learning is definitely a more fancy way to present the machines' perspective of the world. So it is worth learning some of the deep learning frameworks and get your feet wet on deep learning. Here is what I know currently the most popular deep learning frameworks: **_Theano, Tensorflow, Keras, Caffe, Torch._** All of above frameworks support GPU computing and CUDA cuDNN. This post will mainly focus on the Caffe source code and how to work on your own deep learning layers.
Please read this [Caffe Development](https://github.com/BVLC/caffe/wiki/Development) first before you want to get something more advanced in my experience.


# Basic Concept in Caffe
