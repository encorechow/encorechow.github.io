---
layout: post
title: Solution for using cuDNN v5 in faster-RCNN
excerpt: "During the time that faster-RCNN was developing, cuDNN version is v4. The problem comes with installing faster-RCNN in cuDNN v5."
categories: [Caffe]
tags: [deep learning, machine learning, faster-rcnn]
comments: true
---

### What's the problem

Installing faster-RCNN under cuDNN v5 will pop out a bunch of errors regarding to cuDNN, which is the mismatching of cuDNN version.  

The caffe version of faster-RCNN uses cuDNN v4 by default in the time that faster-RCNN was developing.  

Following gives the solution:

```bash
cd caffe-fast-rcnn  
git remote add caffe https://github.com/BVLC/caffe.git  #Add original caffe remote
git fetch caffe  # fetch newest caffe
git merge -X theirs caffe/master  # merge caffe to faster-RCNN
```

There is actually one more thing need to do: comment out ```self_.attr("phase") = static_cast<int>(this->phase_);``` from   ```include/caffe/layers/python_layer.hpp``` after merging.

[#237](https://github.com/rbgirshick/py-faster-rcnn/issues/237)
