# Exam questions QBI
## Exam format
The QBI exam is oral, but you will be provided with pen and paper in case you need to draw figures to better explain your answers. The duration is 30 minutes. If you decided to make a project, we will use the project as a starting point for the exam and extend with questions that were not touched by the project. Below you find a collection of questions that may come during the exam.
## Lecture 1: Introduction to images and the topic
### General stuff
1.	Can you mention some problems (general not specific on methods) that arise with the analysis of real image data?
2.	Why do you want mathematical code to perform the analysis?
3.	What is the difference between qualitative and quantitative assessments?
### Images
1.	What is the smallest element of an image (2D/3D)
2.	Images are discrete, what kind of sampling is done to obtain a digital image. Sampling figure (images_sampling.pdf)
3.	You often see colored images of scalar information. 
  a.	Why are colors used?
  b.	What should you think about when you use a color map?
4.	What is a histogram?
### Image formation
1.	Can you describe the image formation process? Source (radiation, wave, etc), sample, sample response, detection system.
Image analysis approaches
1.	Can you describe the difference between reproducibility and repeatability? 

## Lecture 2: Image enhancement
### General stuff
1.	What factors from the image formation process affect image quality? (resolution, noise, contrast, inhomogeneous contrast, artifacts).
2.	Can you mention two common noise distributions that are used in imaging.
### Filtering
1.	Can you give a general description of a filter in image processing?
2.	Can you mention two basic characteristics of filters. How are these used.
3.	Which filter would you use to remove outliers in the image?
4.	How can you use a high-pass filter?
### Advanced filters
1.	Why would not like to use convolution filter for denoising?
2.	Can you mention some advanced denoising filters?
### Evaluate filter performance
1.	How do you verify the performance of a filter?
2.	Mention two metrics used to evaluate filter performance.

## Lecture 3: Data sets
### General stuff
1.	Why are well-known data sets important?
2.	What makes a good data set? (amount, diversity, labels)
3.	You need different data sets depending on your task, can you mention some tasks? (segmentation, regression, classification, detection)
4.	There are different ways to build a data set, which? (Collect and manually label, simulations, pre-segmentation)
5.	What are typical problems with data sets? (imbalance, few, homogeneous, biasing)
### Baseline algorithm
1.	What is a baseline algorithm, when do you need it?

## Lecture 4: Basic segmentation
### General stuff
1.	What is segmentation? Why do you do it?
2.	How can you perform a basic segmentation?
3.	What is a confusion matrix? What do you need to create it?
4.	What is precision and recall?  (tp/(tp+fp) and tp/(tp+fn))
### Multiphase segmentation
1.	How do you segment multiple classes?
2.	What problems may arise when doing so?
### Morphological image processing
3.	Which are the basic morphological operators? How do they work?
4.	Which operation would you use to remove bright/dark pixels? 
5.	What is the risk with using open and close to clean up an image?
## Lecture 5: Advanced segmentation
### Advanced segmentation
1.	Can you describe the basic principle of histogram-based segmentation?
2.	How can you improve segmentation performance? 
3.	Can you mention an unsupervised segmentation method (e.g. KMeans), how does it work?
### Supervised segmentation
1.	Describe K Nearest neighbors
2.	How can you get more stable decisions with KNN?
3.	How does tree-based segmentation work? 
4.	Briefly describe a how a convolutional neutral network works. (Convolution produce signals in the network, down and upsampling (pooling), output probabilties)

## Lecture 6: Advanced shape and texture analysis
### Component labelling
1.	Can you tell the difference between segmentation, labelling, and classification?
2.	Can you describe the principle of connected component labelling? When does it fail?
3.	What can you learn about an item once it is labelled? (position, area, shape, orientation)
## Lecture 7: Complex objects
1.	What is a distance map, how can it be used?
2.	Which kind structures are usually analyzed with skeletons?
3.	What can you learn about a structure using the skeleton? 
4.	Basic component labelling is often insufficient, what can you use instead, benefits, disadvantages? (Watershed)
## Lecture 8: Dynamic experiments
1.	Which information is relevant to ask for in a dynamic experiment? (Size, shape, velocity, position, etc.)
2.	What is tracking? How can you do it? (Nearest item, correlation)
3.	Dynamic image data is often noisy; how can you perform denoising?
4.	What is image registration and why is it needed?
5.	How can you compute local deformation information? (DIC/DVC)
## Lecture 9: Statistical analysis
1.	Why is a controlled experiment better than an observational experiment to show causation?
2.	How do you define sensitivity for system parameter tuning? Why is parameter tuning important?
3.	What is the p-value and how is it used? What do you prefer, great or small p-value?
4.	How can you promote repeatability?

### Software engineering
1.	What is a unit test? Why do you need it?
2.	What are some key ideas for scientific visualization? (clear message, is it necessary?)

## Lecture 10: Bimodal experiments
1.	Why use multimodal experiments?
2.	Can you mention some modality combinations?
3.	What is data fusion?
4.	Mention a/some methods for multivariate segmentation?

## Lecture 11: Scaling up
### General
1.	What is parallel computing and when would you like to use it?
2.	What is distributed computing?
3.	What resource problems can occur in parallel and distributed computing?
4.	What is a dead-lock? 
5.	What is queue computing.

