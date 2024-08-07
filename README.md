The code files contained within this repository were used as part of a publication accepted for publication in the Journal of Food Science. 

A full description of how this works can be found in our publication:
ADD URL ONCE IT IS PUBLISHED

The code files here were used to do the following:
1) Convert images of paper from RGB to L*, a*, b* and compare to values from a colorimeter
2) Train and validate machine learning models (ANN, RF) to predict the L*, a*, b* values from images
3) Crop pictures of pie crusts using Canny Edge detection
4) Determine the L*, a*, b* values of these images at different lighting conditions.

The folders contain the following:
Detector - functions and code that are used to get the lighting conditions of your image and weight according to the PCA transform approach
Grid Search - the gridsearching that was done for the ANN and RF models made in the paper
Pickles - The saved models and scalers that were trained in this work
Pies - Functions used to process images of the pie crust and run the model on the crust
sample_data - some of the data that was collected. It is impractical to include all pictures taken, but there should be enough to get started with the code
Wrangling - data wrangling code and functions used in this work 
