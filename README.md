# Neural Network Charity Analysis: 
# Loan Prediction Risk Assessment

## Overview
My client, Beks, is a data analyst at Alphabet Soup; a nonprofit, philanthropic foundation that has donated more than $10 billion to organizations that protect the environment, improve people's wellbeing, and unify the world. Beks is responsible for analyzing the impact of every dollar donated and vetting potential recipients, to ensure the foundation's money is being used effectively. I am helping Beks build a machine learning model to try and predict which future recipients should receive donations, and which are too high-risk.

### Development
Using the Tensor Flow library for Python, along with our knowledge of statistics and machine learning, Beks and I built a deep-learning neural network model. We used the features in the dataset Beks provided to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. 

Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within the dataset were a variety of metadata about each organization, such as: name, donation application info, and organizational business info such as income, government classification, business sector, and more. 

### Analysis
We were asked to provide three separate deliverables, along with a report of ouf findings. The three steps include:
  • Preprocess Data for a Neural Network Model
  
  • Compile, Train, and Evaluate the Model
  
  • Optimize the Model

## Data Preprocessing Results
**Target Variable:** "IS_SUCCESSFUL" (Whether a particular recipient is predicted to use a donation successfully, or not)
  
**Feature Variables:** "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", 
  
**Variables that were neither targets nor features (and removed from the input data):** "EIN", "NAME", "ASK_AMT", "STATUS"
  
## Compiling, Training, and Evaluating the Model
**Neurons, layers, and activation functions**
For our neural network model, we selected the following:
• 190 neurons in input layer to increase the original number of variables (80; increased to 190 for higher accuracy), using Relu activation
• Hidden layer with 30 neurons, using Relu
• Output layer with one neuron, using Sigmoid 
• A total of 13,741 neurons
  
**Our attempt to reach 75% accuracy**
We were not able to achieve the target model performance (75%), however we did experience a .0005 increase, from .7257 to .7262. The slight increase is due to the following attempts:
• **Successful attempts**
  • Removed "ASK_AMT" and "STATUS" as noisy variables
  • Increased layer one neurons from 80 to 190
  
• **Failed attempts**
  • Increase hidden layer 2's neurons to 70 from 30
  • Added an additional hidden layer
  • Changed the output activation style to Relu and Tanh
