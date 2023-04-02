# deep-learning-challenge
# Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns.
APPLICATION_TYPE—Alphabet Soup application type.
AFFILIATION—Affiliated sector of industry.
CLASSIFICATION—Government organization classification.
USE_CASE—Use case for funding.
ORGANIZATION—Organization type.
STATUS—Active status.
INCOME_AMT—Income classification.
SPECIAL_CONSIDERATIONS—Special considerations for application.
ASK_AMT—Funding amount requested.
IS_SUCCESSFUL—Was the money used effectively,


## Instructions
# Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

# Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

# Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

# Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

# Data Preprocessing

What variable(s) are the target(s) for your model?
What variable(s) are the features for your model?
What variable(s) should be removed from the input data because they are neither targets nor features?
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.


# Requirements
Preprocess the Data 
Create a dataframe containing the charity_data.csv data , and identify the target and feature variables in the dataset 
Drop the EIN and NAME columns 
Determine the number of unique values in each column 
For columns with more than 10 unique values, determine the number of data points for each unique value 
Create a new value called Other that contains rare categorical variables 
Create a feature array, X, and a target array, y by using the preprocessed data 
Split the preprocessed data into training and testing datasets 
Scale the data by using a StandardScaler that has been fitted to the training data 
Compile, Train and Evaluate the Model 
Create a neural network model with a defined number of input features and nodes for each layer 
Create hidden layers and an output layer with appropriate activation functions 
Check the structure of the model 
Compile and train the model 
Evaluate the model using the test data to determine the loss and accuracy 
Export your results to an HDF5 file named AlphabetSoupCharity.h5 
Optimize the Model 
Repeat the preprocessing steps in a new Jupyter notebook 
Create a new neural network model, implementing at least 3 model optimization methods 
Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5 
Write a Report on the Neural Network Model 
Write an analysis that includes a title and multiple sections, labeled with headers and subheaders 
Format images in the report so that they display correction 
Explain the purpose of the analysis 
Answer all 6 questions in the results section 
Summarize the overall results of your model 
Describe how you could use a different model to solve the same problem, and explain why you would use that model 


# References
IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/Links to an external site.

## Results
Data Processing

To clean the data I removed the EIN and NAME columns since they have no value to the model.
The varibales being considered for my model are as follows: 'STATUS', 'ASK_AMT', 'IS_SUCCESSFUL', 'APPLICATION_TYPE', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'INCOME_AMT'. I dropped "USE_CASE_Other","AFFILIATION_Other" columns.
My Dependent varible is "IS_SUCCESFUL" since we want to try to predict this with high accuracy.
Compiling, Training, and Evaluating the Model 
Attempt #1

2 Hidden Layers
80 neurons (Layer1), 30 neurons(Layer2)
Used Relu and Sigmoid Activations Functions since sigmoid is best for binary classifcation problems as this and relu is for nonlinear datasets.
Removed "USE_CASE_Other","AFFILIATION_Other" columns.


Attempt #2

3 Hidden Layers
80 neurons (Layer1), 35 neurons(Layer2), 15 neurons(Layer3)
Used Relu and Sigmoid Activations Functions since sigmoid is best for binary classifcation problems as this and relu is for nonlinear datasets.
Removed "USE_CASE_Other","AFFILIATION_Other" columns.


Attempt #3

3 Hidden Layers
80 neurons(Layer1), 60 neurons(Layer2), 30 neurons (Layer3)
Used Relu and Sigmoid Activations Functions since sigmoid is best for binary classifcation problems as this and relu is for nonlinear datasets.
Went back to original dataset


I tried to change my models in order to achieve a more than 75% accuracy rate but only got about 73%. I changed my features, activation functions, Hidden Layers, and the number of neurons in order to achieve this. But if one where to get this result it would take longer than a more than expected so I am content with the results I got in one day.

Summary
On Average my models kept around 73% accuracy score which is decent considering it was an improvement. My recommendation to improve this model would be to find better features to help explain what determines "IS_SUCCESFUL" such as more indepth knowledge of the other associates/ firms being funded. At the end of the day, knowledge is power and if we had more indepth data between all these applications, we can create a better model.
