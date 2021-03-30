# Multi-input Multi-function Activated WASD Neuronet
Implementation of a fast 4-layer feed-forward neuronet model for classification that was trained using a multi-function activated WASD (weights-and-structure-determination) algorithm called MA-WASD.\
The purpose of this package is to create classification models that can predict whether a firm is fraudulent based on present and historical risk factors, as well as predict the probability of default on loans and, therefore, decide if a loan application should be approved or denied.\
By considering this problem's approach through neuronet and by using the weight direct determination (WDD) method along with multiple activation functions, the MA-WASD algorithm finds the optimal weights and structure of the MMA-WASDN and, at the same time, manages the model fitting and validation.\
More precisely, the MA-WASD algorithm finds and keeps only the powers of the activation functions that reduce the model's error during the validation procedure.
The main article used is the following:
*	T.E. Simos, V.N.Katsikis, S.D.Mourtas, " A Multi-Input with Multi-Function Activated WASD Neuronet for Classification Problems with Applications in Firm Fraud and Loan Approval," (submitted)\
Also, the kit includes the following three datasets:
*	Α case study of an external government audit firm that is also the external auditor of India's government firms, taken from https://www.kaggle.com/sid321axn/audit-data?select=audit_data.csv 
*	Α dataset that includes the likelihood that someone will face financial difficulty in the next two years as well as its results based on credit rating algorithms, taken from https://www.kaggle.com/brycecf/give-me-some-credit-dataset?select=cs-training.csv 
*	Α subset of the U.S. SBA loan database, where SBA is short for Small Business Administration, taken from https://www.kaggle.com/larsen0966/sba-loans-case-data-set 

# M-files Description
*	Main_MMA_WASDN.m: the main function
*	problem.m: input data and parameters of the MMA-WASDN
*	MA_WASD.m: function for finding the optimal number of hidden-layer neurons, along with the optimal activation function power at each hidden-layer neuron
*	Qmatrix.m: function for calculating the matrix Q
*	predictN.m: function for predicting
*	error_pred.m: function for calculating the mean absolute error (MAE) of the prediction
*	Normalization.m: function for normalization
*	Problem_figures.m: function for creating the results figures
*	data_prep.m: function for data preparation

# Installation
*	Unzip the file you just downloaded and copy the MMA-WASDN directory to a location,e.g.,/my-directory/
*	Run Matlab/ Octave, Go to /my-directory/MMA-WASDN/ at the command prompt
*	run 'Main_MMA_WASDN (Matlab/Octave)

# Results
After running the 'Main_MMA_WASDN.m file, the package outputs are the following:
*	The optimal number of hidden-layer neurons.
*	The optimal activation function power of each hidden-layer neuron.
*	The prediction and MAE of the testing samples.
*	The graphic illustration of the testing and predicting performance.

# Environment
The MMA-WASDN kit has been tested in Matlab 2018b on OS: Windows 10 64-bit.
