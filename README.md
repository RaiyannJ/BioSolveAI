# BioSolveAI
Welcome to our Github Repo! Our BioSolveAI model works on predicting solubility of molecules. We have divided our files as follows:

(1) Baseline and EDA (folder):
    EDA.ipynb #code to inspect solubility distributions and other properties in our database  
    baseline.ipynb #code for XGBoost regression model  

(2) Optimization Trials (folder):
    Bayes Sweep.png #image of our Bayesian optimization for hyperparameter tuning 
    gcn_change1-6.py #different gcn models with various hyperparameters for tuning 
    gcn.py #Python code for the GNN architecture and forward loop
    optimzation.ipynb #Bayesian optimization code 