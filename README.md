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

(3) Data (folder):

    curated-solubility-dataset.csv #raw data from AqSolDB data base

    data_loaders.py #code to extract global features from raw data and get scaffolds and convert to graphs

    testing_data_set.ipynb #this is testing file used to get tensor representations and scaffolds 

(4) Interpretability (folder):

    interpret_utils.py #python code to get node attributions and histograms 

    run_interpret.ipynb #python file for getting attributions and summarizing information across data set 

(5) Additional files (not in folder):

    config.toml #final hyperparameter values used after Baysian optimization

    best_model.pth #final weights/ parameters of GNN model 