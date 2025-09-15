# sathealth
Bayesian optimization can be directly applied by: python bayes_optimize.py if needed data and trained model are under the same directory.
Please notice that autogluon requires python version to be 3.12 so that program will break if yours doesn't meet the demand. However, the difference may not lead to fatal problems. You can make subtle changes to ignore the warning, which helps the program run successfully.

# Dependencies
To install:  

pip install -r requirements.txt

# train and inference 
To train a light model and get the prediction without Bayesian optimization, unzip the file "dataset.zip" and run "totaldataset_optimal_angle_finder.py" with the dataset file put under the right file path.

To train a model with the complete autogluon structure, run "totaldataset_optimal_angle_finder_auto.py" instead.
