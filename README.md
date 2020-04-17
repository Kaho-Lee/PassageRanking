# PassageRanking

Dataset: https://drive.google.com/file/d/1eKDfmDZoVuDADcR_HGMHMnjNJHDrXUs9/view Part2

Dataset: https://drive.google.com/file/d/1eKDfmDZoVuDADcR_HGMHMnjNJHDrXUs9/view Part1

Pre-trained Embedding; http://nlp.stanford.edu/data/glove.6B.zip -> glove.6B.50d.txt

Depandency: nltk, xgboost, tensorflow 2.x, Pandas

Step to run the code:

1. Data preparation: Please use the script dataInit.py to download the needed dataset (very slow) or download the needed dataset mannually. if script used, turn the download flag to ture.

2. Use the script dataInit.py to extract the necessary features for tasks in this project, idf. Please turn init flag to true

3. For running Evaluation Metrics (Task 1).
    1. Run inverted_index.py script to generate the neccesary information for BM25 model, which was developed in the Assignment 1

    2. Run BM25_Retrieval.py. The evaluation metrics will be invoked when the retrieval is finished.
    Note: The BM25 model code is at Retrieval_Engine.py. The evaluation metrics code is at Evalution_Metrics.py

4. For running Logistic Regression (Task 2). Run LogisticRegression.py
Note: change variable mode to 'train' when you want to go through the whole training and predict process. If you want to test an existing saved model parameter, change variable mode to 'test'

5. For running LambdaMART Model (Task 3). Run LambdaMart.py

Note: If you want go through the grid search paramter selection process, change selectModel variable to True. If want to 
train a parameter with all training data, change isTrain variable to True then the model will be saved automatically named as 'LambdaMart.model'. By the way, the best model I got has already been named as 'LambdaMart_all.model'

6. For running Neural Network Model (Task 4). Run nn_passRank.ipynb

Note: I use colab for training model in this task. Please change GPU setting in the code to fit your own envirnment. This notebook present the implementation and the results i got, and you can create a copy of this notebook if you want to play around. The dataset path can be changed with referrence to the previous task. Sorry for any inconvinience.

7. DataGenerator.py is a generic data generation script for the model implemented at this project
