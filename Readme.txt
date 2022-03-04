This is the Readme file for the project:
We implemented six models:

For KNN, SVM, RandomForest, NaiveBayes models:
1. uncomment the code at the bottom
2. replace "PUT THE TESTING LABEL FILE HERE" with the correct file address for the testing labels
3. run the py file to see the actual accuracy of the model on testing data.

For FNN and CNN models:
Run main file (cnn_main.py and fnn_main.py) in terminal:
python cnn_main.py epoch batch_size learning_rate
For example:
python cnn_main.py 10 256 0.001
And we set the default parameters: epochs = 10, batch_size = 256, learning_rate = 0.001
Using the default, you just need to input: python cnn_main.py

We implemented all models for 5-folds validation, so it runs not faster

~All our six models save the predicition result of “testing_X”as a csv file, each of them takes the name of the algorithm, at the end of the implementation execution.

We saved our predicted result in pred_res file.