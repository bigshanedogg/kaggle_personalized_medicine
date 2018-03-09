---
title: "Kaggle_personalized_medicine"
author: "Hodong Lee"
date: '2018 03 03 '
---

# EDA for prescription text analysis
18.03.03

- <a href="#1">1. Introduction</a>
    + <a href="#1.1">1.1 Description</a><br>
    + <a href="#1.2">1.2 Library import & Functions setting</a><br>
    + <a href="#1.3">1.3 Data import & breif check</a><br><br>
- <a href="#2">2. Data visualization</a>
    + <a href="#2.1">2.1 Categorical features : Gene, Variation</a><br>
    + <a href="#2.2">2.2 Text data(1) - non-semantic : word/sentence/text length</a><br>
    + <a href="#2.3">2.3 Text data(2) - semantic : word</a><br>
    + <a href="#2.4">2.4 Text data(3) - semantic : bigram</a><br><br>
- <a href="#3">3. Xgboost model & Result</a>
    + <a href="#3.1">3.1 Data preprocessing</a><br>
    + <a href="#3.2">3.2 Defining functions for model</a><br>
    + <a href="#3.3">3.3 Varing input variables (word/bigram, n/tf-idf)</a><br>
    + <a href="#3.4">3.4 Result comparison</a><br>

<br><hr><br>

 This kernel only contained a single xgboost model, but I made two ensemble models by aggregating support vector machine and multi naive bayesian classfier, and xgboost model above. Results of single models submitted by late submssion are ranked near 280th among 1386 teams, Naive bayes showed better results in private leader board, while Xgboost did in public leader board.<br><br>
As can be seen from the table of the process of calculating the accuracy, the above models are basically models with large variance. To improve this, The first model was using Multi-Layer Perceptron in the voting process of aggregating each model by applying bagging.<br><br>
 The second was an ensemble model that I wanted to make personally in this multi-class classification problem, and it applied RIPPER model which is one of the rule learner. (It is a model that is very minor and difficult to guarantee the result, but it is a model that makes me want to experiment.) Precision and recall were judged synthetically, so I chose the models with the best performance for each class. Using divide & seperate algorithm, Each model executes the binary classification for the best predictable class in order, excluding the rows classified as positive, and applies the following model to the remaining rows. I predicted that the result would be improved because it is processed unequally using the best model for each class in the above accuracy model.<br><br>
As a result, the two ensemble models were not as good as the results. The subsequent improving process is not included, because The purpose of this kernel is explanatory data analysis and to apply the processed variable to xgboost. But, In order to design a model which is more general purpose expressive power, it is necessary to solve the imbalance problem of data I think I will. 

<br><hr><br>

* The full version of this analysis can be seen more clearly on the following kaggle kernal page.
(https://www.kaggle.com/bigshane/eda-for-prescription-text-analysis-and-xgboost)

* The full code is uploaded on git-hub. <br>
(https://github.com/bigshanedogg/kaggle_personalized_medicine/)

* You can download raw data at kaggle competition page.<br>
(https://www.kaggle.com/c/msk-redefining-cancer-treatment/data)


\* The update may be a little slow because of the difference between when the actual model was implemented and when the kernel was created.
<br><br><br><br><br>
