Predicting E-Commerce User Intent Using Spark ML

Introduction

Customer behavior is critical to e-commerce success, by understanding actions and customers intents. This project seeks to forecast user intention, such as product viewing, cart addition, or purchase completion, based on the interaction data capured by a leading online retailer. The creation of a predictive model can provide insightful practical applications for businesses, in that it can be used to guide initiatices tailored user experience, and minimize customer churn. Insights gained through this model can be used to power recomendations engines and fuel dynamic, data driven ad campains through users realtime behavior.

Methos

Data Exploration
* Dataset: Kaggke: E-Commerce Behavior Data
* Soure File: 2019-Nov.csv
* Size: Over 4 million rowa, sampled 100,000 for modeling
* Initial Expolation:
    * Used printSchema() and describe() in PySpark
    * Grouped by event_type, category_doce, and brand
    * Visualized price distribution by event_type using Seaborn after coverting to Pandas

Preprocessing
* Inputed missing calues in brand and category_code with 'unknown'
* Added new features:
    * log_price: log-transformed price
    * day and month: extracted from event_time
* Used one hot encoding for categorical variables
* Normalized values in price to help models learn better
* Final features set includes: price, log_price, hour, day, month, and encoded categorical colums

Model 1: Logistic Regression Classifier\n

'''lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data) '''
* Used a full preprocessing pipleline to encode and assemble features 
* Trained a logistical regression model to classify user action

Feature Work - Model 2 Candidate: Decision Tree
* Plan to train DicissionTreeClassifier for comparison
* Will experiment with binary classification setup

Results
Model 1: Logistic Regression
* Training Accuracy: 0.8089
* Test Accuracy: 0.7654

** Insert Image of output ***

Discussion
Logistic Regression
* Used as a baseline classifier due to its interpretability
* Handeles multiclass calsifications out of the box

Anticipated Changes in Future Models
* Decision Trees may handle nonlinear features better
* Binary classification may simplify learning, especially with calss imbalnace

Challanges
* Needed to impute missing data before modeling
* Price values had a skewed distribution; log_price was added to address this
* one-hot encoding categorical variable increases dimensionality

Scientific Thinking
* We transformed and encoded freatures to better match expectations


Conclusion

This project demonstrates the effectiveness of using PySpark to process and model large scale e-commerce interaction data. Through comprehensive preprocessing including normalization, imputation, and feature engineering, we prepared the dataset for predictive modeling. Our initial logistic regression model provided a strong baseline for identifying user intent. Future iterations will incorporate tree based models and binary classification frameworks to enhance both interpretability and predictive performance.

Collaboration

Chris Spears 
* Team Leader, ML Developer, Peoject Manger, Engineer, Reviewer, Testing and Writer
* Modeling, Spark pipeline, Debugging preprocessing, Tested code, provided feedback. Also ensured Repo was organized and ready to upload

Abran Gonzalez
* ML Developer, Engineer, Reviewer, Testing and Writer
* Modeling, Spark pipeline, Debugging preprocessing, Tested code, provided feedback
