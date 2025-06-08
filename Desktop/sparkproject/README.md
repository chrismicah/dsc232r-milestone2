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

Model 1: Logistic Regression Classifier

'''lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data) '''
* Used a full preprocessing pipleline to encode and assemble features 
* Trained a logistical regression model to classify user action

Feature Work - Model 2 Candidate: Decision Tree
* Plan to train DicissionTreeClassifier for comparison
* Will experiment with binary classification setup

