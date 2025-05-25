We used an e-commerce dataset for this projct.
For preproccessing, we will impute missing categories in the categery_code and brand columns. We will also normalize values in the price column so it is easier to model. Lastly, we 
may possibly one-hot encode event type if we decide to use it for classifaction. 



For part 3, we finished up all major preprocessing tasks. First off, we imputed missing values for 'brand' and 'category_code' by using 'unknown'.
We also added new features. Some of the features we added are log_price, day, and month. 
Also used one hot encoding for categorical features. 

We trained a logistic regression classifier that predicts the action of the user. 

In future iterations, we will try a decision tree as well as binary classification. 