**Credit-Scores**

The goal of this repository is to study this Kaggle dataset: https://www.kaggle.com/rikdifos/credit-card-approval-prediction. This dataset contains anonymized credit history data for credit card accounts at a bank, along with financial background data for the customers owning the accounts, including their income, education, occupation and housing status among other features. A description of each feature can be found in the Kaggle link. The goal of this project is to build a model to forecast credit risk

**Datasets:**

There are two data files given in this dataset. 

1. application_record.csv tracks the financial features of the customers owning each account
2. credit_record.csv tracks each account's credit history

There are a number of issues with the data provided that need to be solved before a model can be trained:

1. The application_record does not associate a unique customer to each account. It is clear from the data that multiple accounts are attached to the same customer.
2. There are accounts in one csv file but not in the other.
3. Labels are not provided. We have to parse the credit_record data to construct a label for each account and then combine the different accounts each customer has into one label for the customer.

We solve these issues in the three different Jupyter notebooks in this repository.

**Feature Cleaning and Label Construction:**

The first notebook, titled "Credit Card Data Cleaning and Label Construction" takes the csv files and uses Pandas to clean them. In this notebook, we begin by identifying the different accounts associated to each customer and then reducing the application dataframe to one row per customer, recording the account numbers in a new column. This is stored in the application_data_with_one_row_per_customer csv file. Next, we reduce both datasets to only include their overlap. Finally, we take the credit_record csv file and use a pivot table to assign labels to each account. Compiling these, we assign labels to customers and store them in the application_data_with_harsh_delinquency_labels csv file. These labels are constructed by assigning a bad label to a customer that has any delinquent accounts at all.

**Exploratory Analysis:**

From here, we construct some exploratory analysis in the "Exploratory Analysis" notebook. Here, we first encode our multiclass features via a one-in-K-encoding, remove some outliers and rescale our continuous features. Since logistic regression is a commonly used method to model credit risk (due to the easy interpretability of the logistic score as a credit rating), we compute feature importance with the Information Value of each feature. This computation is explained in the notebook. Unfortunately, none of our features end up having much predictive power. To see how these features relate to our labels, we also compute the correlation coefficients of feature-label pairs. The correlations obtained are very poor. Finally, we visualize the data via a TSNE embedding. This visualization also shows that the different classes are distributed relatively evenly in feature space, without much obvious separation. The outcome of the explorotary analysis is mostly discouraging.

**Model Construction:**

In the third notebook, titled "Credit Rating Models", we train some models on the data. There is one major issue that needs to be solved here: the good customer class has a 3 to 1 ratio with the bad customer class. This is sufficiently imbalanced that accuracy will not be a good metric and it will be harder for our model to learn to identify the bad customers. To fix this, we adopt two techniques: we score our models using the f1_score, the harmonic mean of the precision and recall, and we use SMOTE oversampling to balance the classes after the test train split. We also incorporate cross-validation via grid search to tune the hyperparameters of our model, being careful to avoid information leakage by leaving a holdout set to test on in the end.

The first model we train uses a logistic regression. After tuning, this provides a precision of approximately 0.6 and a recall of approximately 0.3. This is fairly poor performance, as expected from the data exploration conducted earlier. We then train a decision tree classifier on the data but this performs even worse. 


**Future Work:**

I am not particularly hopeful of obtaining a strong model from this data. The correlation of the features seems too low to capture any interesting patterns, the classes seem hard to separate in feature space. However, the goal of this project has largely been educational. I wanted to learn how to work with difficult data that required a lot of work to get into a usable state and how to construct labels when they werent provided. This has been successful, I have gained a lot of understanding of how to manipulate Pandas to efficiently solve data problems.

My next goal with this project is similar. I've recently learned how to use CatBoost so I want to apply some of the techniques in the library to this project. The primary goal is simply to learn how to use CatBoost better, but I also want to see if this can improve the model precision and recall.
