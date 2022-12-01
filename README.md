# Credit Card Default Model

**By:** Wenying Wu, Declan Stockdale

**Date:** May 16, 2020

### Overview:

This project is to predict which customers will default on their credit card repayments next month. The data set is based on the publicly available credit card default data set from the UCI Machine Learning Repository. 

Details of the original data are [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).


### Output:

- [Written Report](https://github.com/Wenying-Wu/Credit-Card-Default-Model/blob/main/report_credit_card_default_model.pdf)

- [R workfile](https://github.com/Wenying-Wu/Credit-Card-Default-Model/blob/main/workfile_credit_card_default_model.R)

- [Presentation Slides]()

### Result: 
- Various machine learning algorithms have been applied to predict credit card default as part of a Kaggle competition. The performance metric used in the competition was the area under the receiver operator curve (AUC) where an **XGBoost model** scored the highest AUC of **0.795** outperforming a random forest model, generalized linear model, and a linear logistic model. Furthermore, due to the implications of classifying non-defaulters as defaulters (false positives) in a banking scenario, we have decided that recall is the most appropriate metric and would recommend a XGBoost model. 
- Ranked 2nd in Kaggle competition leaderboard within 20 groups.
<p align="center">
  <img src="https://github.com/Wenying-Wu/Credit-Card-Default-Model/blob/main/src/image001.png">
</p>
