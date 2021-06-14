# ML_Regression-Models_and_analysis

The directory uses the following Regressors to predict the outcomes of the data of a powerplant.

  1. Multinomial Regressor
  2. Polynomial Regressor
  3. Support Vector Machine Regressor
  4. Decision Tree Regressor
  5. Random Forest Regressor


The results of each regressor are represented in the graphs using first 20 cases of the test set.

Following are the results achieved via R-squared score:-

    Best results are provided by Random-Forest, and the worst by Decision tree.
    1. Random Forest r-squared score is :  0.9651983913047703
    2. SVR r-squared score is :  0.9480784049986258
    3. Polynomial regressor r-squared score is :  0.9458193001159928
    4. Multi regressor r-squared score is :  0.9325315554761302
    5. Decision Tree regressor r-squared score is :  0.922905874177941
    
Execute the root.py file to start all regressors one by one.

Pros and Cons of each Regressor :-

|Regressor|Pros|Cons|
|---------|----|----|
|Linear Regression|Works on any size of dataset, gives informations about relevance of features|The Linear Regression Assumptions|
|Polynomial Regression|Works on any size of dataset, works very well on non linear problems|Need to choose the right polynomial degree for a good bias/variance tradeoff|
|SVR|Easily adaptable, works very well on non linear problems, not biased by outliers|Compulsory to apply feature scaling, not well known, more difficult to understand|
|Decision Tree Regression|Interpretability, no need for feature scaling,works on both linear / nonlinear problems|Poor results on too small datasets,overfitting can easily occur|
|Random Forest Regression|Powerful and accurate, good performance on many problems, including non linear|No interpretability, overfitting can easily occur, need to choose the number of trees|









