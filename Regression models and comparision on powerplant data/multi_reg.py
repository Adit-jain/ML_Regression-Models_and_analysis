import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
def Multinomial_Regressor():
    
    dataset = pd.read_csv("data.csv")
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values
    
    from sklearn.model_selection import train_test_split
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
    
    from sklearn.linear_model import LinearRegression
    
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    Y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))
    
    from sklearn.metrics import r2_score
    
    multi_reg_score = r2_score(Y_test,Y_pred)
    print("Multi regressor r-squared score is : ",multi_reg_score)
    
    
    plt.plot(range(1,21),Y_test[:20],color='blue')
    plt.plot(range(1,21),Y_pred[:20],color='red')
    plt.title("Multinomial Regressor")
    plt.xlabel("Entry number")
    plt.ylabel("Predicted and actual values")
    plt.show()
 

