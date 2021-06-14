import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Support_Vector_Machine_Regressor():

    dataset = pd.read_csv("data.csv")
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values
    
    Y = Y.reshape(len(Y),1)
    
    from sklearn.model_selection import train_test_split
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
    
    
    from sklearn.preprocessing import StandardScaler
    
    sc_X = StandardScaler()
    sc_Y = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    Y_train_scaled = sc_Y.fit_transform(Y_train)
    X_test_scaled = sc_X.transform(X_test)
    #Y_test_scaled = sc_Y.transform(Y_test)
    
    from sklearn.svm import SVR
    
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train_scaled,Y_train_scaled)
    
    Y_pred = sc_Y.inverse_transform(regressor.predict(X_test_scaled))
    
    np.set_printoptions(precision=2)
    print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))
    
    from sklearn.metrics import r2_score
    SVR_reg_score = r2_score(Y_test,Y_pred)
    print("SVR r-squared score is : ",SVR_reg_score)
    
    plt.plot(range(1,21),Y_test[:20],color='blue')
    plt.plot(range(1,21),Y_pred[:20],color='red')
    plt.title("Sopport Vector Machine Regressor")
    plt.xlabel("Entry number")
    plt.ylabel("Predicted and actual values")
    plt.show()

