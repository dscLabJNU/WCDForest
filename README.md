# WCDForest
The source code of WCDForest


# Dependencies 

+ Python (>= 3.8)
+ NumPy (>= 1.17.3)
+ Scikit-learn (>=0.20)
+ wcdforest package (Copy to site-packages file of python)

## Sept：

       1. Loding trainning set and testing set 
       2. Loding config file data.json
          config = load_json("data.json")
       3. Initialization 
          wcdforest = WCDForest(config)
       4. Fit the training set
          wcdforest.fit_transform(X_train, y_train)
       5. Testing
          wcdforest.predict(np.array(X_test))
