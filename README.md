# WCDForest
The source code of WCDForest
将wcdforest包拷贝到/anaconda3/envs/tf1/lib/python3.7/site-packages下


步骤：1.读取数据集并划分为训练集和测试集
       3.加载配置文件data.json
          config = load_json("data.json")
       4.初始化wcdforest
          wcdforest = WCDForest(config)
       5.开始训练
          wcdforest.fit_transform(X_train, y_train)
       6.验证测试集
          wcdforest.predict(np.array(X_test))
