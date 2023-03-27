# WCDForest
The source code of WCDForest

将wcdforest包拷贝到/anaconda3/envs/tf1/lib/python3.7/site-packages下<br>


## 步骤：

1.读取数据集并划分为训练集和测试集 <br>
      2.加载配置文件data.json<br>
          config = load_json("data.json")<br>
       3.初始化wcdforest<br>
          wcdforest = WCDForest(config)<br>
      4.开始训练<br>
          wcdforest.fit_transform(X_train, y_train)<br>
       5.验证测试集<br>
          wcdforest.predict(np.array(X_test))<br>
