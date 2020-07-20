# CHARM_Example-MC_Scenarios

* List of regressors:
```
    if flag == 0:
        algo_clf = KernelRidge()
    elif flag == 1:
        algo_clf = LinearSVR()
    elif flag == 2:
        algo_clf = SVR()
    elif flag == 3:
        algo_clf = NuSVR()
    elif flag == 4:
        algo_clf = LinearRegression()
    elif flag == 5:
        algo_clf = Ridge()
    elif flag == 6:
        algo_clf = Lasso()
    elif flag == 7:
        algo_clf = ElasticNet()
    elif flag == 8:
        algo_clf = Lars()
    elif flag == 9:
        algo_clf = LassoLars()
    elif flag == 10:
        algo_clf = BayesianRidge()
    elif flag == 11:
        algo_clf = SGDRegressor(loss="squared_loss")
    elif flag == 12:
        algo_clf = SGDRegressor(loss="huber")
    elif flag == 13:
        algo_clf = SGDRegressor(loss="epsilon_insensitive")
    elif flag == 14:
        algo_clf = KNeighborsRegressor()
    elif flag == 15:
        algo_clf = GaussianProcessRegressor()
    elif flag == 16:
        algo_clf = DecisionTreeRegressor()
    elif flag == 17:
        algo_clf = RandomForestRegressor()
    elif flag == 18:
        algo_clf = ExtraTreesRegressor()
    elif flag == 19:
        algo_clf = BaggingRegressor()
    elif flag == 20:
        algo_clf = AdaBoostRegressor()
    elif flag == 21:
        algo_clf = GradientBoostingRegressor()
    elif flag == 22:
        algo_clf = HistGradientBoostingRegressor()

    if combine_type == "multi_output":
        clf = MultiOutputRegressor(algo_clf).fit(X_train, Y_train)
    elif combine_type == "chain":
        clf = RegressorChain(algo_clf).fit(X_train, Y_train)
    

```
* List of evaluation metrics:
```
if flag_evaluation == 0:
        result_loss = mean_squared_error(y_true, y_pred)          #MSE
        name_eval = mean_squared_error.__name__
elif flag_evaluation == 1:
    result_loss = sqrt(mean_squared_error(y_true, y_pred))        #Root MSE  
    name_eval = 'root_mean_squared_error'
elif flag_evaluation == 2:
    result_loss = mean_absolute_error(y_true, y_pred)             #MAE
    name_eval = mean_absolute_error.__name__
elif flag_evaluation == 3:
    result_loss = explained_variance_score(y_true, y_pred)        
    name_eval = explained_variance_score.__name__   
elif flag_evaluation == 4:
    result_loss = r2_score(y_true, y_pred)
    name_eval = r2_score.__name__
```
