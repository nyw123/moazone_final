
from kfp.components import create_component_from_func



def make_x():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

    train = pd.read_csv('https://raw.githubusercontent.com/nyw123/moazone_final/master/%EB%85%B8%EC%97%B0%EC%9A%B0/train.csv')
    train = train.drop(['index'], axis=1)
    train.fillna('NAN', inplace=True) 
    x = train.iloc[:,:-1]
    y = train.iloc[:,-1]
    x = pd.get_dummies(x, dummy_na=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=7)


    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)


    import pandas as pd
    from sklearn import metrics
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc



make_x_op = create_component_from_func(func = make_x,
        packages_to_install=['pandas==1.0.5','scikit-learn==0.22.1'])


from kfp.dsl import pipeline


@pipeline(name="RandomForest_Classifier")
def my_pipeline():
    task_1 = make_x_op()






# from kfp.components import create_component_from_func



# def make_x():
#     import pandas as pd
#     train = pd.read_csv('train.csv')
#     train = train.drop(['index'], axis=1)
#     train.fillna('NAN', inplace=True) 
#     x = train.iloc[:,:-1]
#     return x

# def make_y():
#     import pandas as pd
#     train = pd.read_csv('train.csv')
#     train = train.drop(['index'], axis=1)
#     train.fillna('NAN', inplace=True) 
#     y = train.iloc[:,-1]
#     return y

# def label_encoding(x):
#     import pandas as pd
#     x = pd.get_dummies(x, dummy_na=True)
#     return x

# def x_train(x,y):
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=7)
#     return x_train

# def x_test(x,y):
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=7)
#     return x_test

# def y_train(x,y):
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=7)
#     return y_train

# def y_test(x,y):
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=7)
#     return y_test


# def RF_fit_predict(x_train,y_train,x_test):
#     import pandas as pd
#     from sklearn.ensemble import RandomForestClassifier
#     forest = RandomForestClassifier(n_estimators=100)
#     forest.fit(x_train, y_train)
#     y_pred = forest.predict(x_test)
#     return y_pred

# def RF_predict(y_pred,y_test):
#     import pandas as pd
#     from sklearn import metrics
#     acc = metrics.accuracy_score(y_test, y_pred)
#     return acc



# make_x_op = create_component_from_func(make_x)
# make_y_op = create_component_from_func(make_y)
# label_encoding_op = create_component_from_func(label_encoding)
# x_train_op = create_component_from_func(x_train)
# x_test_op = create_component_from_func(x_test)
# y_train_op = create_component_from_func(y_train)
# y_test_op = create_component_from_func(y_test)
# RF_fit_predict_op = create_component_from_func(RF_fit_predict)
# RF_predict_op = create_component_from_func(RF_predict)

# from kfp.dsl import pipeline


# @pipeline(name="RandomForest_Classifier")
# def my_pipeline():
#     task_1 = make_x_op()
#     task_2 = make_y_op()
#     task_3 = label_encoding_op(task_1.output)

#     task_4 = x_train_op(task_3.output, task_2.output)
#     task_5 = x_test_op(task_3.output, task_2.output)
#     task_6 = y_train_op(task_3.output, task_2.output)
#     task_7 = y_test_op(task_3.output, task_2.output)


#     task_8 = RF_fit_predict_op(task_4.output, task_6.output, task_5.output)
#     task_9 = RF_predict_op(task_8.output, task_7.output)


# # if __name__ == "__main__":
# #     kfp.compiler.Compiler().compile(my_pipeline, "./add_pipeline_2.yaml")