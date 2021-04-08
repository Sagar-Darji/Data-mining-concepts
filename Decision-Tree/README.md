# Decision Tree
> Here used the given database `weather1.csv` to generate Decision tree. 
## 
[![](https://img.shields.io/badge/author-@SagarDarji-blue.svg?style=flat)](https://www.linkedin.com/in/sagar-darji-7b7011165/)

## Import the nessecary libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
## Preparing the Data

```python
df = pd.read_csv('https://github.com/Sagar-Darji/Data-mining-concepts/raw/main/Decision-Tree/weather1.csv')
df
```
>OUT:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Temperature</th>
      <th>Outlook</th>
      <th>Humidity</th>
      <th>Windy</th>
      <th>Played?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mild</td>
      <td>Sunny</td>
      <td>80</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hot</td>
      <td>Sunny</td>
      <td>75</td>
      <td>Yes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hot</td>
      <td>Overcast</td>
      <td>77</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cool</td>
      <td>Rain</td>
      <td>70</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cool</td>
      <td>Overcast</td>
      <td>72</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mild</td>
      <td>Sunny</td>
      <td>77</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cool</td>
      <td>Sunny</td>
      <td>70</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mild</td>
      <td>Rain</td>
      <td>69</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mild</td>
      <td>Sunny</td>
      <td>65</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Mild</td>
      <td>Overcast</td>
      <td>77</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Hot</td>
      <td>Overcast</td>
      <td>74</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Mild</td>
      <td>Rain</td>
      <td>77</td>
      <td>Yes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cool</td>
      <td>Rain</td>
      <td>73</td>
      <td>Yes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Mild</td>
      <td>Rain</td>
      <td>78</td>
      <td>Yes</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


```python
df.dtypes
```
    Temperature    object
    Outlook        object
    Humidity        int64
    Windy          object
    Played?         int64
    dtype: object




```python
df.info()
```

    RangeIndex: 14 entries, 0 to 13
    Data columns (total 5 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   Temperature  14 non-null     object
     1   Outlook      14 non-null     object
     2   Humidity     14 non-null     int64 
     3   Windy        14 non-null     object
     4   Played?      14 non-null     int64 
    dtypes: int64(2), object(3)
    memory usage: 688.0+ bytes
    


```python
df_getdummy= pd.get_dummies(data=df, columns=['Temperature', 'Outlook', 'Windy'])

df_getdummy
```
> OUT:



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Humidity</th>
      <th>Played?</th>
      <th>Temperature_Cool</th>
      <th>Temperature_Hot</th>
      <th>Temperature_Mild</th>
      <th>Outlook_Overcast</th>
      <th>Outlook_Rain</th>
      <th>Outlook_Sunny</th>
      <th>Windy_No</th>
      <th>Windy_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>77</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>69</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>65</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>77</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>74</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>77</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>73</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>78</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### Split the Data in Training data and Testing data

```python
from sklearn.model_selection import train_test_split

X = df_getdummy.drop('Played?',axis=1)
y = df_getdummy['Played?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

## Decision Tree Classifier 

### Fit the model using `DecisionTreeClassifier` 
```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy',max_depth=4)
dtree.fit(X_train,y_train)
```

> OUT:

    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                           max_depth=4, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
predictions = dtree.predict(X_test)

print('Predicted value:',prediction,'Actual value:', y_test)
```

> OUT:

    Predicted value: [1. 1. 1. 0. 1.] 
    Actual value: [0. 1. 1. 1. 1.]    


```python
from sklearn.metrics import classification_report, confusion_matrix

print( confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

> OUT:

                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00         1
               1       0.75      0.75      0.75         4
    
        accuracy                           0.60         5
       macro avg       0.38      0.38      0.38         5
    weighted avg       0.60      0.60      0.60         5
    
    


```python
from sklearn.tree import plot_tree
```


```python
fig = plt.figure(figsize=(16,12))
a = plot_tree(dtree, feature_names=df_getdummy.columns, fontsize=12, filled=True, 
              class_names=['Not Play', 'Play'])
```

> OUT:

![png](decision-tree1.png)



# Decision Tree Regressor

### Fit the model using `DecisionTreeRegressor`
```python
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(max_depth=4)
regressor.fit(X_train, y_train)
```

> OUT:

     DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=4,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, presort='deprecated',
                          random_state=None, splitter='best')

```python
fig = plt.figure(figsize=(16,12))
a = plot_tree(regressor, feature_names=df_getdummy.columns, fontsize=12, filled=True, class_names=['Not Play', 'Play'])
```
> OUT:

![png](decision-tree2.png)



```python
y_pred = regressor.predict(X_test)

print('Predicted value:',y_pred,'Actual value:', y_test)
```

> OUT:
    
    Predicted value: [1. 1. 1. 0. 1.] 
    Actual value: [0. 1. 1. 1. 1.]
    


```python
from sklearn.metrics import mean_squared_error

mse= mean_squared_error(y_pred,y_test)
rmse = np.sqrt(mse)
rmse
```

> OUT:
    
    0.6324555320336759
