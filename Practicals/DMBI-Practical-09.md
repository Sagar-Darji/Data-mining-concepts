# Practical 09: Implement ID3 and CART algorithm for decision tree classification.  
![](https://img.shields.io/badge/Name-Sagar_Darji-blue.svg?style=flat)
![](https://img.shields.io/badge/Enrollment.no-181310132010-blue.svg?style=flat)

- ID3 (Iterative Dichotomiser 3) was developed in 1986 by Ross Quinlan. The algorithm creates a multiway tree, finding for each node (i.e. in a greedy manner) the categorical feature that will yield the largest information gain for categorical targets. Trees are grown to their maximum size and then a pruning step is usually applied to improve the ability of the tree to generalise to unseen data.
- CART (Classification and Regression Trees) is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yields the largest information gain at each node.

![](https://image.slidesharecdn.com/decisiontrees-150917153729-lva1-app6891/95/l3-decision-trees-29-638.jpg?cb=1442505292)

# ID3 Algorithm:

## Import Play Tennis Data 


```python
 import pandas as pd
from pandas import DataFrame 
df_tennis = DataFrame.from_csv('C:\\Users\\Dr.Thyagaraju\\Desktop\\Data\\PlayTennis.csv')
print("\n Given Play Tennis Data Set:\n\n", df_tennis)
```

    
     Given Play Tennis Data Set:
    
        PlayTennis   Outlook Temperature Humidity    Wind
    0          No     Sunny         Hot     High    Weak
    1          No     Sunny         Hot     High  Strong
    2         Yes  Overcast         Hot     High    Weak
    3         Yes      Rain        Mild     High    Weak
    4         Yes      Rain        Cool   Normal    Weak
    5          No      Rain        Cool   Normal  Strong
    6         Yes  Overcast        Cool   Normal  Strong
    7          No     Sunny        Mild     High    Weak
    8         Yes     Sunny        Cool   Normal    Weak
    9         Yes      Rain        Mild   Normal    Weak
    10        Yes     Sunny        Mild   Normal  Strong
    11        Yes  Overcast        Mild     High  Strong
    12        Yes  Overcast         Hot   Normal    Weak
    13         No      Rain        Mild     High  Strong
    

## Entropy of the Training Data Set


```python
#Function to calculate the entropy of probaility of observations
# -p*log2*p

def entropy(probs):  
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

#Function to calulate the entropy of the given Data Sets/List with respect to target attributes
def entropy_of_list(a_list):  
    #print("A-list",a_list)
    from collections import Counter
    cnt = Counter(x for x in a_list)   # Counter calculates the propotion of class
   # print("\nClasses:",cnt)
    #print("No and Yes Classes:",a_list.name,cnt)
    num_instances = len(a_list)*1.0   # = 14
    print("\n Number of Instances of the Current Sub Class is {0}:".format(num_instances ))
    probs = [x / num_instances for x in cnt.values()]  # x means no of YES/NO
    print("\n Classes:",min(cnt),max(cnt))
    print(" \n Probabilities of Class {0} is {1}:".format(min(cnt),min(probs)))
    print(" \n Probabilities of Class {0} is {1}:".format(max(cnt),max(probs)))
    return entropy(probs) # Call Entropy :
    
# The initial entropy of the YES/NO attribute for our dataset.
print("\n  INPUT DATA SET FOR ENTROPY CALCULATION:\n", df_tennis['PlayTennis'])

total_entropy = entropy_of_list(df_tennis['PlayTennis'])

print("\n Total Entropy of PlayTennis Data Set:",total_entropy)
```

    
      INPUT DATA SET FOR ENTROPY CALCULATION:
     0      No
    1      No
    2     Yes
    3     Yes
    4     Yes
    5      No
    6     Yes
    7      No
    8     Yes
    9     Yes
    10    Yes
    11    Yes
    12    Yes
    13     No
    Name: PlayTennis, dtype: object
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    
     Total Entropy of PlayTennis Data Set: 0.9402859586706309
    

## Information Gain of Attributes 


```python
def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    print("Information Gain Calculation of ",split_attribute_name)
    '''
    Takes a DataFrame of attributes, and quantifies the entropy of a target
    attribute after performing a split along the values of another attribute.
    '''
    # Split Data by Possible Vals of Attribute:
    df_split = df.groupby(split_attribute_name)
   # for name,group in df_split:
    #    print("Name:\n",name)
     #   print("Group:\n",group)
    
    # Calculate Entropy for Target Attribute, as well as
    # Proportion of Obs in Each Data-Split
    nobs = len(df.index) * 1.0
   # print("NOBS",nobs)
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
    #print([target_attribute_name])
    #print(" Entropy List ",entropy_of_list)
    #print("DFAGGENT",df_agg_ent)
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    
    # Calculate Information Gain:
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy


print('Info-gain for Outlook is :'+str( information_gain(df_tennis, 'Outlook', 'PlayTennis')),"\n")
print('\n Info-gain for Humidity is: ' + str( information_gain(df_tennis, 'Humidity', 'PlayTennis')),"\n")
print('\n Info-gain for Wind is:' + str( information_gain(df_tennis, 'Wind', 'PlayTennis')),"\n")
print('\n Info-gain for Temperature is:' + str( information_gain(df_tennis, 'Temperature','PlayTennis')),"\n")
```

    Information Gain Calculation of  Outlook
    
     Number of Instances of the Current Sub Class is 4.0:
    
     Classes: Yes Yes
     
     Probabilities of Class Yes is 1.0:
     
     Probabilities of Class Yes is 1.0:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    Info-gain for Outlook is :0.246749819774 
    
    Information Gain Calculation of  Humidity
    
     Number of Instances of the Current Sub Class is 7.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.42857142857142855:
     
     Probabilities of Class Yes is 0.5714285714285714:
    
     Number of Instances of the Current Sub Class is 7.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.14285714285714285:
     
     Probabilities of Class Yes is 0.8571428571428571:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    
     Info-gain for Humidity is: 0.151835501362 
    
    Information Gain Calculation of  Wind
    
     Number of Instances of the Current Sub Class is 6.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 8.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.25:
     
     Probabilities of Class Yes is 0.75:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    
     Info-gain for Wind is:0.0481270304083 
    
    Information Gain Calculation of  Temperature
    
     Number of Instances of the Current Sub Class is 4.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.25:
     
     Probabilities of Class Yes is 0.75:
    
     Number of Instances of the Current Sub Class is 4.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 6.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.3333333333333333:
     
     Probabilities of Class Yes is 0.6666666666666666:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    
     Info-gain for Temperature is:0.029222565659 
    
    

## ID3 Algorithm


```python
def id3(df, target_attribute_name, attribute_names, default_class=None):
    
    ## Tally target attribute:
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])# class of YES /NO
    
    ## First check: Is this split of the dataset homogeneous?
    if len(cnt) == 1:
        return next(iter(cnt))  # next input data set, or raises StopIteration when EOF is hit.
    
    ## Second check: Is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class  # Return None for Empty Data Set
    
    ## Otherwise: This dataset is ready to be devied up!
    else:
        # Get Default Value for next recursive call of this function:
        default_class = max(cnt.keys()) #No of YES and NO Class
        # Compute the Information Gain of the attributes:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names] 
        index_of_max = gainz.index(max(gainz)) # Index of Best Attribute
        # Choose Best Attribute to split on:
        best_attr = attribute_names[index_of_max]
        
        # Create an empty tree, to be populated in a moment
        tree = {best_attr:{}} # Iniiate the tree with best attribute as a node 
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree
```
## Tree Construction

```python
# Run Algorithm:
from pprint import pprint
tree = id3(df_tennis,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
#print(tree)
pprint(tree)
attribute = next(iter(tree))
print("Best Attribute :\n",attribute)
print("Tree Keys:\n",tree[attribute].keys())
```

    Information Gain Calculation of  Outlook
    
     Number of Instances of the Current Sub Class is 4.0:
    
     Classes: Yes Yes
     
     Probabilities of Class Yes is 1.0:
     
     Probabilities of Class Yes is 1.0:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    Information Gain Calculation of  Temperature
    
     Number of Instances of the Current Sub Class is 4.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.25:
     
     Probabilities of Class Yes is 0.75:
    
     Number of Instances of the Current Sub Class is 4.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 6.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.3333333333333333:
     
     Probabilities of Class Yes is 0.6666666666666666:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    Information Gain Calculation of  Humidity
    
     Number of Instances of the Current Sub Class is 7.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.42857142857142855:
     
     Probabilities of Class Yes is 0.5714285714285714:
    
     Number of Instances of the Current Sub Class is 7.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.14285714285714285:
     
     Probabilities of Class Yes is 0.8571428571428571:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    Information Gain Calculation of  Wind
    
     Number of Instances of the Current Sub Class is 6.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 8.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.25:
     
     Probabilities of Class Yes is 0.75:
    
     Number of Instances of the Current Sub Class is 14.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.35714285714285715:
     
     Probabilities of Class Yes is 0.6428571428571429:
    Information Gain Calculation of  Temperature
    
     Number of Instances of the Current Sub Class is 2.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 3.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.3333333333333333:
     
     Probabilities of Class Yes is 0.6666666666666666:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    Information Gain Calculation of  Humidity
    
     Number of Instances of the Current Sub Class is 2.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 3.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.3333333333333333:
     
     Probabilities of Class Yes is 0.6666666666666666:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    Information Gain Calculation of  Wind
    
     Number of Instances of the Current Sub Class is 2.0:
    
     Classes: No No
     
     Probabilities of Class No is 1.0:
     
     Probabilities of Class No is 1.0:
    
     Number of Instances of the Current Sub Class is 3.0:
    
     Classes: Yes Yes
     
     Probabilities of Class Yes is 1.0:
     
     Probabilities of Class Yes is 1.0:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    Information Gain Calculation of  Temperature
    
     Number of Instances of the Current Sub Class is 1.0:
    
     Classes: Yes Yes
     
     Probabilities of Class Yes is 1.0:
     
     Probabilities of Class Yes is 1.0:
    
     Number of Instances of the Current Sub Class is 2.0:
    
     Classes: No No
     
     Probabilities of Class No is 1.0:
     
     Probabilities of Class No is 1.0:
    
     Number of Instances of the Current Sub Class is 2.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    Information Gain Calculation of  Humidity
    
     Number of Instances of the Current Sub Class is 3.0:
    
     Classes: No No
     
     Probabilities of Class No is 1.0:
     
     Probabilities of Class No is 1.0:
    
     Number of Instances of the Current Sub Class is 2.0:
    
     Classes: Yes Yes
     
     Probabilities of Class Yes is 1.0:
     
     Probabilities of Class Yes is 1.0:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    Information Gain Calculation of  Wind
    
     Number of Instances of the Current Sub Class is 2.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.5:
     
     Probabilities of Class Yes is 0.5:
    
     Number of Instances of the Current Sub Class is 3.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.3333333333333333:
     
     Probabilities of Class Yes is 0.6666666666666666:
    
     Number of Instances of the Current Sub Class is 5.0:
    
     Classes: No Yes
     
     Probabilities of Class No is 0.4:
     
     Probabilities of Class Yes is 0.6:
    
    
    The Resultant Decision Tree is :
    
    {'Outlook': {'Overcast': 'Yes',
                 'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
                 'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}
    Best Attribute :
     Outlook
    Tree Keys:
     dict_keys(['Overcast', 'Rain', 'Sunny'])


# CART (Classification and Regression Trees)

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```


```python
dataset=pd.read_csv('temp.csv')
dataset.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Windy</th>
      <th>Play golf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Rainy</td>
      <td>Hot</td>
      <td>High</td>
      <td>False</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Rainy</td>
      <td>Hot</td>
      <td>High</td>
      <td>True</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Overcast</td>
      <td>Hot</td>
      <td>High</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Sunny</td>
      <td>Mild</td>
      <td>High</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Sunny</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>False</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset=pd.DataFrame(data=dataset.iloc[:,1:6].values,columns=["outlook","temprature","humdity","windy","play"])
filter = dataset["outlook"]=="Rainy"
dataset.where(filter).count()
dataset_encoded=dataset.iloc[:,0:5]
le=LabelEncoder()

for i in dataset_encoded:
    dataset_encoded[i]=le.fit_transform(dataset_encoded[i])

print(dataset_encoded)
print(dataset)
  
```

        outlook  temprature  humdity  windy  play
    0         1           1        0      0     0
    1         1           1        0      1     0
    2         0           1        0      0     1
    3         2           2        0      0     1
    4         2           0        1      0     1
    5         2           0        1      1     0
    6         0           0        1      1     1
    7         1           2        0      0     0
    8         1           0        1      0     1
    9         2           2        1      0     1
    10        1           2        1      1     1
    11        0           2        0      1     1
    12        0           1        1      0     1
    13        2           2        0      1     0
         outlook temprature humdity  windy play
    0      Rainy        Hot    High  False   No
    1      Rainy        Hot    High   True   No
    2   Overcast        Hot    High  False  Yes
    3      Sunny       Mild    High  False  Yes
    4      Sunny       Cool  Normal  False  Yes
    5      Sunny       Cool  Normal   True   No
    6   Overcast       Cool  Normal   True  Yes
    7      Rainy       Mild    High  False   No
    8      Rainy       Cool  Normal  False  Yes
    9      Sunny       Mild  Normal  False  Yes
    10     Rainy       Mild  Normal   True  Yes
    11  Overcast       Mild    High   True  Yes
    12  Overcast        Hot  Normal  False  Yes
    13     Sunny       Mild    High   True   No
    


```python
#Feature Set
X=dataset_encoded.iloc[:,0:4].values
#Label Set
y=dataset_encoded.iloc[:,4].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1,random_state=2)

model=DecisionTreeClassifier(criterion='gini')
model.fit(X_train,y_train)


if model.predict([[0,0,0,1]])==1:
    print("yes you can play")
else:
    print("no you cant")
```

    yes you can play
