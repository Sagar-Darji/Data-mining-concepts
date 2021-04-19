# Practical 12: Implement Linear Regression
[![](https://img.shields.io/badge/Name-Sagar_Darji-blue.svg?style=flat)](https://www.linkedin.com/in/sagar-darji-7b7011165/)
![](https://img.shields.io/badge/Enrollment.no-181310132010-blue.svg?style=flat)

# Linear Regression:
It is the basic and commonly used type for predictive analysis. It is a statistical approach to modelling the relationship between a dependent variable and a given set of independent variables.

**These are of two types:**

  1.Simple linear Regression
   
  2.Multiple Linear Regression

# Simple Linear Regrssion:


Simple linear regression is an approach for predicting a response using a single feature.

It is assumed that the two variables are linearly related. Hence, we try to find a linear function that predicts the response value(y) as accurately as possible as a function of the feature or independent variable(x).

Let us consider a dataset where we have a value of response y for every feature x:

![Example](https://media.geeksforgeeks.org/wp-content/uploads/python-linear-regression.png)

For generality, we define:

x as feature vector, i.e x = [x_1, x_2, …., x_n],

y as response vector, i.e y = [y_1, y_2, …., y_n]

for n observations (in above example, n=10).


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

> Read the Data and make a dataframe


```python
url="http://bit.ly/w-data"
df=pd.read_csv(url)
print(df.shape)
df.head(10)
```

    (25, 2)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.5</td>
      <td>30</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.5</td>
      <td>20</td>
    </tr>
    <tr>
      <td>6</td>
      <td>9.2</td>
      <td>88</td>
    </tr>
    <tr>
      <td>7</td>
      <td>5.5</td>
      <td>60</td>
    </tr>
    <tr>
      <td>8</td>
      <td>8.3</td>
      <td>81</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2.7</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



> To check linearity in the data, Plot the scatter plot 


```python
%matplotlib inline
df.plot(x="Hours", y="Scores", style='o')
plt.xlabel('Hours of study')
plt.ylabel('Score')
plt.show()
```


![png](https://github.com/Sagar-Darji/Data-mining-concepts/blob/main/Linear-regression/output_6_0.png)

Yes, Hours of study and Score  has linear relationship so we can apply linear regression model to our Data. Here we have single variable so that we use simple linear regression.


Now, the task is to find a line which fits best in above scatter plot so that we can predict the response for any new feature values. (i.e a value of x not present in dataset)

This line is called `regression line` .


The equation of regression line is represented as:

![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b84e68c9c191fb6d2bfc659003de4f44_l3.svg)

Here, 


h(x_i) represents the predicted response value for ith observation.

b_0 and b_1 are regression coefficients and represent y-intercept and slope of regression line respectively.



Before spliting the data, We'll reshape the data because different models have different requirement, Store independent and dependent variable in X and Y varibles.


```python
x=np.array(df['Hours']).reshape(-1,1)
y=np.array(df['Scores']).reshape(-1,1)
print('Hours =\n',x)
print('Score =\n',y)
```
| Hours  | Score |
|--------|-------|
| [[2.5] | [[21] |
| [5.1]  | [47]  |
| [3.2]  | [27]  |
| [8.5]  | [75]  |
| [3.5]  | [30]  |
| [1.5]  | [20]  |
| [9.2]  | [88]  |
| [5.5]  | [60]  |
| [8.3]  | [81]  |
| [2.7]  | [25]  |
| [7.7]  | [85]  |
| [5.9]  | [62]  |
| [4.5]  | [41]  |
| [3.3]  | [42]  |
| [1.1]  | [17]  |
| [8.9]  | [95]  |
| [2.5]  | [30]  |
| [1.9]  | [24]  |
| [6.1]  | [67]  |
| [7.4]  | [69]  |
| [2.7]  | [30]  |
| [4.8]  | [54]  |
| [3.8]  | [35]  |
| [6.9]  | [76]  |
| [7.8]] | [86]] |
    

## Split the Data


```python
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=101)
print("Spliting data completed")
```

    Spliting data completed

    

## Training the Model

To create our model, we must “learn” or estimate the values of regression coefficients b_0 and b_1. And once we’ve estimated these coefficients, we can use the model to predict responses!
In this article, we are going to use the principle of  Least Squares .
Now consider:

![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-2c5c24325091449d857bf900827530b9_l3.svg)

Here, e_i is residual error in ith observation.

So, our aim is to minimize the total residual error.

We define the squared error or cost function, J as: 

![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-f27e28f19adac49446a57b98dadd8368_l3.svg)  

and our task is to find the value of b_0 and b_1 for which J(b_0,b_1) is minimum!

Without going into the mathematical details, we present the result here:

![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-68a66dc7677c8c9a2933175bb217c35c_l3.svg)

![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-4bc9cbff05aa167bfe09c90636dff73c_l3.svg)

where SS_xy is the sum of cross-deviations of y and x: 

![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-185a4a366be2907476014df24e56c763_l3.svg) 

and SS_xx is the sum of squared deviations of x: 

![](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-998c95f530ccfc891d0f46c9707a7ba6_l3.svg) 



```python
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)
print('Training completed')
```

    Training completed


## Visualisation on Trained Model


```python
plt.scatter(xtrain, ytrain, color='red', label='Training data')
plt.plot(xtrain,regressor.predict(xtrain), color='blue', label='Regression line')
plt.title('Visuals for Trained Model')
plt.xlabel("Hours of Study")
plt.ylabel('Scores')
plt.legend()
plt.show()
```


![png](https://github.com/Sagar-Darji/Data-mining-concepts/blob/main/Linear-regression/output_15_0.png)


## Visualisation on Testing Data


```python
plt.scatter(xtest, ytest, color='red', label='Testing data')
plt.plot(xtrain,regressor.predict(xtrain), color='blue', label='Regression line')
plt.title('Visuals for Testing Data')
plt.xlabel("Hours of Study")
plt.ylabel('Scores')
plt.legend()
plt.show()
```


![png](https://github.com/Sagar-Darji/Data-mining-concepts/blob/main/Linear-regression/output_17_0.png)


### Making predictions


```python
ypred= regressor.predict(xtest)
dp = pd.Dataframe(({'Test value': xtest, 'Actual': y_test, 'Predicted': y_pred})
```

|   | Test value | Absolute | Predicted      |
|---|------------|----------|----------------|
| 0 | [[2.5]     | [[30]    | [[26.84539693] |
| 1 | [7.7]      | [85]     | [77.45859361]  |
| 2 | [3.8]      | [35]     | [39.4986961 ]  |
| 3 | [7.4]      | [69]     | [74.53860149]  |
| 4 | [5.5]]     | [60]]    | [56.04531809]] |
    



```python
# You can also test with your own data
hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
```

    No of Hours = [[9.25]]
    Predicted Score = [92.54521954]
    

## Evaluating  Model

The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

### R-score value

This value ranges from 0 to 1. Value ‘1’ indicates predictor perfectly accounts for all the variation in Y. Value ‘0’ indicates that predictor ‘x’ accounts for no variation in ‘y’.

Value of R2 may end up being negative if the regression line is made to pass through a point forcefully. This will lead to forcefully making regression line to pass through the origin (no intercept) giving an error higher than the error produced by the horizontal line. This will happen if the data is far away from the origin.

```python
from sklearn.metrics import r2_score
r2_score(ytest, ypred)
```




    0.9377551740781869



### Error

![](https://miro.medium.com/max/613/1*Utp8sgyLk7H39qOQY9pf1A.png)


```python
from sklearn import metrics
print('MAE=',metrics.mean_absolute_error(ytest,ypred),'(Mean absolute error)')
print('MSE',metrics.mean_squared_error(ytest,ypred),'(Mean squared error)')
print('RMSE',np.sqrt(metrics.mean_squared_error(ytest,ypred)),'(Root mean squared error)')
```

    MAE= 4.937597792467705 (Mean absolute error)
    MSE 26.675642597052235 (Mean squared error)
    RMSE 5.164846812544612 (Root mean squared error)
    


# Multiple Linear Regression:
Multiple Linear Regression attempts to model the relationship between two or more features and a response by fitting a linear equation to observed data. The steps to perform multiple linear Regression are almost similar to that of simple linear Regression. The Difference Lies in the evaluation. We can use it to find out which factor has the highest impact on the predicted output and now different variable relate to each other.

>Example

```
Y = b0 + b1 * x1 + b2 * x2 + b3 * x3 + …… bn * xn

Y = Dependent variable and x1, x2, x3, …… xn = multiple independent variables
```


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
```

Take a look at the data set below, it contains some information about Houses.

```python
#Import DataSet 
dataset = pd.read_csv("house_data_new.csv",sep=',')
print(dataset.shape)
dataset.head()
```

    (21613, 21)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>20141013T000000</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>20141209T000000</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>20150225T000000</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>20141209T000000</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>20150218T000000</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



We can predict the price of a House based on the size of house in square ft, but with multiple regression we can throw in more variables, like the no of bathroom, Grade, etc, to make the prediction more accurate.

## Indentfy the independent variables

To Indentfy the independent variables, Pandas `dataframe.corr()` is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.
```python
corr=dataset.corr()
print(corr)
```

                         id     price  bedrooms  bathrooms  sqft_living  sqft_lot  \
    id             1.000000 -0.016762  0.001286   0.005160    -0.012258 -0.132109   
    price         -0.016762  1.000000  0.308350   0.525138     0.702035  0.089661   
    bedrooms       0.001286  0.308350  1.000000   0.515884     0.576671  0.031703   
    bathrooms      0.005160  0.525138  0.515884   1.000000     0.754665  0.087740   
    sqft_living   -0.012258  0.702035  0.576671   0.754665     1.000000  0.172826   
    sqft_lot      -0.132109  0.089661  0.031703   0.087740     0.172826  1.000000   
    floors         0.018525  0.256794  0.175429   0.500653     0.353949 -0.005201   
    waterfront    -0.002721  0.266369 -0.006582   0.063744     0.103818  0.021604   
    view           0.011592  0.397293  0.079532   0.187737     0.284611  0.074710   
    condition     -0.023783  0.036362  0.028472  -0.124982    -0.058753 -0.008958   
    grade          0.008130  0.667434  0.356967   0.664983     0.762704  0.113621   
    sqft_above    -0.010842  0.605567  0.477600   0.685342     0.876597  0.183512   
    sqft_basement -0.005151  0.323816  0.303093   0.283770     0.435043  0.015286   
    yr_built       0.021380  0.054012  0.154178   0.506019     0.318049  0.053080   
    yr_renovated  -0.016907  0.126434  0.018841   0.050739     0.055363  0.007644   
    zipcode       -0.008224 -0.053203 -0.152668  -0.203866    -0.199430 -0.129574   
    lat           -0.001891  0.307003 -0.008931   0.024573     0.052529 -0.085683   
    long           0.020799  0.021626  0.129473   0.223042     0.240223  0.229521   
    sqft_living15 -0.002901  0.585379  0.391638   0.568634     0.756420  0.144608   
    sqft_lot15    -0.138798  0.082447  0.029244   0.087175     0.183286  0.718557   
    
                     floors  waterfront      view  condition     grade  \
    id             0.018525   -0.002721  0.011592  -0.023783  0.008130   
    price          0.256794    0.266369  0.397293   0.036362  0.667434   
    bedrooms       0.175429   -0.006582  0.079532   0.028472  0.356967   
    bathrooms      0.500653    0.063744  0.187737  -0.124982  0.664983   
    sqft_living    0.353949    0.103818  0.284611  -0.058753  0.762704   
    sqft_lot      -0.005201    0.021604  0.074710  -0.008958  0.113621   
    floors         1.000000    0.023698  0.029444  -0.263768  0.458183   
    waterfront     0.023698    1.000000  0.401857   0.016653  0.082775   
    view           0.029444    0.401857  1.000000   0.045990  0.251321   
    condition     -0.263768    0.016653  0.045990   1.000000 -0.144674   
    grade          0.458183    0.082775  0.251321  -0.144674  1.000000   
    sqft_above     0.523885    0.072075  0.167649  -0.158214  0.755923   
    sqft_basement -0.245705    0.080588  0.276947   0.174105  0.168392   
    yr_built       0.489319   -0.026161 -0.053440  -0.361417  0.446963   
    yr_renovated   0.006338    0.092885  0.103917  -0.060618  0.014414   
    zipcode       -0.059121    0.030285  0.084827   0.003026 -0.184862   
    lat            0.049614   -0.014274  0.006157  -0.014941  0.114084   
    long           0.125419   -0.041910 -0.078400  -0.106500  0.198372   
    sqft_living15  0.279885    0.086463  0.280439  -0.092824  0.713202   
    sqft_lot15    -0.011269    0.030703  0.072575  -0.003406  0.119248   
    
                   sqft_above  sqft_basement  yr_built  yr_renovated   zipcode  \
    id              -0.010842      -0.005151  0.021380     -0.016907 -0.008224   
    price            0.605567       0.323816  0.054012      0.126434 -0.053203   
    bedrooms         0.477600       0.303093  0.154178      0.018841 -0.152668   
    bathrooms        0.685342       0.283770  0.506019      0.050739 -0.203866   
    sqft_living      0.876597       0.435043  0.318049      0.055363 -0.199430   
    sqft_lot         0.183512       0.015286  0.053080      0.007644 -0.129574   
    floors           0.523885      -0.245705  0.489319      0.006338 -0.059121   
    waterfront       0.072075       0.080588 -0.026161      0.092885  0.030285   
    view             0.167649       0.276947 -0.053440      0.103917  0.084827   
    condition       -0.158214       0.174105 -0.361417     -0.060618  0.003026   
    grade            0.755923       0.168392  0.446963      0.014414 -0.184862   
    sqft_above       1.000000      -0.051943  0.423898      0.023285 -0.261190   
    sqft_basement   -0.051943       1.000000 -0.133124      0.071323  0.074845   
    yr_built         0.423898      -0.133124  1.000000     -0.224874 -0.346869   
    yr_renovated     0.023285       0.071323 -0.224874      1.000000  0.064357   
    zipcode         -0.261190       0.074845 -0.346869      0.064357  1.000000   
    lat             -0.000816       0.110538 -0.148122      0.029398  0.267048   
    long             0.343803      -0.144765  0.409356     -0.068372 -0.564072   
    sqft_living15    0.731870       0.200355  0.326229     -0.002673 -0.279033   
    sqft_lot15       0.194050       0.017276  0.070958      0.007854 -0.147221   
    
                        lat      long  sqft_living15  sqft_lot15  
    id            -0.001891  0.020799      -0.002901   -0.138798  
    price          0.307003  0.021626       0.585379    0.082447  
    bedrooms      -0.008931  0.129473       0.391638    0.029244  
    bathrooms      0.024573  0.223042       0.568634    0.087175  
    sqft_living    0.052529  0.240223       0.756420    0.183286  
    sqft_lot      -0.085683  0.229521       0.144608    0.718557  
    floors         0.049614  0.125419       0.279885   -0.011269  
    waterfront    -0.014274 -0.041910       0.086463    0.030703  
    view           0.006157 -0.078400       0.280439    0.072575  
    condition     -0.014941 -0.106500      -0.092824   -0.003406  
    grade          0.114084  0.198372       0.713202    0.119248  
    sqft_above    -0.000816  0.343803       0.731870    0.194050  
    sqft_basement  0.110538 -0.144765       0.200355    0.017276  
    yr_built      -0.148122  0.409356       0.326229    0.070958  
    yr_renovated   0.029398 -0.068372      -0.002673    0.007854  
    zipcode        0.267048 -0.564072      -0.279033   -0.147221  
    lat            1.000000 -0.135512       0.048858   -0.086419  
    long          -0.135512  1.000000       0.334605    0.254451  
    sqft_living15  0.048858  0.334605       1.000000    0.183192  
    sqft_lot15    -0.086419  0.254451       0.183192    1.000000  
    

### HEAT MAP

For better visualization and understanding plot the Heat Map

```python
import seaborn as sns
sns.heatmap(corr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20a29c1ea08>




![png](https://github.com/Sagar-Darji/Data-mining-concepts/blob/main/Linear-regression/output_5_1.png)

From above Heat map, We can conclude that 'sqft_living','bathrooms','grade','sqft_above' these attributes have more corelation with price of House. 

Then make a list of the independent values and call this variable X.

Put the dependent values in a variable called y.

```python
feature_cols = ['sqft_living','bathrooms','grade','sqft_above']
space=dataset[feature_cols]
price=dataset['price']
```


```python
x = np.array(space)
y = np.array(price)
```
## Split the Data 

```python
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
print(ytrain) 
print(ytest)
```

    [495000. 635000. 382500. ... 431000. 411000. 699900.]
    [ 297000. 1578000.  562100. ...  369950.  300000.  575950.]
 
## Fit the Model

Now, Multiple regression is like linear regression. 

Do the same as linear regression

```python
#Fit Multiple linear regression on Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
print("Training complete.")
```

    Training complete.
    
The coefficient is a factor that describes the relationship with an unknown variable.

Example: if x is a variable, then 2x is x two times. x is the unknown variable, and the number 2 is the coefficient.

The answer(s) we get tells us what would happen if we increase, or decrease, one of the independent values.

```python
#Apply on Test Set 
ypred = regressor.predict(xtest)
print('Predicted Data =\n',ypred)

print('Slope =\n',regressor.coef_)

print('Intercept =\n',regressor.intercept_)
```

    Predicted Data =
     [ 350809.60170938 1438192.59530085  398956.16794733 ...  300243.12927225
      169817.41553372  395588.76691402]
    Slope =
     [ 2.58571743e+02 -3.77222761e+04  1.11778835e+05 -7.50832248e+01]
    Intercept =
     -637447.4129246195
    
### Make the prediction 

```python
# You can also test with your own data
# sqft_living=2000, bathrooms=2, grade=6, sqft_above=2000
regressor.predict([[2000,2,6,2000]])
```




    array([324758.08391123])



### R-SQUARED VALUE


```python
from sklearn.metrics import r2_score
r2_score(ytest, ypred)
```




    0.5481199319093966



### ERROR


```python
from sklearn import metrics
print(metrics.mean_absolute_error(ytest,ypred))
print(metrics.mean_squared_error(ytest,ypred))
print(np.sqrt(metrics.mean_squared_error(ytest,ypred)))
```

    157412.92126076441
    53739506907.39486
    231817.83129732462
    
