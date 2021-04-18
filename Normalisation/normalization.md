```python
import pandas as pd
```


```python
df = pd.read_csv("winequality.csv")
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>white</td>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <td>1</td>
      <td>white</td>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <td>2</td>
      <td>white</td>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <td>3</td>
      <td>white</td>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>white</td>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
    <tr>
      <td>5</td>
      <td>white</td>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <td>6</td>
      <td>white</td>
      <td>6.2</td>
      <td>0.32</td>
      <td>0.16</td>
      <td>7.0</td>
      <td>0.045</td>
      <td>30.0</td>
      <td>136.0</td>
      <td>0.9949</td>
      <td>3.18</td>
      <td>0.47</td>
      <td>9.6</td>
      <td>6</td>
    </tr>
    <tr>
      <td>7</td>
      <td>white</td>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <td>8</td>
      <td>white</td>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <td>9</td>
      <td>white</td>
      <td>8.1</td>
      <td>0.22</td>
      <td>0.43</td>
      <td>1.5</td>
      <td>0.044</td>
      <td>28.0</td>
      <td>129.0</td>
      <td>0.9938</td>
      <td>3.22</td>
      <td>0.45</td>
      <td>11.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## Min-Max Normalization


```python
def Max_min(v):
    b = (((v-Minimum)/(Maximum-Minimum))*(New_Maximum-New_minimum))+New_minimum
    return b

a = df['total sulfur dioxide']
print(a)
Minimum = df['total sulfur dioxide'].min()
print("Minimum = ",Minimum)
Maximum = df['total sulfur dioxide'].max()
print("Maximum = ",Maximum)
```

    0       170.0
    1       132.0
    2        97.0
    3       186.0
    4       186.0
            ...  
    6492     44.0
    6493     51.0
    6494     40.0
    6495     44.0
    6496     42.0
    Name: total sulfur dioxide, Length: 6497, dtype: float64
    Minimum =  6.0
    Maximum =  440.0
    


```python
New_Maximum = 1
New_minimum = 0                    #Range(0-1)
b = []
for i in list(a):               #Normalization
    b.append(Max_min(i))

df["Normalizes"] = b             #adding column
df = df.filter(["total sulfur dioxide","Normalizes"])     #Extract only required Column
df    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total sulfur dioxide</th>
      <th>Normalizes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>170.0</td>
      <td>0.377880</td>
    </tr>
    <tr>
      <td>1</td>
      <td>132.0</td>
      <td>0.290323</td>
    </tr>
    <tr>
      <td>2</td>
      <td>97.0</td>
      <td>0.209677</td>
    </tr>
    <tr>
      <td>3</td>
      <td>186.0</td>
      <td>0.414747</td>
    </tr>
    <tr>
      <td>4</td>
      <td>186.0</td>
      <td>0.414747</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>6492</td>
      <td>44.0</td>
      <td>0.087558</td>
    </tr>
    <tr>
      <td>6493</td>
      <td>51.0</td>
      <td>0.103687</td>
    </tr>
    <tr>
      <td>6494</td>
      <td>40.0</td>
      <td>0.078341</td>
    </tr>
    <tr>
      <td>6495</td>
      <td>44.0</td>
      <td>0.087558</td>
    </tr>
    <tr>
      <td>6496</td>
      <td>42.0</td>
      <td>0.082949</td>
    </tr>
  </tbody>
</table>
<p>6497 rows × 2 columns</p>
</div>



##  Z-Score Normalization


```python
b = df["total sulfur dioxide"]

def Z_score(v,Mean,Standard_deviation):
    Z = (v-Mean)/Standard_deviation       
    return Z

Mean = df["total sulfur dioxide"].mean()
print("Mean = ",Mean)
Standard_deviation = df["total sulfur dioxide"].std()
print("Standard_deviation = ",Standard_deviation)
```

    Mean =  115.7445744189626
    Standard_deviation =  56.52185452263032
    


```python
c = []
for j in list(b):
    c.append(Z_score(j,Mean,Standard_deviation))     #Normalization

df["Normalizes"] = c
df = df.filter(["total sulfur dioxide","Normalizes"])     #Extract only required Column
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total sulfur dioxide</th>
      <th>Normalizes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>170.0</td>
      <td>0.959902</td>
    </tr>
    <tr>
      <td>1</td>
      <td>132.0</td>
      <td>0.287595</td>
    </tr>
    <tr>
      <td>2</td>
      <td>97.0</td>
      <td>-0.331634</td>
    </tr>
    <tr>
      <td>3</td>
      <td>186.0</td>
      <td>1.242978</td>
    </tr>
    <tr>
      <td>4</td>
      <td>186.0</td>
      <td>1.242978</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>6492</td>
      <td>44.0</td>
      <td>-1.269324</td>
    </tr>
    <tr>
      <td>6493</td>
      <td>51.0</td>
      <td>-1.145479</td>
    </tr>
    <tr>
      <td>6494</td>
      <td>40.0</td>
      <td>-1.340094</td>
    </tr>
    <tr>
      <td>6495</td>
      <td>44.0</td>
      <td>-1.269324</td>
    </tr>
    <tr>
      <td>6496</td>
      <td>42.0</td>
      <td>-1.304709</td>
    </tr>
  </tbody>
</table>
<p>6497 rows × 2 columns</p>
</div>



##  Decimal Normalization


```python
c = df["total sulfur dioxide"]

Maximum = df['total sulfur dioxide'].max()
print("Maximum = ",Maximum)
length = len(str(int(Maximum)))
print(length)

def Decimal(v):
    D = v / 10**length            #Decimal Normalizations
    return D

x = []
for j in list(c):
    x.append(Decimal(j))                #Normalization
    
df["Normalizes"] = x
df = df.filter(["total sulfur dioxide","Normalizes"])     #Extract only required Column
df
```

    Maximum =  440.0
    3
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total sulfur dioxide</th>
      <th>Normalizes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>170.0</td>
      <td>0.170</td>
    </tr>
    <tr>
      <td>1</td>
      <td>132.0</td>
      <td>0.132</td>
    </tr>
    <tr>
      <td>2</td>
      <td>97.0</td>
      <td>0.097</td>
    </tr>
    <tr>
      <td>3</td>
      <td>186.0</td>
      <td>0.186</td>
    </tr>
    <tr>
      <td>4</td>
      <td>186.0</td>
      <td>0.186</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>6492</td>
      <td>44.0</td>
      <td>0.044</td>
    </tr>
    <tr>
      <td>6493</td>
      <td>51.0</td>
      <td>0.051</td>
    </tr>
    <tr>
      <td>6494</td>
      <td>40.0</td>
      <td>0.040</td>
    </tr>
    <tr>
      <td>6495</td>
      <td>44.0</td>
      <td>0.044</td>
    </tr>
    <tr>
      <td>6496</td>
      <td>42.0</td>
      <td>0.042</td>
    </tr>
  </tbody>
</table>
<p>6497 rows × 2 columns</p>
</div>


