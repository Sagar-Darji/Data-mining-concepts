# Practical 15: Perform Various OLAP operations such slice, dice, roll up, drill up and pivot on given dataset
[![](https://img.shields.io/badge/Name-Sagar_Darji-blue.svg?style=flat)](https://www.linkedin.com/in/sagar-darji-7b7011165/)
![](https://img.shields.io/badge/Enrollment.no-181310132010-blue.svg?style=flat)
#
OLAP (Online Analytic Processing) is an approach to answer multi-dimensional analytical queries swiftly in computing. OLAP operations help in making ad-hoc business decision. Some operations in OLAP include: Drill-Up, Drill-Down, Slice, Dice and Pivot.

# Slicing:


```python
#Slicing Function 

#Each sublist is [imdb id, days since launch, rating]

a = [
 [['123', '1', '5.6'], ['123', '5', '7.7'], ['123', '10', '8.1']],
 [['124', '1', '1.5'], ['124', '5', '4.2'], ['124', '10', '9.1']],
 [['125', '1', '9'], ['125', '5', '8.1'], ['125', '10', '7.5']]
 ]


def search_by_id(_id, _list):
    return _list[0] == str(_id)

def search_by_days(_id, index, _list):
    return _list[index - 1] == str(_id)

""" 
Result first params 

Search by id - j=1
Search by day - j=2

"""

def result(j,val):
    if j == 1:    
        for i in a:
            for b in i:
                if search_by_id(val, b):
                    print(b)
    if j==2:
        for i in a:
            for b in i:
                if search_by_days(val, 2, b):
                    print(b)

result(2,5)
```

    ['123', '5', '7.7']
    ['124', '5', '4.2']
    ['125', '5', '8.1']
    

# Dicing:


```python
a = [[['123', '1', '5.6'], ['123', '5', '7.7'], ['123', '10', '8.1']],
 [['124', '1', '1.5'], ['124', '5', '4.2'], ['124', '10', '9.1']],
 [['125', '1', '9'], ['125', '5', '8.1'], ['125', '10', '7.5']]]

def search_by_id(_id, index, _list):
    return _list[index - 1] == str(_id)

#Find Data in OR and AND format
for i in a:
    for b in i:
        if (search_by_id(123, 1, b) or search_by_id(124, 1, b)) and search_by_id(1, 2, b):
            print(b)

```

    ['123', '1', '5.6']
    ['124', '1', '1.5']
    
