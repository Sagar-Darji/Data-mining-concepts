# Practical 14: Data Warehouse Schema
[![](https://img.shields.io/badge/Name-Sagar_Darji-blue.svg?style=flat)](https://www.linkedin.com/in/sagar-darji-7b7011165/)
![](https://img.shields.io/badge/Enrollment.no-181310132010-blue.svg?style=flat)

# What is Schema?

A schema is a collection of data objects such as tables, views, indexes, etc.

In other words, Schema is the logical representation of database.

### Fact Table:
The table containing foreign keys of dimension tables and the measure.
### Dimension Table:
Contains Primary Key present in Fact table to create a join with fact table and other attributes/properties
### Measure:
Numerical representation of certain values. (e.g. Number of students in class, total income, etc..)  

## Schema Types:
  1.	Star Schema
  2.	SnowFlake Schema
  3.	Fact Constellation Schema

## Star Schema:
![](https://static.javatpoint.com/tutorial/datawarehouse/images/data-warehouse-what-is-star-schema.png)
    
![](https://static.javatpoint.com/tutorial/datawarehouse/images/data-warehouse-what-is-star-schema3.png)

Star Schema is also known as star-join schema.

It has a fact table with a single table for each dimension table.

### Advantage:
•	It is the simplest and easiest schema to design
•	Simple queries to perform on database
•	Most suitable for query processing

### Disadvantage:
•	Repetition of data, hence highly denormalized 

## SnowFlake Schema:

![](https://static.javatpoint.com/tutorial/datawarehouse/images/data-warehouse-what-is-snowflake-schema.png)  

![](https://static.javatpoint.com/tutorial/datawarehouse/images/data-warehouse-what-is-snowflake-schema2.png)

SnowFlake Schema is the variation of Star Schema.
It has multiple levels of dimension tables.
Hence, dimension tables are normalized.

### Advantage:
•	Less redundancies due to normalized dimension tables
•	Dimension table’s attributes are minimized
### Disadvantage:
•	Complex structure than Star Schema
•	Not suitable for query processing as there are multi levels of dimension tables

## Fact Constellation Schema:

![](https://static.javatpoint.com/tutorial/datawarehouse/images/data-warehouse-what-is-fact-constellation-schema.png)

![](https://static.javatpoint.com/tutorial/datawarehouse/images/data-warehouse-what-is-fact-constellation-schema2.png)

Fact Constellation Schema is also known as Galaxy schema.
It has multiple fact tables which shares some dimension tables, means there are some dimension tables common to many fact tables.
### Advantage:
•	Multiple relationships between many field can be obtained
### Disadvantage:
•	Much Complex structure due to multiple fact table 
•	Difficult to manage
•	Dimension tables are very large
