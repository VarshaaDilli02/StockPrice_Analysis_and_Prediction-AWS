aws = LOAD '/AWS_Project/Input/AWS_Project.txt' USING PigStorage(',') AS (date:datetime,Open:double,High:double,Low:double,Close:double,AdjClose:double,Volume:int);

