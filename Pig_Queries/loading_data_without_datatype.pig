aws = LOAD '/AWS_Project/Input/AWS_Project.txt' USING PigStorage(',') AS (date,Open,High,Low,Close,AdjClose,Volume);

