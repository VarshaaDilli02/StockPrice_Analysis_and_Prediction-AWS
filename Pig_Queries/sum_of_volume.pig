aws_grp=GROUP aws ALL;
result=FOREACH aws_grp GENERATE SUM(aws.Volume);
dump result;
