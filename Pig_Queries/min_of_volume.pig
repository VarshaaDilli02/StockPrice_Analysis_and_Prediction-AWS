aws_grp = GROUP aws ALL;
result = FOREACH aws_grp GENERATE MIN(aws.Volume);
dump result;
