aws_grp = GROUP aws ALL;
result = FOREACH aws_grp GENERATE MAX(aws.Volume);
dump result;
