a = FOREACH aws GENERATE date;
b = FOREACH a GENERATE date, GetYear(Date);
dump b;
