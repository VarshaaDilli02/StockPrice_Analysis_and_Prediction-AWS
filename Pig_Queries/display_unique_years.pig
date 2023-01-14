dt = FOREACH aws GENERATE GetYear(Date);
unique_dt = distinct dt;
dump unique_dt;
