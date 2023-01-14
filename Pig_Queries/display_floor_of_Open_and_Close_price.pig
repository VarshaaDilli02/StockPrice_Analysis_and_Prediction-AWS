math_fn = FOREACH aws GENERATE Open, CEIL(Open), Close, CEIL(Close);
result = limit math_fn 5;
dump result;
