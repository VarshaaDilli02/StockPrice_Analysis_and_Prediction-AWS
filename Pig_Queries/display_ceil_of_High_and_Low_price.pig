math_fn = FOREACH aws GENERATE High, CEIL(High), Low, CEIL(Low);
result = limit math_fn 5;
dump result;
