#!/usr/bin/env bash
export LC_ALL=C TIMEFORMAT='%R %U %S'

date
for ((rows=512; rows<=16384; rows*=2))
	do 
		for it in {1..100}
		do 
			echo -ne "N:" $rows "- iteration:" $it '\r'
			echo $rows $( (time ./bench.out $rows) 2>&1 ) >> logs/cufft/times.csv
		done
done
echo "I am done."
date
