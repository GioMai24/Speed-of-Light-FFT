#!/usr/bin/env bash
export LC_ALL=C TIMEFORMAT='%R %U %S'

date
#for ((rows=512; rows<=16384; rows*=2))
#	do 
		for it in {8..25}
		do 
#			echo -ne "N:" $rows "- iteration:" $it '\r'
#			echo $rows $( (time ./bench.out $rows) 2>&1 ) >> logs/multImages/3Streams.csv
			echo -ne "N:" 4096 "- iteration:" $it '\r'
			echo 4096 $( (time ./bench.out 4096) 2>&1 ) >> logs/multImages/omp.csv
		done
#done
echo "I am done."
date
