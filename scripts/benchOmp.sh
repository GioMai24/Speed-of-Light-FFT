#!/usr/bin/env bash
export LC_ALL=C TIMEFORMAT='%R %U %S'

for threads in {2..4} 8
do 
	export OMP_NUM_THREADS=$threads
	for ((rows=512; rows<=16384; rows*=2))
	do 
		for it in {1..100}
		do 
			if (($it % 10 == 0 || $it == 1))
				then echo "threads:" $threads "- N:" $rows "- iteration:" $it
			fi
			echo $threads $rows $( (time ./bench.out $rows) 2>&1 ) >> logs/OMP/pipelineTimes.csv
		done
	done
	echo ""
done
