#!/usr/bin/env bash
export LC_ALL=C TIMEFORMAT='%R %U %S' OMP_SCHEDULE="dynamic,1"

date
for threads in {2..8}  #1
do 
	export OMP_NUM_THREADS=$threads
	for ((rows=512; rows<=16384; rows*=2)) #rows in 4096 #
	do 
			for it in {1..50}
			do 
				if [[ $threads -eq 1 ]]
				then
					echo -ne "N:" $rows "- iteration:" $it '\r'
#					echo $rows $( (time ./bench.out $rows) 2>&1 ) >> logs/multImages/3Streams.csv
				else
					echo -ne "threads:" $threads "- N:" $rows "- iteration:" $it '\r'
					echo $threads $rows $( (time ./bench.out $rows) 2>&1 ) >> logs/singleImage/OMP/dynamicTimes.csv
				fi
			done
	done
done
echo "I am done."
date
