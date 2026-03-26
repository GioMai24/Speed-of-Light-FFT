# <p align='center'> BEHOLD! <\p>
The stuff I did.

## 18/03/2026
After 6 hours of cpp crash course, after 5 days of reading the CUDA programming guide, after 2 days of OMP 'introduction`, and after a week and a half spent on trying to implement the serial version of the Cooley-Tukey Radix 2 FFT from scratch, I finally gave up and looked for a pseudocode.

Well, the thing works... At least it looks like it works... I had to use dynamic allocation to target the heap memory, I know little about the difference between heap and stack, I hope it will not give me problems when I translate the algorithm to CUDA. Just thinking about translating six nested for loops in CUDA is making me alcohol-thirsty.

## 24/03/2026
Even though I haven't been updating the journal, I've been losing my mind on the code[1]. I was desperately trying to figure out why the time registered for the OMP implementation was higher than the serial code... After days I realised I was using a function to count clock cycles used on the CPU... Not the execution time... Well I mean... You know what I mean... I guess both times are interesting in their own ways. I'm tired boss...

I kept playing with the serial code. Transpose is expensive and utterly useless since you can just change the way you look at the array without modifying the values in memory.

After a month and a half things seems to look better and funny. I can't wait to hit my head against an invisible wall again. Both figuratively and irl.

[1]: code::blocks not Virtual Studio Code. I won't fall in that useless trap again.

## 26/03/2026
Even the serial code shows "memory coalescence" issues. The columns part of the algorithm is slower than the rows. This is possibly due to cache lines. A naive transposition of the matrix does not speed up the algorithm for the same issue. There must be a way to do this efficiently, but for now I'll keep it that way.
