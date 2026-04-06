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

## 27/03/2026
Once again, I have been defeated by the cache. I am still trying to optimize the serial code, man, I do not even think it is necessary. I am burning out. The goddamn accesses and locality. I am also starting dreaming about cpp. Pls help.

## 31/03/2026
I am kinda blocked on the OMP code. It looks quite too simple. All I did was parallelizing over the rows, but having 8 cores, I guess it is a fair work.

I switched to CUDA because I hate myself and I wanted to finally put my hands on it. Took me again some days. I first tried to simply parallelize over the rows, much like the OMP desguhsting version. It did not look great at all. There must have been another way. So I parallellized the work over the rows for each stride thingy.

As you may grasp it is quite late, I am very sleepy, and I quite cannot write or think.

The FFT looks fast indeed, but I only did the transform over the rows. Now we must transpose and do the columns. Also I guess the revBitOrder might be done over the GPU, but we'll see.

I must sleep. Good night

## 06/04/2026
Implemented cuFFT library. It runs in ~0.2 ms, which is 10 times faster than my raw cuda implementation.

I'm kind of stuck in CUDA, shared memory seems usable only with the transpose kernel. Which is not event the slowest kernel. The main coolSubKer loop is the slowest, yet I cannot find a way to make it faster. Lest I change paradigm, which would be weird?
