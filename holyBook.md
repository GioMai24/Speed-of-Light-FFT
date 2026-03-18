# <p align='center'> BEHOLD! <\p>
The stuff I did.

## 18/03/2026
After 6 hours of cpp crash course, after 5 days of reading the CUDA programming guide, after 2 days of OMP 'introduction`, and after a week and a half spent on trying to implement the serial version of the Cooley-Tukey Radix 2 FFT from scratch, I finally gave up and looked for a pseudocode.

Well, the thing works... At least it looks like it works... I had to use dynamic allocation to target the heap memory, I know little about the difference between heap and stack, I hope it will not give me problems when I translate the algorithm to CUDA. Just thinking about translating six nested for loops in CUDA is making me alcohol-thirsty.
