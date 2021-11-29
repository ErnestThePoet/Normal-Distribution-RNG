# Normal-Distribution-Random-Number-Generator

### Description
A fast, lightweight normal distribution random number generator(RNG) based on Box-Muller method, with SSE and AVX2 accelerated versions.

### Quality&Performance
We implemented a simple LCG at the bottom to directly generate uniform distributed floats.
- The normal distribution RNG passed the Kolmogorov-Smirnov test with α=0.01. During the test, the value of `max{abs(Fobs(xi)-Fexp(xi))}` was almost the same as that of numpy.
- For performance, the bare `Floats` is a bit slower than numpy, i.e. about 1.45 times slower. But as we use `FloatsSSE`, the speed will boost to about 3 times higher than numpy. If we continue with `FloatsAVX2`, it can be 7x faster than numpy! Howerer, as numpy generates `np.float64`, our comparison is quite unfair. *** **So this RNG should ONLY be used on occasions where you just want speed and don't need doubles.** ***

### List of exported functions

```
void CreateGenerator(float mu, float sigma_square);
float NextFloat();
float* Floats(unsigned int count);
float* FloatsSSE(unsigned int count);
float* FloatsAVX2(unsigned int count);
```

### How to use
After loading the DLL library, call
`CreateGenerator(float mu, float sigma_square)` to create a generator.
Then, call the following functions based on your need:
- `NextFloat()`: Get next normal distributed float. `Nan` will be returned if you have never called `CreateGenerator` before.
- `Floats(unsigned int count)`: Generate `count` normal distributed floats in an array, and return a pointer to the first element. `nullptr` will be returned if you have never called `CreateGenerator` before, which also applies to the following two functions.
- `FloatsSSE(unsigned int count)`: SSE accelerated `Floats` (~4x faster). *** **NOTE: SSE, SSE2, SSE4.1 intrinsics MUST be supported by your CPU, or your program will crash.** ***
- `FloatsAVX2(unsigned int count)`: AVX+AVX2 accelerated `Floats` (~8x faster). *** **NOTE: AVX and AVX2 intrinsics MUST be supported by your CPU, or your program will crash.** ***

### Analysis figures
1) Fobs and Fexp figure in K-S test
<image src="https://github.com/ErnestThePoet/Normal-Distribution-Random-Number-Generator/blob/master/fig1.png" height="350"/>

2) Random number distribution histogram (v.s. numpy)
<image src="https://github.com/ErnestThePoet/Normal-Distribution-Random-Number-Generator/blob/master/fig2.png" height="350"/>

3) Performance benchmark
<image src="https://github.com/ErnestThePoet/Normal-Distribution-Random-Number-Generator/blob/master/fig3.png" height="350"/>

# Happy Coding!
