# Normal-Distribution-RNG

### Description
A lightweight normal distribution random number generator(RNG) based on Box-Muller method, with SSE and AVX2 accelerated versions.

### Quality&Performance
We implemented a simple LCG at the bottom to directly generate uniform distributed floats.
- The normal distribution RNG passed the Kolmogorov-Smirnov test with α=0.01. During the test, the value of max{abs(Fobs(xi)-Fexp(xi))} was almost the same as that of numpy.
- For performance, the bare `Floats` is a bit slower than numpy, i.e. about 1.45 times slower. But as we use `FloatsSSE`, the speed will boost to about 3 times higher than numpy. If we continue with `FloatsAVX`, it can be 7x faster than numpy! Howerer, as numpy generates `np.float64`, our comparison is quite unfair. **So this RNG should only be used on occasions where you just want speed and don't need doubles.**

### List of exported functions

```
void CreateGenerator(float mu, float sigma_square);
float NextFloat();
float* Floats(unsigned int count);
float* FloatsSSE(unsigned int count);
float* FloatsAVX(unsigned int count);
```

### How to use
After loading the DLL library, call
`CreateGenerator(float mu, float sigma_square)` to create a generator.
Then, call the following functions based on your need:
- `NextFloat()`: Get next normal distributed float.
- `Floats(unsigned int count)`: Generate `count` normal distributed floats in an array, and return a pointer to the first element.
- `FloatsSSE(unsigned int count)`: SSE accelerated `Floats` (~4x faster).  
- **NOTE: SSE, SSE2, SSE4.1 intrinsics MUST be supported by your CPU, or your program may crash.**
- `FloatsAVX(unsigned int count)`: AVX+AVX2 accelerated `Floats` (~8x faster).  
- **NOTE: AVX and AVX2 intrinsics MUST be supported by your CPU, or your program may crash.**

# Happy Coding!
