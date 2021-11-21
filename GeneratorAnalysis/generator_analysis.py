import time
from ctypes import *
import numpy as np
import scipy.stats as sps
from matplotlib import pyplot as pypl

x_data = [1000000, 10000000, 100000000]
t_numpy = [13.963, 136.635, 1358.368]
t_bm = [19.91, 201.461, 1978.71]
t_bm_sse = [4.025, 45.91, 452.791]
t_bm_avx = [1.995, 19.948, 208.443]

pypl.plot(x_data, t_numpy, label="numpy")
pypl.plot(x_data, t_bm, label="Box-Muller")
pypl.plot(x_data, t_bm_sse, label="Box-Muller(SSE)")
pypl.plot(x_data, t_bm_avx, label="Box-Muller(AVX)")
pypl.xlabel("Count")
pypl.ylabel("Time/ms")
pypl.legend()
pypl.grid()
pypl.show()
exit(0)

# =========================测试数据设置=========================
# 正态分布均值
MU = 0
# 正态分布标准差
SIGMA = 1
# 正态分布随机数生成数量
TOTAL_COUNT = 10000

# 分桶计数时每个桶计数区间的大小
BUCKET_SIZE = 2
# 分桶数量
BUCKET_COUNT = 50

# 耗时测试中生成随机数的数量
TIMING_TEST_TOTAL_COUNT = 100000000
# ============================================================


low_bucket_count = BUCKET_COUNT // 2
high_bucket_count = BUCKET_COUNT - low_bucket_count
bucket_start_numbers = range(MU - BUCKET_SIZE * low_bucket_count,
                             MU + BUCKET_SIZE * high_bucket_count,
                             BUCKET_SIZE)


# 分桶统计，统计各个区间内随机数的数量，返回每个桶区间起始值和各桶内元素数量
def get_bucket_count_data(data):
    buckets = [0] * BUCKET_COUNT

    for i, ie in enumerate(bucket_start_numbers):
        for j in data:
            # 注: Python允许使用连续比较简化代码
            if ie < j <= ie + BUCKET_SIZE:
                buckets[i] += 1

    return bucket_start_numbers, buckets


# 计算并打印随机数<=k的概率与概率分布函数（理论值）的残差平方和，其中k取各个桶区间起始值
# 已被弃用，因为改用Kolmogorov-Smirnov检验
# def analyse_accuracy(buckets):
#     buckets = np.cumsum(buckets)
#     residual_square_sum = 0
#
#     for i in range(0, BUCKET_COUNT):
#         theoretical_probability = \
#             sps.norm.cdf(bucket_start_numbers[i]+BUCKET_SIZE, MU, np.sqrt(
#                 SIGMA_SQUARE))
#         print(f"{i} {buckets[i] / TOTAL_COUNT} "
#               f"{theoretical_probability} "
#               f"{buckets[i] / TOTAL_COUNT - theoretical_probability}")
#         residual_square_sum += (buckets[i] / TOTAL_COUNT -
#                                 theoretical_probability) ** 2
#
#     print(f"残差平方和：{residual_square_sum}")


# Kolmogorov-Smirnov检验
def kolmogorov(result, description, show_plot=False):
    result.sort()

    f_obs = [0] * len(result)
    f_exp = [0] * len(result)
    diff = [0] * len(result)

    for i, ie in enumerate(result):
        f_exp[i] = sps.norm.cdf(ie, MU, SIGMA)
        f_obs[i] = (i + 1) / len(result)
        diff[i] = abs(f_obs[i] - f_exp[i])

    standard_cdf_x = np.arange(MU - 15, MU + 15, 0.1)
    standard_cdf_y = [0] * len(standard_cdf_x)
    for i, ie in enumerate(standard_cdf_x):
        standard_cdf_y[i] = sps.norm.cdf(ie, MU, SIGMA)

    print(f"{description}最大差值：{max(diff)}")

    if not show_plot:
        return

    pypl.plot(standard_cdf_x, standard_cdf_y,
              label="Fexp", color="r", linewidth=1)
    pypl.plot(result, f_obs, label="Fobs", color="b", linewidth=1)
    pypl.xlabel("x")
    pypl.ylabel("F(x)")
    pypl.legend()
    pypl.grid()
    pypl.show()


# =========================测试准备工作=========================
# 加载DLL
bm_module = cdll.LoadLibrary(r"./NormalDistributionGenerator.dll")
bm_module.CreateGenerator.argtypes = [c_float, c_float]
bm_module.NextFloat.restype = c_float
bm_module.Floats.argtypes = [c_uint]
bm_module.Floats.restype = POINTER(c_float)
bm_module.FloatsSSE.argtypes = [c_uint]
bm_module.FloatsSSE.restype = POINTER(c_float)
bm_module.FloatsAVX.argtypes = [c_uint]
bm_module.FloatsAVX.restype = POINTER(c_float)
# 创建随机数生成器
bm_module.CreateGenerator(
    c_float(MU),
    c_float(SIGMA * SIGMA))

rng = np.random.default_rng()
# ============================================================

# =========================生成测试随机数========================
# 使用numpy生成正态随机数作为参照
np_result = rng.normal(MU, SIGMA, TOTAL_COUNT)
np_bucket_data = get_bucket_count_data(np_result)

# 使用NextFloat函数逐个生成一组正态随机数
bm_single_result = [0] * TOTAL_COUNT
for i in range(0, TOTAL_COUNT):
    bm_single_result[i] = bm_module.NextFloat()
bm_single_bucket_data = get_bucket_count_data(bm_single_result)

# 使用Floats函数生成一组正态随机数
_bm_result_ptr = bm_module.Floats(c_uint(TOTAL_COUNT))
bm_result = [0] * TOTAL_COUNT
for i in range(0, TOTAL_COUNT):
    bm_result[i] = _bm_result_ptr[i]
bm_bucket_data = get_bucket_count_data(bm_result)

# 使用以SSE指令集优化的函数生成一组正态随机数
# 【警告】若CPU不支持SSE指令集，则程序将发生错误
_bm_sse_result_ptr = bm_module.FloatsSSE(c_uint(TOTAL_COUNT))
bm_sse_result = [0] * TOTAL_COUNT
for i in range(0, TOTAL_COUNT):
    bm_sse_result[i] = _bm_sse_result_ptr[i]
bm_sse_bucket_data = get_bucket_count_data(bm_sse_result)

# 使用以AVX指令集优化的函数生成一组正态随机数
# 【警告】若CPU不支持AVX指令集，则程序将发生错误
_bm_avx_result_ptr = bm_module.FloatsAVX(c_uint(TOTAL_COUNT))
bm_avx_result = [0] * TOTAL_COUNT
for i in range(0, TOTAL_COUNT):
    bm_avx_result[i] = _bm_avx_result_ptr[i]
bm_avx_bucket_data = get_bucket_count_data(bm_avx_result)
# ============================================================

# =========================测试结果分析=========================
# 进行Kolmogorov-Smirnov检验
# kolmogorov(np_result, "numpy")
# kolmogorov(bm_result, "Box-Muller算法")

# 在同一张图中展示numpy和Box-Muller算法分桶统计结果
# pypl.bar(*bm_bucket_data, label="Box-Muller", color="darkorange")
# pypl.plot(*np_bucket_data, label="numpy", color="orangered")
# pypl.xlabel("x")
# pypl.ylabel("Freq")
# pypl.legend()
# pypl.grid()
# pypl.show()

# 与正态分布的分布函数(理论值)进行残差平方和分析
# analyse_accuracy(np_bucket_data[1])
# analyse_accuracy(bm_bucket_data[1])
# analyse_accuracy(bm_avx_bucket_data[1])
# ============================================================


# =========================算法耗时测试=========================
begin_time = time.time_ns()
result = rng.normal(MU, SIGMA, TIMING_TEST_TOTAL_COUNT)
end_time = time.time_ns()

print(f"numpy耗时(生成{TIMING_TEST_TOTAL_COUNT}个数)"
      f"：{round((end_time - begin_time) / 1000 / 1000, 3)} ms")

begin_time = time.time_ns()
result = bm_module.Floats(TIMING_TEST_TOTAL_COUNT)
end_time = time.time_ns()

bm_time = end_time - begin_time

print(f"Box-Muller算法耗时(生成{TIMING_TEST_TOTAL_COUNT}个数)"
      f"：{round(bm_time / 1000 / 1000, 3)} ms")

begin_time = time.time_ns()
result = bm_module.FloatsSSE(TIMING_TEST_TOTAL_COUNT)
end_time = time.time_ns()

bm_sse_time = end_time - begin_time

print(f"Box-Muller算法(SSE优化)耗时(生成{TIMING_TEST_TOTAL_COUNT}个数)"
      f"：{round(bm_sse_time / 1000 / 1000, 3)} ms "
      f"速率提升倍数{round(bm_time / bm_sse_time, 3)}")

begin_time = time.time_ns()
result = bm_module.FloatsAVX(TIMING_TEST_TOTAL_COUNT)
end_time = time.time_ns()

bm_avx_time = end_time - begin_time

print(f"Box-Muller算法(AVX优化)耗时(生成{TIMING_TEST_TOTAL_COUNT}个数)"
      f"：{round(bm_avx_time / 1000 / 1000, 3)} ms "
      f"速率提升倍数{round(bm_time / bm_avx_time, 3)}")
# ============================================================
