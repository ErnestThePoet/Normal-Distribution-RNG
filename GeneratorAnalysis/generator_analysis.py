import time
from ctypes import *
import numpy as np
import scipy.stats as sps
from matplotlib import pyplot as pypl

dll_module = cdll.LoadLibrary(r"./NormalDistributionGenerator.dll")
dll_module.CreateGenerator.argtypes = [c_float, c_float]
dll_module.NextFloat.restype = c_float
dll_module.Floats.argtypes = [c_uint]
dll_module.Floats.restype = POINTER(c_float)
dll_module.FloatsAVX2.argtypes = [c_uint]
dll_module.FloatsAVX2.restype = POINTER(c_float)

# 正态分布均值
MU = 5
# 正态分布方差
SIGMA_SQUARE = 16
# 正态分布随机数生成数量
TOTAL_COUNT = 10000

# 中心极限生成算法中被求和的随机变量的个数
RN_COUNT = 10

# 分桶计数时每个桶计数区间的大小
BUCKET_SIZE = 2
# 分桶数量
BUCKET_COUNT = 50

# 耗时测试中生成随机数的数量
TIMING_TEST_TOTAL_COUNT = 3000000

low_bucket_count = BUCKET_COUNT // 2
high_bucket_count = BUCKET_COUNT - low_bucket_count
bucket_start_numbers = range(MU - BUCKET_SIZE * low_bucket_count,
                             MU + BUCKET_SIZE * high_bucket_count,
                             BUCKET_SIZE)


def get_bucket_count_data(data):
    buckets = [0] * BUCKET_COUNT

    for i, ie in enumerate(bucket_start_numbers):
        # 此处不应使用for in循环，因为使用C++生成器的Floats方法时返回指针对象，无法迭代
        for j in range(0, TOTAL_COUNT):
            # 注: Python允许使用连续比较简化代码
            if ie <= data[j] < ie + BUCKET_SIZE:
                buckets[i] += 1

    return bucket_start_numbers, buckets


def analyse_accuracy(buckets):
    buckets = np.cumsum(buckets)
    residual_square_sum = 0

    for i in range(0, BUCKET_COUNT):
        theoretical_probability = \
            sps.norm.cdf(bucket_start_numbers[i]+BUCKET_SIZE, MU, np.sqrt(
                SIGMA_SQUARE))
        print(f"{i} {buckets[i] / TOTAL_COUNT} "
              f"{theoretical_probability} {buckets[i] / TOTAL_COUNT - theoretical_probability}")
        residual_square_sum += (buckets[i] / TOTAL_COUNT -
                                theoretical_probability) ** 2

    print(f"残差平方和：{residual_square_sum}")


rng = np.random.default_rng()
np_result = rng.normal(MU, np.sqrt(SIGMA_SQUARE), TOTAL_COUNT)
np_bucket_data = get_bucket_count_data(np_result)

dll_module.CreateGenerator(
    c_float(MU),
    c_float(SIGMA_SQUARE))

gen_result = dll_module.FloatsAVX2(c_uint(TOTAL_COUNT))
gen_bucket_data = get_bucket_count_data(gen_result)

pypl.plot(*np_bucket_data, label="numpy")
pypl.plot(*gen_bucket_data, label="clgen")
pypl.legend()
pypl.show()

# 与正态分布的分布函数(理论值)进行残差平方和分析
analyse_accuracy(np_bucket_data[1])
analyse_accuracy(gen_bucket_data[1])

# 耗时测试
begin_time = time.time_ns()
result = rng.normal(MU, np.sqrt(SIGMA_SQUARE), TIMING_TEST_TOTAL_COUNT)
end_time = time.time_ns()

print(f"numpy耗时(生成{TIMING_TEST_TOTAL_COUNT}个数)"
      f"：{(end_time - begin_time) / 1000 / 1000} ms")

begin_time = time.time_ns()
result = dll_module.FloatsAVX2(TIMING_TEST_TOTAL_COUNT)
end_time = time.time_ns()

print(f"中心极限生成算法耗时(生成{TIMING_TEST_TOTAL_COUNT}个数)"
      f"：{(end_time - begin_time) / 1000 / 1000} ms")
