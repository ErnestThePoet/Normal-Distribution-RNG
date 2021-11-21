// dllmain.cpp : Defines the entry point for the DLL application.

#include "pch.h"

#include <immintrin.h>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <memory>
#include <array>

#include "float_lcg.h"

#define TWO_PI 6.28318530717958647692f	

using std::array;
using std::unique_ptr;
using std::make_unique;

// 正态分布随机数生成器类
class NormalDistributionRNG
{
private:
	float mu_ = 0.0f;
	float sigma_square_ = 1.0f;

	float* generated_floats_ = nullptr;

	// 提前创建好生成float的LCG的实例，防止频繁调用NextFloat()时反复创建对象影响性能
	FloatLCG<float> float_lcg_float_;

public:
	/************************************************
	* 构造函数。
	* 参数mu: 正态分布的期望
	* 参数sigma_square: 正态分布的方差
	*************************************************/
	NormalDistributionRNG(
		float mu,
		float sigma_square) :
		mu_(mu),
		sigma_square_(sigma_square)
	{}

	~NormalDistributionRNG()
	{
		if (generated_floats_)
		{
			delete[] generated_floats_;
		}
	}

	/************************************************
	* 成员函数 NextFloat
	* 使用Box-Muller方法，生成并返回一个满足
	* 给定正态分布的随机数。
	*************************************************/
	float NextFloat();

	/************************************************
	* 成员函数 Floats
	* 生成一个数组，包含使用Box-Muller方法生成的给定数量的随机数。
	* 返回一个指向float变量的指针，它指向数组的第一个元素。
	* 参数count: 要生成的随机数数量。
	*************************************************/
	float* Floats(unsigned int count);

	/************************************************
	* 成员函数 FloatsSSE
	* 使用SSE指令集进行并行优化的Floats函数。
	* 参数count: 要生成的随机数数量。
	* 【注意】需要CPU支持SSE, SSE2, SSE4.1指令集
	*************************************************/
	float* FloatsSSE(unsigned int count);

	/************************************************
	* 成员函数 FloatsAVX
	* 使用AVX指令集进行并行优化的Floats函数。
	* 参数count: 要生成的随机数数量。
	* 【注意】需要CPU支持AVX, AVX2指令集
	*************************************************/
	float* FloatsAVX(unsigned int count);
};

float NormalDistributionRNG::NextFloat()
{
	return sqrtf(-2 * sigma_square_ * logf(float_lcg_float_.GetNext()))
		* cosf(TWO_PI * float_lcg_float_.GetNext()) + mu_;
}

float* NormalDistributionRNG::Floats(unsigned int count)
{
	if (generated_floats_)
	{
		delete[] generated_floats_;
	}

	generated_floats_ = new float[count];

	for (unsigned int i = 0; i < count; i++)
	{
		generated_floats_[i] = 
			sqrtf(-2 * sigma_square_ * logf(float_lcg_float_.GetNext()))
			* cosf(TWO_PI * float_lcg_float_.GetNext()) + mu_;
	}

	return generated_floats_;
}

float* NormalDistributionRNG::FloatsSSE(unsigned int count)
{
	if (generated_floats_)
	{
		delete[] generated_floats_;
	}

	generated_floats_ = new float[count];

	constexpr size_t vector_size = 4;

	FloatLCG<__m128> float_lcg_packed;

	__m128 packed_m2_sigma_square = _mm_set1_ps(-2 * sigma_square_);
	__m128 packed_mu = _mm_set1_ps(mu_);

	M128_32BIT_CONST_TYPE_ALIGNED(packed_2pi, float, TWO_PI);

	for (unsigned int i = 0; i < count / vector_size; i++)
	{
		__m128 result = _mm_sqrt_ps(_mm_mul_ps(
			_mm_log_ps(float_lcg_packed.GetNext()),
			packed_m2_sigma_square));

		result = _mm_mul_ps(
			_mm_cos_ps(_mm_mul_ps(
				*(__m128*)packed_2pi,
				float_lcg_packed.GetNext())),
			result);

		_mm_store_ps(generated_floats_ + vector_size * i,
			_mm_add_ps(result, packed_mu));
	}

	__m128 remaining_result = _mm_sqrt_ps(_mm_mul_ps(
		_mm_log_ps(float_lcg_packed.GetNext()),
		packed_m2_sigma_square));

	remaining_result = _mm_mul_ps(
		_mm_cos_ps(_mm_mul_ps(
			*(__m128*)packed_2pi,
			float_lcg_packed.GetNext())),
		remaining_result);

	array<float, vector_size> remaining_results_array{ 0 };
	_mm_store_ps(remaining_results_array.data(), remaining_result);

	for (unsigned int i = vector_size * (count / vector_size);
		i < count;
		i++)
	{
		generated_floats_[i] =
			remaining_results_array[i - vector_size * (count / vector_size)];
	}

	return generated_floats_;
}

float* NormalDistributionRNG::FloatsAVX(unsigned int count)
{
	if (generated_floats_)
	{
		delete[] generated_floats_;
	}

	generated_floats_ = new float[count];

	constexpr size_t vector_size = 8;

	FloatLCG<__m256> float_lcg_packed;

	__m256 packed_m2_sigma_square = _mm256_set1_ps(-2 * sigma_square_);
	__m256 packed_mu = _mm256_set1_ps(mu_);

	M256_32BIT_CONST_TYPE_ALIGNED(packed_2pi, float, TWO_PI);

	for (unsigned int i = 0; i < count / vector_size; i++)
	{
		__m256 result = _mm256_sqrt_ps(_mm256_mul_ps(
			_mm256_log_ps(float_lcg_packed.GetNext()),
			packed_m2_sigma_square));

		result = _mm256_mul_ps(
			_mm256_cos_ps(_mm256_mul_ps(
				*(__m256*)packed_2pi,
				float_lcg_packed.GetNext())),
			result);

		_mm256_store_ps(generated_floats_ + vector_size * i,
			_mm256_add_ps(result, packed_mu));
	}

	__m256 remaining_result = _mm256_sqrt_ps(_mm256_mul_ps(
		_mm256_log_ps(float_lcg_packed.GetNext()),
		packed_m2_sigma_square));

	remaining_result = _mm256_mul_ps(
		_mm256_cos_ps(_mm256_mul_ps(
			*(__m256*)packed_2pi,
			float_lcg_packed.GetNext())),
		remaining_result);

	array<float, vector_size> remaining_results_array{ 0 };
	_mm256_store_ps(remaining_results_array.data(), remaining_result);

	for (unsigned int i = vector_size * (count / vector_size);
		i < count;
		i++)
	{
		generated_floats_[i] = 
			remaining_results_array[i - vector_size * (count / vector_size)];
	}

	return generated_floats_;
}

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

// 管理正态随机数生成器的实例
unique_ptr<NormalDistributionRNG> kRNG = nullptr;

// 以下均为Dll导出函数
void ExportCreateGenerator(
	float mu,
	float sigma_square)
{
	kRNG = make_unique<NormalDistributionRNG>(mu, sigma_square);
}

float ExportNextFloat()
{
	if (kRNG)
	{
		return kRNG->NextFloat();
	}
	else
	{
		return nanf("RNG Uninitialized");
	}
}

float* ExportFloats(unsigned int count)
{
	if (kRNG)
	{
		return kRNG->Floats(count);
	}
	else
	{
		return nullptr;
	}
}

float* ExportFloatsSSE(unsigned int count)
{
	if (kRNG)
	{
		return kRNG->FloatsSSE(count);
	}
	else
	{
		return nullptr;
	}
}

float* ExportFloatsAVX(unsigned int count)
{
	if (kRNG)
	{
		return kRNG->FloatsAVX(count);
	}
	else
	{
		return nullptr;
	}
}