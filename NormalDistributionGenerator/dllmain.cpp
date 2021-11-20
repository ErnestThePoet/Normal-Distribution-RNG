// dllmain.cpp : Defines the entry point for the DLL application.

#include "pch.h"

#include <immintrin.h>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <memory>
#include <array>
#include <random>

#define TWO_PI 6.28318530717958647692f	

#define M256_32BIT_CONST_TYPE_ALIGNED(Name,Type,Val) \
	static constexpr __declspec(align(32)) Type Name[8]={Val,Val,Val,Val,Val,Val,Val,Val}

using std::mt19937;
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

	static constexpr size_t vector_size_ = 8;

	static constexpr uint32_t lcg32_a_ = 1664525;
	static constexpr uint32_t lcg32_b_ = 1013904223;
	static constexpr uint32_t float_exp0_mask_ = 0x3F800000;

	M256_32BIT_CONST_TYPE_ALIGNED(packed_lcg32_a_, uint32_t, lcg32_a_);
	M256_32BIT_CONST_TYPE_ALIGNED(packed_lcg32_b_, uint32_t, lcg32_b_);

	M256_32BIT_CONST_TYPE_ALIGNED(packed_float_exp0_mask_, uint32_t, float_exp0_mask_);

	M256_32BIT_CONST_TYPE_ALIGNED(packed_1f_, float, 1.0f);
	M256_32BIT_CONST_TYPE_ALIGNED(packed_2pi_, float, TWO_PI);

	uint32_t previous_lcg_ui32_;

	float GetNextLCGFloat_()
	{
		previous_lcg_ui32_ = previous_lcg_ui32_ * lcg32_a_ + lcg32_b_;
		union
		{
			uint32_t ui;
			float f;
		} u;
		u.ui = (previous_lcg_ui32_ >> 9) | float_exp0_mask_;
		return u.f - 1.0f;
	}

	__m256i packed_previous_lcg_ui32_;

	__m256 GetNextLCGPack_()
	{
		// 对整数向量完成一次线性同余迭代
		packed_previous_lcg_ui32_ = _mm256_add_epi32(
			_mm256_mullo_epi32(packed_previous_lcg_ui32_, *(__m256i*)packed_lcg32_a_),
			*(__m256i*)packed_lcg32_b_);
		
		// 将整数向量的位模式转换为能产生[1.0,2.0)之间的浮点数位模式
		// 首先右移清空符号位和指数位，然后使用掩码将指数位置127
		__m256i float_bit_ready_i32s = _mm256_or_epi32(
			_mm256_srli_epi32(packed_previous_lcg_ui32_, 9),
			*(__m256i*) packed_float_exp0_mask_);

		// 转换为float向量，减去1.0并返回
		__m256 result = _mm256_sub_ps(
			_mm256_castsi256_ps(float_bit_ready_i32s), 
			*(__m256*) packed_1f_);

		return result;
	}

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
	{
		mt19937 initial_ui32_generator(time(nullptr));

		previous_lcg_ui32_ = initial_ui32_generator();

		packed_previous_lcg_ui32_ = _mm256_set_epi32(
			initial_ui32_generator(),
			initial_ui32_generator(),
			initial_ui32_generator(),
			initial_ui32_generator(),
			initial_ui32_generator(),
			initial_ui32_generator(),
			initial_ui32_generator(),
			initial_ui32_generator()
		);
	}

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
	* 成员函数 FloatsAVX2
	* 使用AVX2指令集进行并行优化的Floats函数。
	* 参数count: 要生成的随机数数量。
	*************************************************/
	float* FloatsAVX2(unsigned int count);
};

float NormalDistributionRNG::NextFloat()
{
	return sqrtf(-2 * sigma_square_ * logf(GetNextLCGFloat_()))
		* cosf(TWO_PI * GetNextLCGFloat_()) + mu_;
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
			sqrtf(-2 * sigma_square_ * logf(GetNextLCGFloat_()))
			* cosf(TWO_PI * GetNextLCGFloat_()) + mu_;
	}

	return generated_floats_;
}

float* NormalDistributionRNG::FloatsAVX2(unsigned int count)
{
	if (generated_floats_)
	{
		delete[] generated_floats_;
	}

	generated_floats_ = new float[count];

	__m256 packed_m2_sigma_square = _mm256_set1_ps(-2 * sigma_square_);
	__m256 packed_mu = _mm256_set1_ps(mu_);

	for (unsigned int i = 0; i < count / vector_size_; i++)
	{
		__m256 result = _mm256_sqrt_ps(_mm256_mul_ps(
			_mm256_log_ps(GetNextLCGPack_()),
			packed_m2_sigma_square));

		result = _mm256_mul_ps(
			_mm256_cos_ps(_mm256_mul_ps(
				*(__m256*)packed_2pi_,
				GetNextLCGPack_())),
			result);

		_mm256_store_ps(generated_floats_ + vector_size_ * i,
			_mm256_add_ps(result, packed_mu));
	}

	__m256 remaining_result = _mm256_sqrt_ps(_mm256_mul_ps(
		_mm256_log_ps(GetNextLCGPack_()),
		packed_m2_sigma_square));

	remaining_result = _mm256_mul_ps(
		_mm256_cos_ps(_mm256_mul_ps(
			*(__m256*)packed_2pi_,
			GetNextLCGPack_())),
		remaining_result);

	array<float, vector_size_> remaining_results_array{ 0 };
	_mm256_store_ps(remaining_results_array.data(), remaining_result);

	for (unsigned int i = vector_size_ * (count / vector_size_);
		i < count;
		i++)
	{
		generated_floats_[i] = 
			remaining_results_array[i - vector_size_ * (count / vector_size_)];
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
		return 0.0f;
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

float* ExportFloatsAVX2(unsigned int count)
{
	if (kRNG)
	{
		return kRNG->FloatsAVX2(count);
	}
	else
	{
		return nullptr;
	}
}