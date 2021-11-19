// dllmain.cpp : Defines the entry point for the DLL application.

#include "pch.h"

#include <immintrin.h>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <random>
#include <memory>

#define TWO_PI 6.28318530717958647692f
#define GENERATE_FLOAT \
	sqrtf(-2 * sigma_square_ * logf(uniform_distribution_(engine_))) \
	* cosf(TWO_PI* uniform_distribution_(engine_)) + mu_

#define M256_CONST_TYPE_ALIGN32(Name,Type,Val) \
	static constexpr __declspec(align(32)) Type Name[8]={Val,Val,Val,Val,Val,Val,Val,Val}

using std::mt19937;
using std::uniform_real_distribution;
using std::unique_ptr;
using std::make_unique;

// 正态分布随机数生成器类
class NormalDistributionGenerator
{
private:
	float mu_ = 0.0f;
	float sigma_square_ = 1.0f;

	mt19937 engine_;
	uniform_real_distribution<float> uniform_distribution_;

	float* generated_floats_ = nullptr;

	M256_CONST_TYPE_ALIGN32(lcg_uint32_a_, uint32_t, 1664525);
	M256_CONST_TYPE_ALIGN32(lcg_uint32_b_, uint32_t, 1013904223);

	M256_CONST_TYPE_ALIGN32(packed_uint32_exp0_mask_, uint32_t, 0x3F800000);

	M256_CONST_TYPE_ALIGN32(packed_float_1_, float, 1.0f);
	M256_CONST_TYPE_ALIGN32(packed_float_2pi_, float, TWO_PI);

	__m256i previous_lcg_i32s_;

	__m256 GetNextLCGPack_()
	{
		// 对整数向量完成一次线性同余迭代
		previous_lcg_i32s_ = _mm256_add_epi32(
			_mm256_mullo_epi32(previous_lcg_i32s_, *(__m256i*)lcg_uint32_a_),
			*(__m256i*)lcg_uint32_b_);
		
		// 将整数向量的位模式转换为能产生[1.0,2.0)之间的浮点数位模式
		// 首先右移清空符号位和指数位，然后使用掩码将指数位置127
		__m256i float_bit_ready_i32s = _mm256_or_epi32(
			_mm256_srli_epi32(previous_lcg_i32s_, 9),
			*(__m256i*) packed_uint32_exp0_mask_);

		// 转换为float向量，减去1.0并返回
		__m256 result = _mm256_sub_ps(
			_mm256_castsi256_ps(float_bit_ready_i32s), 
			*(__m256*) packed_float_1_);

		return result;
	}

public:
	/************************************************
	* 构造函数。
	* 参数mu: 正态分布的期望
	* 参数sigma_square: 正态分布的方差
	*************************************************/
	NormalDistributionGenerator(
		float mu,
		float sigma_square) :
		mu_(mu),
		sigma_square_(sigma_square),
		engine_(time(nullptr)),
		uniform_distribution_(0, 1),
		previous_lcg_i32s_(_mm256_set_epi32(
			engine_(),
			engine_(),
			engine_(), 
			engine_(), 
			engine_(), 
			engine_(), 
			engine_(),
			engine_()))
	{}

	~NormalDistributionGenerator()
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

float NormalDistributionGenerator::NextFloat()
{
	return GENERATE_FLOAT;
}

float* NormalDistributionGenerator::Floats(unsigned int count)
{
	if (generated_floats_)
	{
		delete[] generated_floats_;
	}

	generated_floats_ = new float[count];

	for (unsigned int i = 0; i < count; i++)
	{
		generated_floats_[i] = GENERATE_FLOAT;
	}

	return generated_floats_;
}

float* NormalDistributionGenerator::FloatsAVX2(unsigned int count)
{
	if (generated_floats_)
	{
		delete[] generated_floats_;
	}

	generated_floats_ = new float[count];

	__m256 packed_m2_sigma_square = _mm256_set1_ps(-2 * sigma_square_);
	__m256 packed_mu = _mm256_set1_ps(mu_);

	for (unsigned int i = 0; i < count / 8; i++)
	{
		__m256 result = _mm256_sqrt_ps(_mm256_mul_ps(
			_mm256_log_ps(GetNextLCGPack_()),
			packed_m2_sigma_square));

		result = _mm256_mul_ps(
			_mm256_cos_ps(_mm256_mul_ps(
				*(__m256*)packed_float_2pi_,
				GetNextLCGPack_())),
			result);

		_mm256_store_ps(generated_floats_ + 8 * i,
			_mm256_add_ps(result, packed_mu));
	}

	__m256 remaining_result = _mm256_sqrt_ps(_mm256_mul_ps(
		_mm256_log_ps(GetNextLCGPack_()),
		packed_m2_sigma_square));

	remaining_result = _mm256_mul_ps(
		_mm256_cos_ps(_mm256_mul_ps(
			*(__m256*)packed_float_2pi_,
			GetNextLCGPack_())),
		remaining_result);

	float remaining_results_array[8] = { 0 };
	_mm256_store_ps(remaining_results_array, remaining_result);

	for (unsigned int i = 8 * (count / 8); i < count; i++)
	{
		generated_floats_[i] = remaining_results_array[i - 8 * (count / 8)];
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

unique_ptr<NormalDistributionGenerator> normal_distribution_generator = nullptr;

// 以下均为Dll导出函数
void ExportCreateGenerator(
	float mu,
	float sigma_square)
{
	normal_distribution_generator =
		make_unique<NormalDistributionGenerator>(mu, sigma_square);
}

float ExportNextFloat()
{
	if (normal_distribution_generator)
	{
		return normal_distribution_generator->NextFloat();
	}
	else
	{
		return 0.0;
	}
}

float* ExportFloats(unsigned int count)
{
	if (normal_distribution_generator)
	{
		return normal_distribution_generator->Floats(count);
	}
	else
	{
		return nullptr;
	}
}

float* ExportFloatsAVX2(unsigned int count)
{
	if (normal_distribution_generator)
	{
		return normal_distribution_generator->FloatsAVX2(count);
	}
	else
	{
		return nullptr;
	}
}