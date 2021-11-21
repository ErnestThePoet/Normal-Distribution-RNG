#include "pch.h"

#include <immintrin.h>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <array>

#include "nd_rng.h"

#define TWO_PI 6.28318530717958647692f	

using std::array;

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