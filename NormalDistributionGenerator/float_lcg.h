#pragma once

#include <ctime>
#include <cstdint>
#include <random>
#include <immintrin.h>

using std::mt19937;

static constexpr uint32_t kLcg32A = 1664525;
static constexpr uint32_t kLcg32B = 1013904223;
static constexpr uint32_t kFloatExp0Mask = 0x3F800000;

#define M128_32BIT_CONST_TYPE_ALIGNED(Name,Type,Val) \
	static constexpr __declspec(align(32)) Type Name[4]={Val,Val,Val,Val}

#define M256_32BIT_CONST_TYPE_ALIGNED(Name,Type,Val) \
	static constexpr __declspec(align(32)) Type Name[8]={Val,Val,Val,Val,Val,Val,Val,Val}

// ����������ͬ�෢�����ӿ�
template<typename T>
class IFloatLCG
{
public:
	IFloatLCG() = default;
	virtual ~IFloatLCG() = default;

	virtual T GetNext() = 0;
};

template<typename T>
class FloatLCG;

template<>
class FloatLCG<float> final :public IFloatLCG<float>
{
private:
	uint32_t previous_lcg_uint32_;

public:
	FloatLCG()
	{
		mt19937 initial_seeder(time(nullptr));
		previous_lcg_uint32_ = initial_seeder();
	}

	~FloatLCG() override = default;

	float GetNext() override
	{
		previous_lcg_uint32_ = previous_lcg_uint32_ * kLcg32A + kLcg32B;
		// ����union����float��λģʽ
		union
		{
			uint32_t ui;
			float f;
		} u;
		u.ui = (previous_lcg_uint32_ >> 9) | kFloatExp0Mask;
		return u.f - 1.0f;
	}
};

template<>
class FloatLCG<__m128> final :public IFloatLCG<__m128>
{
private:
	__m128i packed_previous_lcg_int32_;

	M128_32BIT_CONST_TYPE_ALIGNED(packed_lcg32_a_, uint32_t, kLcg32A);
	M128_32BIT_CONST_TYPE_ALIGNED(packed_lcg32_b_, uint32_t, kLcg32B);
	M128_32BIT_CONST_TYPE_ALIGNED(packed_float_exp0_mask_, uint32_t, kFloatExp0Mask);
	M128_32BIT_CONST_TYPE_ALIGNED(packed_1f_, float, 1.0f);

public:
	FloatLCG()
	{
		mt19937 initial_seeder(time(nullptr));
		packed_previous_lcg_int32_ = _mm_set_epi32(
			initial_seeder(),
			initial_seeder(),
			initial_seeder(),
			initial_seeder()
		);
	}

	~FloatLCG() override = default;

	__m128 GetNext() override
	{
		// �������������һ������ͬ�����
		packed_previous_lcg_int32_ = _mm_add_epi32(
			_mm_mullo_epi32(
				packed_previous_lcg_int32_,
				*(__m128i*)packed_lcg32_a_
			),
			*(__m128i*)packed_lcg32_b_
		);

		// ������������λģʽת��Ϊ�ܲ���[1.0,2.0)֮��ĸ�������λģʽ
		// ����������շ���λ��ָ��λ��Ȼ��ʹ�����뽫ָ��λ��127��
		// ע�⣬��Ȼ�˴������������ǽ�����Ϊ�з����������еģ�
		// �����ƺ���_mm_srli_epi32��0�����λ����˴˴�������Ϊ
		// ���޷�����������һ�¡�
		// ͬʱȡ����ʹ��_mm_or_epi32������ҪAVX512F+AVX512VLָ�
		__m128i packed_float_bit_ready_int32 = _mm_or_si128(
			_mm_srli_epi32(packed_previous_lcg_int32_, 9),
			*(__m128i*)packed_float_exp0_mask_
		);

		// ת��Ϊfloat������ת�������޿���������ȥ1.0f������
		return _mm_sub_ps(
			_mm_castsi128_ps(packed_float_bit_ready_int32),
			*(__m128*)packed_1f_
		);
	}
};

template<>
class FloatLCG<__m256> final :public IFloatLCG<__m256>
{
private:
	__m256i packed_previous_lcg_uint32_;

	M256_32BIT_CONST_TYPE_ALIGNED(packed_lcg32_a_, uint32_t, kLcg32A);
	M256_32BIT_CONST_TYPE_ALIGNED(packed_lcg32_b_, uint32_t, kLcg32B);
	M256_32BIT_CONST_TYPE_ALIGNED(packed_float_exp0_mask_, uint32_t, kFloatExp0Mask);
	M256_32BIT_CONST_TYPE_ALIGNED(packed_1f_, float, 1.0f);

public:
	FloatLCG()
	{
		mt19937 initial_seeder(time(nullptr));
		packed_previous_lcg_uint32_ = _mm256_set_epi32(
			initial_seeder(),
			initial_seeder(),
			initial_seeder(),
			initial_seeder(),
			initial_seeder(),
			initial_seeder(),
			initial_seeder(),
			initial_seeder()
		);
	}

	~FloatLCG() override = default;

	__m256 GetNext() override
	{
		packed_previous_lcg_uint32_ = _mm256_add_epi32(
			_mm256_mullo_epi32(
				packed_previous_lcg_uint32_,
				*(__m256i*)packed_lcg32_a_
			),
			*(__m256i*)packed_lcg32_b_
		);

		__m256i packed_float_bit_ready_int32 = _mm256_or_si256(
			_mm256_srli_epi32(packed_previous_lcg_uint32_, 9),
			*(__m256i*) packed_float_exp0_mask_);

		return _mm256_sub_ps(
			_mm256_castsi256_ps(packed_float_bit_ready_int32),
			*(__m256*) packed_1f_);
	}
};