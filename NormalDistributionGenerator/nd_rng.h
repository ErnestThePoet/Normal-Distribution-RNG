#pragma once

#include "float_lcg.h"

// ��̬�ֲ��������������
class NormalDistributionRNG
{
private:
	float mu_ = 0.0f;
	float sigma_square_ = 1.0f;

	float* generated_floats_ = nullptr;

	// ��ǰ����������float��LCG��ʵ������ֹƵ������NextFloat()ʱ������������Ӱ������
	FloatLCG<float> float_lcg_float_;

public:
	/************************************************
	* ���캯����
	* ����mu: ��̬�ֲ�������
	* ����sigma_square: ��̬�ֲ��ķ���
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
	* ��Ա���� NextFloat
	* ʹ��Box-Muller���������ɲ�����һ������
	* ������̬�ֲ����������
	*************************************************/
	float NextFloat();

	/************************************************
	* ��Ա���� Floats
	* ����һ�����飬����ʹ��Box-Muller�������ɵĸ����������������
	* ����һ��ָ��float������ָ�룬��ָ������ĵ�һ��Ԫ�ء�
	* ����count: Ҫ���ɵ������������
	*************************************************/
	float* Floats(unsigned int count);

	/************************************************
	* ��Ա���� FloatsSSE
	* ʹ��SSEָ����в����Ż���Floats������
	* ����count: Ҫ���ɵ������������
	* ��ע�⡿��ҪCPU֧��SSE, SSE2, SSE4.1ָ�
	*************************************************/
	float* FloatsSSE(unsigned int count);

	/************************************************
	* ��Ա���� FloatsAVX
	* ʹ��AVXָ����в����Ż���Floats������
	* ����count: Ҫ���ɵ������������
	* ��ע�⡿��ҪCPU֧��AVX, AVX2ָ�
	*************************************************/
	float* FloatsAVX(unsigned int count);
};