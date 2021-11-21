#pragma once

#include "float_lcg.h"

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