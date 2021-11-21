// dllmain.cpp : Defines the entry point for the DLL application.

#include "pch.h"

#include <memory>

#include "nd_rng.h"

using std::unique_ptr;
using std::make_unique;

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