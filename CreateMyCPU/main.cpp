#include<iostream>
#include<fstream>
#include<cstdlib>
#include "Timer.h"
//#include<sys/time.h>
#include<functional>
#include<emmintrin.h>
//#include<immintrin.h>
//void SIMDIntrinsicSSEGaussUnalign8(float** matrix, int size) {
//	for (int k = 0; k < size; k++) {
//		__m256 divisor = _mm256_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
//		//�Ե�һ�в��л�
//		int j = k + 1;
//		for (; j + 7 < size; j += 8) {
//			__m256 diviend = _mm256_loadu_ps(matrix[k] + j);
//			diviend = _mm256_div_ps(diviend, divisor);
//			_mm256_storeu_ps(matrix[k] + j, diviend);
//		}
//		//����ʣ�ಿ��
//		for (j; j < size; j++) {
//			matrix[k][j] /= matrix[k][k];
//		}
//		matrix[k][k] = 1.0;
//		//���Ӿ�����
//		for (int i = k + 1; i < size; i++) {
//			__m256 firstEle = _mm256_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
//			int t = k + 1;
//			for (; t + 7 < size; t += 8) {
//				__m256 multi = _mm256_loadu_ps(matrix[k] + t);
//				multi = _mm256_mul_ps(firstEle, multi);
//				__m256 outcome = _mm256_loadu_ps(matrix[i] + t);
//				outcome = _mm256_sub_ps(outcome, multi);
//				_mm256_storeu_ps(matrix[i] + t, outcome);
//			}
//			for (t; t < size; t++) {
//				matrix[i][t] -= matrix[k][t] * matrix[i][k];
//			}
//			matrix[i][k] = 0;
//		}
//	}
//}

//void SIMDIntrinsicSSEGaussUnalign16(float** matrix, int size) {
//	for (int k = 0; k < size; k++) {
//		__m512 divisor = _mm512_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],
//			matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
//		//�Ե�һ�в��л�
//		int j = k + 1;
//		for (; j + 15 < size; j += 16) {
//			__m512 diviend = _mm512_loadu_ps(matrix[k] + j);
//			diviend = _mm512_div_ps(diviend, divisor);
//			_mm512_storeu_ps(matrix[k] + j, diviend);
//		}
//		//����ʣ�ಿ��
//		for (j; j < size; j++) {
//			matrix[k][j] /= matrix[k][k];
//		}
//		matrix[k][k] = 1.0;
//		//���Ӿ�����
//		for (int i = k + 1; i < size; i++) {
//			__m512 firstEle = _mm512_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k],
//				matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
//			int t = k + 1;
//			for (; t + 15 < size; t += 16) {
//				__m512 multi = _mm512_loadu_ps(matrix[k] + t);
//				multi = _mm512_mul_ps(firstEle, multi);
//				__m512 outcome = _mm512_loadu_ps(matrix[i] + t);
//				outcome = _mm512_sub_ps(outcome, multi);
//				_mm512_storeu_ps(matrix[i] + t, outcome);
//			}
//			for (t; t < size; t++) {
//				matrix[i][t] -= matrix[k][t] * matrix[i][k];
//			}
//			matrix[i][k] = 0;
//		}
//	}
//}
//SIMD Instrinsic SSE without alignment
void SIMDIntrinsicSSEGaussUnalign(float** matrix, int size) {
	for (int k = 0; k < size; k++) {
		__m128 divisor = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
		//�Ե�һ�в��л�
		int j = k + 1;
		for (; j + 3 < size; j += 4) {
			__m128 diviend = _mm_loadu_ps(matrix[k] + j);
			diviend = _mm_div_ps(diviend, divisor);
			_mm_storeu_ps(matrix[k] + j, diviend);
		}
		//����ʣ�ಿ��
		for (j; j < size; j++) {
			matrix[k][j] /= matrix[k][k];
		}
		matrix[k][k] = 1.0;
		//���Ӿ�����
		for (int i = k + 1; i < size; i++) {
			__m128 firstEle = _mm_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
			int t = k + 1;
			for (; t + 3 < size; t += 4) {
				__m128 multi = _mm_loadu_ps(matrix[k] + t);
				multi = _mm_mul_ps(firstEle, multi);
				__m128 outcome = _mm_loadu_ps(matrix[i] + t);
				outcome = _mm_sub_ps(outcome, multi);
				_mm_storeu_ps(matrix[i] + t, outcome);
			}
			for (t; t < size; t++) {
				matrix[i][t] -= matrix[k][t] * matrix[i][k];
			}
			matrix[i][k] = 0;
		}
	}
}

//SIMD Instrinsic SSE with alignment
void SIMDIntrinsicSSEGaussAlign(float** matrix, int size) {
	for (int k = 0; k < size; k++) {
		__m128 divisor = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
		//�Ե�һ�в��л�
		int j = k + 1;
		for (; j + 3 < size; j += 4) {
			__m128 diviend = _mm_load_ps(matrix[k] + j);
			diviend = _mm_div_ps(diviend, divisor);
			_mm_store_ps(matrix[k] + j, diviend);
		}
		//����ʣ�ಿ��
		for (j; j < size; j++) {
			matrix[k][j] /= matrix[k][k];
		}
		matrix[k][k] = 1.0;
		//���Ӿ�����
		for (int i = k + 1; i < size; i++) {
			__m128 firstEle = _mm_set_ps(matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]);
			int t = k + 1;
			for (; t + 3 < size; t += 4) {
				__m128 multi = _mm_load_ps(matrix[k] + t);
				multi = _mm_mul_ps(firstEle, multi);
				__m128 outcome = _mm_load_ps(matrix[i] + t);
				outcome = _mm_sub_ps(outcome, multi);
				_mm_store_ps(matrix[i] + t, outcome);
			}
			for (t; t < size; t++) {
				matrix[i][t] -= matrix[k][t] * matrix[i][k];
			}
			matrix[i][k] = 0;
		}
	}
}
//#include<arm_neon.h>
//#include <arm64_neon.h>
////SIMD Instrinsic Neon
//void SIMDIntrinsicNEONGauss(float** matrix, int size) {
//	for (int k = 0; k < size; k++) {
//		float32x4_t divisor = vdupq_n_f32(matrix[k][k]);
//		//�Ե�һ�в��л�
//		int j = k + 1;
//		for (; j + 3 < size; j += 4) {
//			float32x4_t diviend = vld1q_f32(matrix[k] + j);
//			diviend = vdivq_f32(diviend, divisor);
//			vst1q_f32(matrix[k] + j, diviend);
//		}
//		//����ʣ�ಿ��
//		for (j; j < size; j++) {
//			matrix[k][j] /= matrix[k][k];
//		}
//		matrix[k][k] = 1.0;
//		//���Ӿ�����
//		for (int i = k + 1; i < size; i++) {
//			float32x4_t firstEle = vdupq_n_f32(matrix[i][k]);
//			int t = k + 1;
//			for (; t + 3 < size; t += 4) {
//				float32x4_t multi = vld1q_f32(matrix[k] + t);
//				multi = vmulq_f32(firstEle, multi);
//				float32x4_t outcome = vld1q_f32(matrix[i] + t);
//				outcome = vsubq_f32(outcome, multi);
//				vst1q_f32(matrix[i] + t, outcome);
//			}
//			for (t; t < size; t++) {
//				matrix[i][t] -= matrix[k][t] * matrix[i][k];
//			}
//			matrix[i][k] = 0;
//		}
//	}
//}

//���ݻ�ȡ�Ͷ�̬�ͷ��ڴ�
float** getMatrix(int size) {
	float** matrix = new float* [size];
	for (int i = 0; i < size; i++) {
		//matrix[i] = (float*)aligned_alloc(16, sizeof(float) * static_cast<int>(ceilf((float)size / 16)) * 16);
		//matrix[i] = (float*)_aligned_malloc(sizeof(float) * static_cast<int>(ceilf((float)size / 16)) * 16, 16);
		matrix[i] = new float[size];
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[i][j] = 0.0;
		}
		matrix[i][i] = 1.0;
		for (int j = i + 1; j < size; j++) {
			matrix[i][j] = rand();
		}
	}
	for (int k = 0; k < size; k++) {
		for (int i = k + 1; i < size; i++) {
			for (int j = 0; j < size; j++) {
				matrix[i][j] += matrix[k][j];
			}
		}
	}
	return matrix;
}
float* get1DMatrix(int size) {
	float* matrix = new float[size * size];
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[i * size + j] = 0.0;
		}
		matrix[i * size + i] = 1.0;
		for (int j = i + 1; j < size; j++) {
			matrix[i * size + j] = rand();
		}
	}
	for (int k = 0; k < size; k++) {
		for (int i = k + 1; i < size; i++) {
			for (int j = 0; j < size; j++) {
				matrix[i * size + j] += matrix[k * size + j];
			}
		}
	}
	return matrix;
}
void delMatrix(float** matrix, int size) {
	for (int i = 0; i < size; i++) {
		//free(matrix[i]);
		//_aligned_free(matrix[i]);
		delete[] matrix[i];
	}
	delete[] matrix;
}
//��ͨ��˹��Ԫ��
void ordinarilyGauss(float** matrix, int size)
{
	for (int k = 0; k < size; k++) {
		for (int j = k + 1; j < size; j++)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < size; i++)
		{
			for (int j = k + 1; j < size; j++)
			{
				matrix[i][j] -= matrix[k][j] * matrix[i][k];
			}
			matrix[i][k] = 0;
		}
	}
}

//һά����cache�Ż�����
void cacheGauss(float* matrix, int size)
{
	for (int k = 0; k < size; k++)
	{
		for (int j = k + 1; j < size; j++)
		{
			matrix[k * size + j] /= matrix[k * size + k];
		}
		matrix[k * size + k] = 1.0;
		for (int i = k + 1; i < size; i++)
		{
			for (int j = k + 1; j < size; j++)
			{
				matrix[i * size + j] -= matrix[k * size + j] * matrix[i * size + k];
			}
			matrix[i * size + k] = 0;
		}
	}
}
void testPerformance(int size, std::function<void(float**, int)> func, const std::string& fileName)
{
	//�õ�����ļ�����ض���
	std::fstream fout(fileName, std::ios::in | std::ios::out | std::ios::trunc);
	std::streambuf* coutBackup;
	coutBackup = std::cout.rdbuf(fout.rdbuf());
	//���Բ���,repeat�ǲ��Դ���
	int step = 10;
	int repeat = 100;
	//i�ǲ��Թ�ģ
	for (int i = 0; i <= size; i += step)
	{
		//��������취ʵ��С��ģ�ܼ����ԣ����ģϡ�����
		float** matrix = getMatrix(i);
		//timeval start, finish;
		//����ʵ�ֵ�Timer�������������ڣ����Խ������һ����������
		{
			//��ʱ������
			Timer timer(repeat);
			//gettimeofday(&start, NULL);
			for (int j = 0; j < repeat; j++)
			{
				func(matrix, i);
			}
			//gettimeofday(&finish, NULL);
			//double seconds= 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
			std::cout << i << " ";//<< seconds / repeat/1000 << std::endl;
		}
		delMatrix(matrix, i);
		//���ݾ����ģ�����ظ�����
		switch (i)
		{
		case 100:
			repeat = 20;
			step = 100;
			break;
		case 1000:
			repeat = 10;
			step = 1000;
			break;
		case 10000:
			repeat = 5;
			step = 10000;
			break;
		default:
			break;
		}
	}
	//�ƺ���
	std::cout.rdbuf(coutBackup);
	fout.close();
}
void testPerformance1D(int size, std::function<void(float*, int)> func, const std::string& fileName)
{
	//�õ�����ļ�����ض���
	std::fstream fout(fileName, std::ios::in | std::ios::out | std::ios::trunc);
	std::streambuf* coutBackup;
	coutBackup = std::cout.rdbuf(fout.rdbuf());
	//���Բ���,repeat�ǲ��Դ���
	int step = 10;
	int repeat = 100;
	//i�ǲ��Թ�ģ
	for (int i = 0; i <= size; i += step)
	{
		//��������취ʵ��С��ģ�ܼ����ԣ����ģϡ�����
		float* matrix = get1DMatrix(i);
		//timeval start, finish;
		//����ʵ�ֵ�Timer�������������ڣ����Խ������һ����������
		{
			//��ʱ������
			Timer timer(repeat);
			//gettimeofday(&start, NULL);
			for (int j = 0; j < repeat; j++)
			{
				func(matrix, i);
			}
			//gettimeofday(&finish, NULL);
			//double seconds= 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
			std::cout << i << " ";//<< seconds / repeat/1000 << std::endl;
		}
		delete[] matrix;
		//���ݾ����ģ�����ظ�����
		switch (i)
		{
		case 100:
			repeat = 20;
			step = 100;
			break;
		case 1000:
			repeat = 10;
			step = 1000;
			break;
		case 10000:
			repeat = 5;
			step = 10000;
			break;
		default:
			break;
		}
	}
	//�ƺ���
	std::cout.rdbuf(coutBackup);
	fout.close();
}
int main()
{
	const int TESTSCALE = 1000;
	//testPerformance(TESTSCALE, ordinarilyGauss, "ordinarilyGauss.txt");
	//testPerformance1D(TESTSCALE, cacheGauss, "cacheGauss.txt");
	//testPerformance(TESTSCALE, SIMDIntrinsicNEONGauss, "SIMDIntrinsicNEONGauss.txt");
	testPerformance(TESTSCALE, SIMDIntrinsicSSEGaussAlign, "SIMDIntrinsicSSEGaussAlignAVX.txt");
	//testPerformance(TESTSCALE, SIMDIntrinsicSSEGaussUnalign, "SIMDIntrinsicSSEGaussUnalign.txt");
	//testPerformance(TESTSCALE, SIMDIntrinsicSSEGaussUnalign8, "SIMDIntrinsicSSEGaussUnalign8.txt");
	//testPerformance(TESTSCALE, SIMDIntrinsicSSEGaussUnalign16, "SIMDIntrinsicSSEGaussUnalign16.txt");

}

