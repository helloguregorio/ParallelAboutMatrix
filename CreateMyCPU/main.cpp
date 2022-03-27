#include<iostream>
#include<chrono>
#include<functional>
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
struct Timer
{
	std::chrono::time_point<std::chrono::steady_clock> start, end;
	std::chrono::duration<float> duration;
	Timer()
	{
		start = std::chrono::high_resolution_clock::now();
	}
	~Timer()
	{
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		float ms = duration.count() * 1000.0f;
		std::cout << "Timer took" << ms << "ms" << std::endl;
	}
};
void testPerformance(int size,std::function<void(float**,int)> func)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	float** hello = nullptr;
	std::chrono::duration<float,std::milli> duration = end - start;
	std::cout << duration.count() << "ms" << std::endl;
}

int main()
{
	testPerformance(10, ordinarilyGauss);
}

