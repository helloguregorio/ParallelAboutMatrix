#pragma once
#include<chrono>
#include<iostream>
struct Timer
{
	std::chrono::time_point<std::chrono::steady_clock> start, end;
	std::chrono::duration<float> duration;
	int repeat;//repeat是实验重复次数
	Timer(int num) :repeat(num) {
		start = std::chrono::high_resolution_clock::now();
	}
	~Timer()
	{
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		float ms = duration.count() * 1000.0f / repeat;
		//std::cout << " Average time consume: " << ms << "ms" << std::endl;
		std::cout << ms << std::endl;
	}
};