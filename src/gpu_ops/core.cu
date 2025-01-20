#include <stdio.h>
#include <gpu_ops/core.cuh>


__global__ void gpu_print() {
	printf("Hello, GPU-accelarated world!\n");
}


void test_gpu_print() {
	gpu_print<<<1, 1>>>();
}