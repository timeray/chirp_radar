#pragma once

#include <sstream>
#include <stdexcept>
#include <cufft.h>


void check_cuda_error(cudaError_t err) {
	if (err != cudaSuccess) {
		std::ostringstream buf;
		buf << "CUDA ERROR: ";
		auto res = cudaGetErrorString(err); 
		buf << res;
		buf << '\n';
		throw std::runtime_error(buf.str());
	}
}

void check_cufft_error(cufftResult_t res) {
	if (res != CUFFT_SUCCESS) {
		std::stringstream buf;
		buf << "CUFFT ERROR: error code ";
		buf << res;
		buf << '\n';
		throw std::runtime_error(buf.str());
	}
}