#include <complex>
#include <vector>
#include <cmath>
#include <type_traits>

#include <gtest/gtest.h>
#include <fftw3.h>
#include <cufft.h>

#include "gpu_ops/tests/basic_chirp_fft.cuh"
#include "gpu_ops/utils.cuh"


template <typename T>
::testing::AssertionResult isClose(T val1, T val2, T rel_err, T abs_err) {
    T diff = std::abs(val1 - val2);
    T lowest = val1 < val2 ? val1 : val2;
    T rel_diff = diff / lowest;
    if ((rel_diff < rel_err) || (diff < abs_err)) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure()
            << "\nThe relative difference is " << rel_diff << " vs " << rel_err
            << "\nThe absolute difference is " << diff <<  " vs " << abs_err
            << "\nval1 = " << val1 << "\n" << "val2 = " << val2 << "\n";
    }
}

using float_type_list = ::testing::Types<float, double>;
template <typename> struct TypedTestSuite : testing::Test {};
TYPED_TEST_SUITE(TypedTestSuite, float_type_list);


TYPED_TEST(TypedTestSuite, TestIsClose) {
    EXPECT_TRUE(isClose<TypeParam>(0.0001, 0.0, 1e-3, 1e-3));
    EXPECT_FALSE(isClose<TypeParam>(0.01, 0.0, 1e-3, 1e-3));
    EXPECT_TRUE(isClose<TypeParam>(1000.0, 1000.1, 1e-3, 1e-3));
    EXPECT_FALSE(isClose<TypeParam>(1000.0, 1002.0, 1e-3, 1e-3));
}


template <typename T>
using typed_fftw_complex = std::conditional_t<std::is_same_v<T, float>, fftwf_complex, fftw_complex>;
template <typename T>
using typed_fftw_plan = std::conditional_t<std::is_same_v<T, float>, fftwf_plan, fftw_plan>;
fftw_plan typed_fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags) {
    return fftw_plan_dft_1d(n, in, out, sign, flags);
}
fftwf_plan typed_fftw_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags) {
    return fftwf_plan_dft_1d(n, in, out, sign, flags);
}
void typed_fftw_execute(const fftw_plan p) {
    fftw_execute(p);
}
void typed_fftw_execute(const fftwf_plan p) {
    fftwf_execute(p);
}
void typed_fftw_execute_dft(const fftw_plan p, fftw_complex* in, fftw_complex* out) {
    fftw_execute_dft(p, in, out);
}
void typed_fftw_execute_dft(const fftwf_plan p, fftwf_complex* in, fftwf_complex* out) {
    fftwf_execute_dft(p, in, out);
}
void typed_fftw_destroy_plan(fftw_plan p) {
    fftw_destroy_plan(p);
}
void typed_fftw_destroy_plan(fftwf_plan p) {
    fftwf_destroy_plan(p);
}


TYPED_TEST(TypedTestSuite, TestFFT) {
    size_t n_fft = 16;

    using float_t = TypeParam;
    using data_t = std::complex<float_t>;
    std::vector<data_t> series(n_fft);
    std::vector<data_t> out_cufft(n_fft);

    for (size_t i = 0; i < n_fft; ++i) {
        float_t v = static_cast<float_t>(i);
        series[i] = data_t(v, -v);
    }

    simpleFFT1d(series, out_cufft);

    // Check against FFTW3
    std::vector<data_t> out_fftw3(n_fft);
    typed_fftw_plan<TypeParam> p;
    p = typed_fftw_plan_dft_1d(
        n_fft,
        reinterpret_cast<typed_fftw_complex<TypeParam>*>(series.data()),
        reinterpret_cast<typed_fftw_complex<TypeParam>*>(out_fftw3.data()),
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    typed_fftw_execute(p);
    typed_fftw_destroy_plan(p);

    for (size_t i = 0; i < n_fft; ++i) {
        EXPECT_TRUE(isClose(out_cufft[i].real(), out_fftw3[i].real(), float_t(1e-3), float_t(1e-5))) 
                        << "Vectors differ at [" << i << "]";
        EXPECT_TRUE(isClose(out_cufft[i].imag(), out_fftw3[i].imag(), float_t(1e-3), float_t(1e-5))) 
                        << "Vectors differ at [" << i << "]";
    }
}


TYPED_TEST(TypedTestSuite, TestMovingFFT) {
    size_t n_series = 2048;
    size_t n_fft = 1024;
    size_t n_wins = n_series - n_fft + 1;
    size_t expected_out_size = n_wins * n_fft;

    using float_t = TypeParam;
    using data_t = std::complex<float_t>;
    std::vector<data_t> series(n_series);

    for (size_t i = 0; i < n_series; ++i) {
        float_t v = static_cast<float_t>(i);
        series[i] = data_t(v, -v);
    }

    std::vector<data_t> out_cufft = movingFFT1d(series, n_fft);
    ASSERT_EQ(out_cufft.size(), expected_out_size) << "Output array has incorrect size";

    // Check against FFTW3
    std::vector<data_t> out_fftw3(expected_out_size);

    // Static cast 2nd and 3rd argument to resolve the overloaded function
    typed_fftw_plan<TypeParam> p = typed_fftw_plan_dft_1d(
        n_fft,
        static_cast<typed_fftw_complex<TypeParam>*>(NULL),
        static_cast<typed_fftw_complex<TypeParam>*>(NULL),
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    for (size_t i = 0; i < n_wins; ++i) {
        typed_fftw_execute_dft(
            p,
            reinterpret_cast<typed_fftw_complex<TypeParam>*>(series.data() + i),
            reinterpret_cast<typed_fftw_complex<TypeParam>*>(out_fftw3.data() + i * n_fft)
        );
    }
    typed_fftw_destroy_plan(p);

    for (size_t i = 0; i < expected_out_size; ++i) {
        EXPECT_TRUE(isClose(out_cufft[i].real(), out_fftw3[i].real(), float_t(7e-2), float_t(2e-2)))
                        << "Vectors differ at [" << i << "]";
        EXPECT_TRUE(isClose(out_cufft[i].imag(), out_fftw3[i].imag(), float_t(7e-2), float_t(2e-2)))
                        << "Vectors differ at [" << i << "]";
    }
}


TYPED_TEST(TypedTestSuite, TestChirpFFT) {
    size_t n_series = 2048;
    size_t n_fft = 1024;
    size_t n_wins = n_series - n_fft + 1;
    size_t expected_out_size = n_wins * n_fft;

    using float_t = TypeParam;
    using data_t = std::complex<float_t>;
    std::vector<data_t> series(n_series);
    std::vector<data_t> chirp(n_fft);

    for (size_t i = 0; i < n_series; ++i) {
        float_t v = static_cast<float_t>(i) / n_series;
        series[i] = data_t(v, -v);
        if (i < n_fft) {
            chirp[i] = data_t(v, -v);
        }
    }

    std::vector<data_t> out_cufft = chirpFFT(series, chirp);
    ASSERT_EQ(out_cufft.size(), expected_out_size) << "Output array has incorrect size";

    // Calculate with FFTW
    // Static cast 2nd and 3rd argument to resolve the overloaded function
    typed_fftw_plan<TypeParam> p = typed_fftw_plan_dft_1d(
        n_fft,
        static_cast<typed_fftw_complex<TypeParam>*>(NULL),
        static_cast<typed_fftw_complex<TypeParam>*>(NULL),
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    std::vector<data_t> tmp(n_fft);
    std::vector<data_t> out_fftw3(expected_out_size);
    for (size_t i = 0; i < n_wins; ++i) {
        for (size_t j = 0; j < n_fft; ++j) {
            tmp[j] = series[i + j] * std::conj(chirp[j]);
        }
        typed_fftw_execute_dft(
            p,
            reinterpret_cast<typed_fftw_complex<TypeParam>*>(tmp.data()),
            reinterpret_cast<typed_fftw_complex<TypeParam>*>(out_fftw3.data() + i * n_fft)
        );
    }
    typed_fftw_destroy_plan(p);

    for (size_t i = 0; i < expected_out_size; ++i) {
        EXPECT_TRUE(isClose(out_cufft[i].real(), out_fftw3[i].real(), float_t(5e-2), float_t(1e-5)))
                        << "Vectors differ at [" << i << "]";
        EXPECT_TRUE(isClose(out_cufft[i].imag(), out_fftw3[i].imag(), float_t(5e-2), float_t(1e-5)))
                        << "Vectors differ at [" << i << "]";
    }
}
