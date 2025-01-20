#include <string_view>
#include <stdexcept>

using namespace std;


namespace ise {

enum class ParameterCode {
	reserved = 0,     // reserved
    mode,             // mode
    step,             // decimation step
    number_all,       // number of samples
    number_after,     // number of samples after decimation
    first_delay,      // first sample delay relative to Tk0, us
    freq_code,        // frequency code
    channel,          // channel number
    data_type,        // Data type: actual data or calibration signal
    date_year,        // date: year
    date_mon_day,     // date: month (2-nd byte) / day (1-st byte)
    time_h_m,         // time: hour (1-st byte) / minute (2-nd byte)
    time_sec,         // time: sec
    time_msec,        // time: msec
    st1_long_fr_lo,   // STEL1 long pulse frequency, 1-2 bytes
    st1_long_fr_hi,   // STEL1 long pulse frequency, 3-4 bytes
    st1_short_fr_lo,  // STEL1 short pulse frequency, 1-2 bytes
    st1_short_fr_hi,  // STEL1 short pulse frequency, 3-4 bytes
    st2_long_fr_lo,   // STEL2 long pulse frequency, 1-2 bytes
    st2_long_fr_hi,   // STEL2 long pulse frequency, 3-4 bytes
    st2_short_fr_lo,  // STEL2 short pulse frequency, 1-2 bytes
    st2_short_fr_hi,  // STEL2 short pulse frequency, 3-4 bytes
    st1_long_len,     // STEL1 long pulse length
    st1_short_len,    // STEL1 short pulse length
    st2_long_len,     // STEL2 long pulse length
    st2_short_len,    // STEL2 short pulse length
    st1_long_phase,   // STEL1 long pulse phase modulation
    st1_short_phase,  // STEL1 short pulse phase modulation
    st2_long_phase,   // STEL2 long pulse phase modulation
    st2_short_phase,  // STEL2 short pulse phase modulation
    sample_freq,      // sampling frequency, kHz
    average,          // steady component mean
    phase_code,       // code of phase manipulation
    offset_st1,       // bias
    timer_lo,         // timer 1-2 bytes (unused here)
    timer_hi,         // timer 3-4 bytes (unused here)
    version           // ISE file version
};


enum class HeaderCode {
    SUPER = 1,
    DATA = 2,
    GLOBGAL = 3
};


enum class DataType {
    DATA = 1,
    CALIBRATION = 2
};


class DataAccess {
private:
    DataSource reader;
};


enum class ErrorPolicy {
    kRaise = 0,
    kWarn
};


class ErrorHandler {
public:
    ErrorHandler(ErrorPolicy policy) : policy_(policy) {}

    void handle_error(string_view msg) {
        if (policy_ == ErrorPolicy::kRaise) {
            throw runtime_error(msg);
        } else if (policy_ == ErrorPolicy::kWarn) {
            cout << msg << endl;
        } else {
            throw invalid_argument("Unexpected error policy");
        }
    }

private:
    ErrorPolicy policy_;
};


class IseHeaderReader {
public:
    IseHeaderReader(IseIOStream stream, bool conjugate_quadratures = true, 
        ErrorPolicy error_policy = ErrorPolicy::kRaise)
        : stream_(stream)
        , conjugate_quadratures_(conjugate_quadratures)
        , error_handler_(ErrorHandler(error_policy))
    {}



private:
    IseIOStream stream_;
    bool conjugate_quadratures_;
    ErrorHandler error_handler_;
};

}  // namespace