#include <complex>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <iosfwd>
#include <optional>
#include <string_view>
#include <stdexcept>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <array>
#include <span>


namespace ise {

constexpr std::array<std::byte, 4> KEYWORD = {
    std::byte{'O'}, std::byte{'R'}, std::byte{'D'}, std::byte{'A'}
};

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
    ErrorHandler(ErrorPolicy policy) : m_policy(policy) {}

    void handle_error(std::string_view msg) {
        if (m_policy == ErrorPolicy::kRaise) {
            throw std::runtime_error(msg);
        } else if (m_policy == ErrorPolicy::kWarn) {
            std::cerr << msg << std::endl;
        } else {
            throw std::invalid_argument("Unexpected error policy");
        }
    }

private:
    ErrorPolicy m_policy;
};


struct ParamsHeaderValue {
    std::unordered_map<ParameterCode, uint16_t> params;
};

struct DataHeaderValue {
    std::vector<std::complex<float>> data;
};


class Header {
    HeaderCode m_code;
    size_t m_length;
    std::variant<ParamsHeaderValue, DataHeaderValue> m_value;
};


class AbstractIseStream {
public:
    virtual ~AbstractIseStream() {}

    // Read the next header
    virtual std::optional<Header> read_header() = 0;
};


class IseFileStream : public AbstractIseStream {
public:
    IseFileStream(std::span<std::string_view> paths, ErrorPolicy policy)
    : m_paths(paths)
    , m_policy(policy) {
        read_next_file();
    }

    std::optional<Header> read_header() override {
        while (true) {
            advance_to_keyword();
            if (m_pos >= m_size && !read_next_file()) {
                // No more files to read
                return std::nullopt;
            }
        }
    }

private:
    std::span<std::string_view> m_paths;
    ErrorPolicy m_policy;
    size_t m_path_idx = 0;
    std::ifstream m_ifstream;
    size_t m_size = 0;
    size_t m_pos = 0;
    std::vector<std::byte> m_buffer;

    void advance_to_keyword() {
        // Search for the keyword in the buffer
        auto result = std::ranges::search(m_buffer.begin() + m_pos, m_buffer.end(),
                                          KEYWORD.begin(), KEYWORD.end());
        if (result != m_buffer.end()) {
            m_pos = result.begin() - m_buffer.begin() + KEYWORD.size();
        } else {
            m_pos = m_size;
        }
    }

    std::span<const std::byte> read(size_t n_bytes) override {
        // Check if we've reached the end of the file
        if ((m_pos + n_bytes) >= m_size) {
            return std::span<const std::byte>();
        }

        // Search for the keyword in the buffer
        auto result = std::ranges::search(m_buffer, KEYWORD);
        if (result == m_buffer.end()) {
            m_pos = m_size;
            return std::span<const std::byte>();
        }

        // Advance the position to the end of the keyword
        m_pos = static_cast<std::streampos>(result - m_buffer.begin() + KEYWORD.size());
        return std::span<const std::byte>(m_buffer);
    }

    bool read_next_file() {
        if (m_path_idx >= m_paths.size()) {
            return false;
        }
        if (m_path_idx != 0) {
            m_ifstream.close();
        }
        m_ifstream.open(m_paths[m_path_idx], std::ios::binary | std::ios::ate);
        m_size = static_cast<size_t>(m_ifstream.tellg());
        m_ifstream.seekg(0);
        m_buffer.resize(m_size);
        m_ifstream.read(reinterpret_cast<char*>(m_buffer.data()),
                        static_cast<std::streamsize>(m_size));
        m_path_idx++;
        return true;
    }
};


class IseMulticastStream : public AbstractIseStream {
public:
    IseMulticastStream(MulticastParams params) : m_params(params) {}

    std::span<const uint8_t> advanced_read() override {

    }

private:
    MulticastParams m_params;
};


class IseHeaderReader {
public:
    IseHeaderReader(IseIOStream stream, ErrorPolicy error_policy = ErrorPolicy::kRaise)
        : m_stream(stream)
        , m_error_handler(ErrorHandler(error_policy))
    {}

    Header read_header() {
        while (true) {
            auto buffer = m_stream.read_chunk(9);
            if (buffer.size() != 9) {
                throw std::runtime_error("Header size mismatch");
            }
            auto result = std::ranges::search(buffer, KEYWORD);
            if (result != buffer.end()) {
                return Header();
            }
        }
    }

private:
    IseIOStream m_stream;
    ErrorHandler m_error_handler;
};

}  // namespace
