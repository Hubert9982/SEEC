// Integer arithmetic coding for the ESC experiment. Zero-width intervals
// are permitted for symbols that are never encoded (the collapsed tail).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;
constexpr uint32_t HALF = 0x80000000u;
constexpr uint32_t QUARTER = 0x40000000u;
constexpr uint32_t THREE_QUARTERS = 0xc0000000u;
constexpr uint64_t TOTAL = 65536;
using Cdf = py::array_t<int32_t, py::array::c_style>;
using Symbols = py::array_t<int16_t, py::array::c_style>;

struct BitWriter {
    std::string bytes;
    uint8_t byte = 0;
    unsigned used = 0;
    void put(unsigned bit) {
        byte = static_cast<uint8_t>((byte << 1) | bit);
        if (++used == 8) {
            bytes.push_back(static_cast<char>(byte));
            byte = 0;
            used = 0;
        }
    }
    void finish() {
        if (used) bytes.push_back(static_cast<char>(byte << (8 - used)));
    }
};

struct BitReader {
    const std::string& bytes;
    size_t position = 0;
    unsigned get() {
        const size_t index = position / 8;
        const unsigned shift = 7 - position++ % 8;
        return index < bytes.size() ? (static_cast<uint8_t>(bytes[index]) >> shift) & 1 : 0;
    }
};

void check_shape(const Cdf& cdf) {
    if (cdf.ndim() != 2 || cdf.shape(1) < 2) throw std::invalid_argument("expected a 2D integer CDF");
}

py::bytes encode(const Cdf& cdf, const Symbols& symbols) {
    check_shape(cdf);
    if (symbols.ndim() != 1 || symbols.shape(0) != cdf.shape(0))
        throw std::invalid_argument("symbols and CDF shapes disagree");
    auto tables = cdf.unchecked<2>();
    auto values = symbols.unchecked<1>();
    uint32_t low = 0, high = 0xffffffffu;
    uint64_t pending = 0;
    BitWriter output;
    auto emit = [&](unsigned bit) {
        output.put(bit);
        while (pending) { output.put(1 - bit); --pending; }
    };
    for (py::ssize_t i = 0; i < symbols.shape(0); ++i) {
        const int symbol = values(i);
        if (symbol < 0 || symbol + 1 >= cdf.shape(1)) throw std::invalid_argument("symbol outside CDF");
        const int32_t begin = tables(i, symbol), end = tables(i, symbol + 1);
        if (begin < 0 || end <= begin || end > TOTAL) throw std::invalid_argument("invalid encoded interval");
        const uint64_t span = static_cast<uint64_t>(high) - low + 1;
        high = low + span * end / TOTAL - 1;
        low = low + span * begin / TOTAL;
        for (;;) {
            if (high < HALF) emit(0);
            else if (low >= HALF) {
                emit(1); low -= HALF; high -= HALF;
            } else if (low >= QUARTER && high < THREE_QUARTERS) {
                ++pending; low -= QUARTER; high -= QUARTER;
            } else break;
            low <<= 1;
            high = (high << 1) | 1;
        }
    }
    ++pending;
    emit(low < QUARTER ? 0 : 1);
    output.finish();
    return py::bytes(output.bytes);
}

Symbols decode(const Cdf& cdf, const std::string& bytes) {
    check_shape(cdf);
    auto tables = cdf.unchecked<2>();
    Symbols output(cdf.shape(0));
    auto values = output.mutable_unchecked<1>();
    uint32_t low = 0, high = 0xffffffffu, value = 0;
    BitReader input{bytes};
    for (unsigned i = 0; i < 32; ++i) value = (value << 1) | input.get();
    for (py::ssize_t i = 0; i < cdf.shape(0); ++i) {
        const uint64_t span = static_cast<uint64_t>(high) - low + 1;
        const uint32_t target = ((static_cast<uint64_t>(value) - low + 1) * TOTAL - 1) / span;
        // Upper-bound search, including duplicate entries, selects the
        // rightmost boundary <= target and skips every zero-width interval.
        py::ssize_t left = 0, right = cdf.shape(1) - 1;
        while (left + 1 < right) {
            const auto middle = (left + right) / 2;
            if (tables(i, middle) <= target) left = middle;
            else right = middle;
        }
        const int32_t begin = tables(i, left), end = tables(i, left + 1);
        if (begin < 0 || end <= begin || end > TOTAL) throw std::invalid_argument("invalid decoded interval");
        values(i) = static_cast<int16_t>(left);
        high = low + span * end / TOTAL - 1;
        low = low + span * begin / TOTAL;
        for (;;) {
            if (high < HALF) {}
            else if (low >= HALF) {
                low -= HALF; high -= HALF; value -= HALF;
            } else if (low >= QUARTER && high < THREE_QUARTERS) {
                low -= QUARTER; high -= QUARTER; value -= QUARTER;
            } else break;
            low <<= 1;
            high = (high << 1) | 1;
            value = (value << 1) | input.get();
        }
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("encode", &encode);
    module.def("decode", &decode);
}
