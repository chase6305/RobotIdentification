#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <utility>

#include "robot_identification/identification.h"

#if defined(__unix__)
#include <sys/resource.h>

#include <csignal>
#endif

namespace {

struct Fixture {
    const std::string path = "robot_identification_csv_fixture.csv";
    ~Fixture() { std::remove(path.c_str()); }
    void Write(const std::string& text) const {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << text;
        output.close();
        if (!output) throw std::runtime_error("could not create CSV fixture");
    }
};

std::string Header(bool raw) {
    return raw ? "time_s,q/joint,tau/joint" : "time_s,q/joint,dq/joint,ddq/joint,tau/joint";
}

std::string Row(bool raw, const std::string& time, const std::string& position) {
    return time + "," + position + (raw ? ",3" : ",1,2,3");
}

void Read(bool raw, const std::string& path) {
    if (raw) {
        (void)robot_identification::ReadRawDatasetCsv(path);
    } else {
        (void)robot_identification::ReadDatasetCsv(path);
    }
}

void ExpectRejected(bool raw, const Fixture& fixture, const std::string& text) {
    fixture.Write(text);
    bool rejected = false;
    try {
        Read(raw, fixture.path);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("invalid CSV was accepted");
}

void TestNumericGrammar() {
    const Fixture fixture;
    for (bool raw : {false, true}) {
        for (const std::string token :
             {"0", "-0", " +1.25", "\t-2.5e+2", "\r\f\v1.25", "0x1.8p+2", "-0x.8", " +0X1.8P+2", "0x1e2", "1.", ".5",
              "0e-9999", "-0x0p-9999", "1.7976931348623157e308", "2.2250738585072014e-308"}) {
            const double expected = std::stod(token);
            for (const std::string newline : {"\n", "\r\n"}) {
                // Blank rows and a final row without a newline remain valid.
                fixture.Write(Header(raw) + newline + Row(raw, "0", token) + newline + newline + Row(raw, "1", token) +
                              newline + Row(raw, "2", token));
                double value = 0;
                if (raw) {
                    const auto dataset = robot_identification::ReadRawDatasetCsv(fixture.path);
                    if (dataset.time_seconds.size() != 3) throw std::runtime_error("raw CSV lost rows");
                    value = dataset.position(2, 0);
                } else {
                    const auto dataset = robot_identification::ReadDatasetCsv(fixture.path);
                    if (dataset.samples.size() != 3) throw std::runtime_error("processed CSV lost rows");
                    value = dataset.samples.back().position[0];
                }
                if (value != expected || std::signbit(value) != std::signbit(expected)) {
                    throw std::runtime_error("CSV changed a supported numeric value");
                }
            }
        }
        for (const std::string& token : std::vector<std::string>{"",        " ",         "1 ",
                                                                 "1junk",   "1e",        "nan",
                                                                 "inf",     "-infinity", "1e9999",
                                                                 "1e-9999", "\"1\"",     "1,2",
                                                                 "--1",     "+-1",       "-+1",
                                                                 "++1",     "0x-1",      "0x+1",
                                                                 "-0x-1",   "0x",        "0x1p",
                                                                 "0x1e+2",  "0x1p+1024", "0x1p-1075",
                                                                 "1_000",   "１２",      std::string("1\0junk", 6)}) {
            ExpectRejected(raw, fixture,
                           Header(raw) + "\n" + Row(raw, "0", token) + "\n" + Row(raw, "1", "2") + "\n" +
                               Row(raw, "2", "3") + "\n");
        }
        const std::string valid_prefix = Header(raw) + "\n" + Row(raw, "0", "1") + "\n" + Row(raw, "1", "2") + "\n";
        for (const std::string& malformed :
             std::vector<std::string>{Row(raw, "2", "3") + ",", Row(raw, "2", "3") + ",4", "2,3"}) {
            ExpectRejected(raw, fixture, valid_prefix + malformed);
        }
        ExpectRejected(raw, fixture, valid_prefix + Row(raw, "1", "3"));
        ExpectRejected(raw, fixture, Header(raw) + "\n\r\n");
        ExpectRejected(raw, fixture, "time_s,q/a,tau/b\n0,1,2\n1,2,3\n2,3,4\n");
    }
}

void TestTimestampDiagnostics() {
    const Fixture fixture;
    for (bool raw : {false, true}) {
        for (const std::string timestamp : {"0", "-1"}) {
            // The invalid timestamp must be diagnosed before a later malformed
            // numeric row, and blank lines must count toward the physical line.
            fixture.Write(Header(raw) + "\n" + Row(raw, "0", "1") + "\n\n" + Row(raw, timestamp, "2") + "\n" +
                          Row(raw, "2", "bad"));
            bool rejected = false;
            try {
                Read(raw, fixture.path);
            } catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("timestamp must be strictly increasing at line 4") !=
                           std::string::npos;
            }
            if (!rejected) throw std::runtime_error("CSV did not locate the first invalid timestamp");
        }
        fixture.Write(Header(raw) + "\r\n" + Row(raw, "-3", "1") + "\r\n\r\n" + Row(raw, "-2", "2") + "\r\n" +
                      Row(raw, "-1", "3"));
        Read(raw, fixture.path);
    }
}

void TestHeaderNamesBeforeRows() {
    const Fixture fixture;
    for (bool raw : {false, true}) {
        for (const auto& names :
             {std::vector<std::string>{"same", "same"}, std::vector<std::string>{"a\rb", "other"}}) {
            std::string header = "time_s";
            for (const auto& prefix :
                 raw ? std::vector<std::string>{"q/", "tau/"} : std::vector<std::string>{"q/", "dq/", "ddq/", "tau/"}) {
                for (const auto& name : names) header += "," + prefix + name;
            }
            fixture.Write(header + "\ninvalid-data-must-not-be-read\n");
            bool rejected = false;
            try {
                Read(raw, fixture.path);
            } catch (const std::runtime_error& error) {
                const std::string message = error.what();
                // A CR inside a name may end a malformed CR-style header.
                // Both diagnoses must reject it before parsing the bad row.
                rejected =
                    message.find("CSV joint names") != std::string::npos || message.find("header") != std::string::npos;
            }
            if (!rejected) throw std::runtime_error("invalid CSV header names reached row parsing");
        }
    }
}

void RequireSameValue(double actual, double expected) {
    if (actual != expected || std::signbit(actual) != std::signbit(expected)) {
        throw std::runtime_error("CSV did not preserve a finite value exactly");
    }
}

void CheckRoundTrip(const Fixture& fixture, const std::vector<double>& values) {
    robot_identification::RawDataset raw;
    raw.joint_names = {"joint"};
    raw.time_seconds.resize(values.size());
    raw.position.resize(values.size(), 1);
    raw.torque.resize(values.size(), 1);
    robot_identification::Dataset processed;
    processed.joint_names = raw.joint_names;
    for (std::size_t index = 0; index < values.size(); ++index) {
        raw.time_seconds[index] = index * 0.125;
        raw.position(index, 0) = values[index];
        raw.torque(index, 0) = -values[index];
        processed.samples.push_back({raw.time_seconds[index], Eigen::VectorXd::Constant(1, values[index]),
                                     Eigen::VectorXd::Constant(1, -values[index]),
                                     Eigen::VectorXd::Constant(1, values[index]),
                                     Eigen::VectorXd::Constant(1, -values[index])});
    }
    robot_identification::WriteRawDatasetCsv(raw, fixture.path);
    const auto restored_raw = robot_identification::ReadRawDatasetCsv(fixture.path);
    if (restored_raw.joint_names != raw.joint_names || restored_raw.time_seconds.size() != raw.time_seconds.size()) {
        throw std::runtime_error("raw round trip changed dimensions");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        RequireSameValue(restored_raw.time_seconds[index], raw.time_seconds[index]);
        RequireSameValue(restored_raw.position(index, 0), values[index]);
        RequireSameValue(restored_raw.torque(index, 0), -values[index]);
    }
    robot_identification::WriteDatasetCsv(processed, fixture.path);
    const auto restored = robot_identification::ReadDatasetCsv(fixture.path);
    if (restored.joint_names != processed.joint_names || restored.samples.size() != values.size()) {
        throw std::runtime_error("processed round trip changed dimensions");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto& sample = restored.samples[index];
        RequireSameValue(sample.time_seconds, raw.time_seconds[index]);
        RequireSameValue(sample.position[0], values[index]);
        RequireSameValue(sample.velocity[0], -values[index]);
        RequireSameValue(sample.acceleration[0], values[index]);
        RequireSameValue(sample.torque[0], -values[index]);
    }
}

void TestFiniteRangeRoundTrip() {
    const Fixture fixture;
    std::vector<double> values{0.0,
                               -0.0,
                               std::numeric_limits<double>::denorm_min(),
                               std::nextafter(std::numeric_limits<double>::min(), 0.0),
                               std::numeric_limits<double>::min(),
                               std::numeric_limits<double>::max(),
                               1e-310};
    // Fixed bit patterns cover every finite exponent without an RNG library.
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    std::uint64_t bits = 0x9e3779b97f4a7c15ULL;
    for (int index = 0; index < 4096; ++index) {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        const std::uint64_t pattern = (bits & 0x800fffffffffffffULL) | (std::uint64_t(index % 2047) << 52);
        double value = 0;
        std::memcpy(&value, &pattern, sizeof(value));
        if (!std::isfinite(value)) throw std::runtime_error("finite bit-pattern fixture is invalid");
        values.push_back(value);
    }
    CheckRoundTrip(fixture, values);
    for (bool raw : {false, true}) {
        for (const std::string token : {"4.9406564584124654e-324", "0x1p-1074", "+0X1P-1074", "-0x1p-1074"}) {
            fixture.Write(Header(raw) + "\n" + Row(raw, "0", token) + "\n" + Row(raw, "1", token) + "\n" +
                          Row(raw, "2", token));
            const double expected = (token.front() == '-' ? -1.0 : 1.0) * std::numeric_limits<double>::denorm_min();
            const double actual = raw ? robot_identification::ReadRawDatasetCsv(fixture.path).position(0, 0)
                                      : robot_identification::ReadDatasetCsv(fixture.path).samples[0].position[0];
            RequireSameValue(actual, expected);
        }
    }
}

class CommaPunctuation : public std::numpunct<char> {
protected:
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};

class RestoreLocales {
public:
    RestoreLocales() : cpp_locale_(std::locale()), numeric_locale_(std::setlocale(LC_NUMERIC, nullptr)) {}
    ~RestoreLocales() {
        std::locale::global(cpp_locale_);
        std::setlocale(LC_NUMERIC, numeric_locale_.c_str());
    }

private:
    std::locale cpp_locale_;
    std::string numeric_locale_;
};

void TestLocaleIndependence() {
    const RestoreLocales restore;
    // The custom C++ locale always exercises decimal and grouping changes.
    std::locale::global(std::locale(std::locale::classic(), new CommaPunctuation));
    // Also exercise the independent C numeric locale when installed on the host.
    for (const char* name : {"en_DK.utf8", "de_DE.UTF-8", "fr_FR.UTF-8"}) {
        if (std::setlocale(LC_NUMERIC, name) && std::string(std::localeconv()->decimal_point) == ",") break;
    }
    const Fixture fixture;
    for (bool raw : {false, true}) {
        fixture.Write(Header(raw) + "\n" + Row(raw, "0", "1.25") + "\n" + Row(raw, "1", "2.5") + "\n" +
                      Row(raw, "2", "3.75"));
        const double actual = raw ? robot_identification::ReadRawDatasetCsv(fixture.path).position(0, 0)
                                  : robot_identification::ReadDatasetCsv(fixture.path).samples[0].position[0];
        RequireSameValue(actual, 1.25);
    }
    CheckRoundTrip(fixture, {1234.125, -1234567.75, 1e-310});
    std::ifstream input(fixture.path);
    std::string header;
    std::string row;
    std::getline(input, header);
    std::getline(input, row);
    if (row != "0,1234.125,-1234.125,1234.125,-1234.125") {
        throw std::runtime_error("numeric locale changed the CSV output bytes");
    }
}

void TestRangeErrorLocation() {
    const Fixture fixture;
    for (bool raw : {false, true}) {
        fixture.Write(Header(raw) + "\n\n" + Row(raw, "0", "1e-9999") + "\n" + Row(raw, "1", "2") + "\n" +
                      Row(raw, "2", "3"));
        bool located = false;
        try {
            Read(raw, fixture.path);
        } catch (const std::out_of_range& error) {
            located = std::string(error.what()).find("line 3, column 2") != std::string::npos;
        }
        if (!located) throw std::runtime_error("CSV range error lost its source location");
    }
}

void TestMultipleJoints() {
    const Fixture fixture;
    for (int dof : {1, 2, 9}) {
        robot_identification::RawDataset raw;
        robot_identification::Dataset processed;
        for (int joint = 0; joint < dof; ++joint) raw.joint_names.push_back("关节" + std::to_string(joint));
        processed.joint_names = raw.joint_names;
        constexpr int count = 257;
        raw.time_seconds.resize(count);
        raw.position.resize(count, dof);
        raw.torque.resize(count, dof);
        for (int row = 0; row < count; ++row) {
            raw.time_seconds[row] = row * 0.03125;
            for (int joint = 0; joint < dof; ++joint) {
                raw.position(row, joint) = std::sin(row * 0.13 + joint);
                raw.torque(row, joint) = row * dof + joint;
            }
            processed.samples.push_back({raw.time_seconds[row], raw.position.row(row).transpose(),
                                         Eigen::VectorXd::Constant(dof, row * 0.125),
                                         Eigen::VectorXd::Constant(dof, -row * 0.25), raw.torque.row(row).transpose()});
        }
        robot_identification::WriteRawDatasetCsv(raw, fixture.path);
        const auto restored = robot_identification::ReadRawDatasetCsv(fixture.path);
        if (restored.joint_names != raw.joint_names || !restored.time_seconds.isApprox(raw.time_seconds, 0) ||
            !restored.position.isApprox(raw.position, 0) || !restored.torque.isApprox(raw.torque, 0)) {
            throw std::runtime_error("raw CSV changed joint order or matrix layout");
        }
        robot_identification::WriteDatasetCsv(processed, fixture.path);
        const auto restored_processed = robot_identification::ReadDatasetCsv(fixture.path);
        if (restored_processed.joint_names != raw.joint_names || restored_processed.samples.size() != count) {
            throw std::runtime_error("processed CSV changed joint order or sample count");
        }
        for (int row = 0; row < count; ++row) {
            const auto& actual = restored_processed.samples[row];
            const auto& expected = processed.samples[row];
            if (actual.time_seconds != expected.time_seconds || !actual.position.isApprox(expected.position, 0) ||
                !actual.velocity.isApprox(expected.velocity, 0) ||
                !actual.acceleration.isApprox(expected.acceleration, 0) ||
                !actual.torque.isApprox(expected.torque, 0)) {
                throw std::runtime_error("processed CSV changed signal group layout");
            }
        }
    }
}

#if defined(__unix__)
class RejectFileWrites {
public:
    RejectFileWrites() {
        if (getrlimit(RLIMIT_FSIZE, &previous_) != 0) throw std::runtime_error("cannot read file size limit");
        previous_handler_ = std::signal(SIGXFSZ, SIG_IGN);
        if (previous_handler_ == SIG_ERR) throw std::runtime_error("cannot ignore file size signal");
        auto limited = previous_;
        limited.rlim_cur = 0;
        if (setrlimit(RLIMIT_FSIZE, &limited) != 0) {
            std::signal(SIGXFSZ, previous_handler_);
            throw std::runtime_error("cannot set file size limit");
        }
    }
    ~RejectFileWrites() {
        if (setrlimit(RLIMIT_FSIZE, &previous_) != 0) std::terminate();
        std::signal(SIGXFSZ, previous_handler_);
    }

private:
    struct rlimit previous_ {};
    void (*previous_handler_)(int) = SIG_DFL;
};

void TestFinalWriteFailure() {
    const Fixture fixture;
    for (bool raw : {false, true}) {
        fixture.Write(Header(raw) + "\n" + Row(raw, "0", "1") + "\n" + Row(raw, "1", "2") + "\n" + Row(raw, "2", "3"));
        const auto raw_data =
            raw ? robot_identification::ReadRawDatasetCsv(fixture.path) : robot_identification::RawDataset{};
        const auto processed =
            raw ? robot_identification::Dataset{} : robot_identification::ReadDatasetCsv(fixture.path);
        bool reported_write_failure = false;
        {
            const RejectFileWrites limit;
            try {
                if (raw) {
                    robot_identification::WriteRawDatasetCsv(raw_data, fixture.path);
                } else {
                    robot_identification::WriteDatasetCsv(processed, fixture.path);
                }
            } catch (const std::runtime_error& error) {
                reported_write_failure = std::string(error.what()).find("failed while writing") != std::string::npos;
            }
        }
        if (!reported_write_failure) throw std::runtime_error("buffered CSV write failure was silently ignored");
    }
}
#endif

void TestLargeRawStorageBoundaries() {
    const Fixture fixture;
    for (const auto dimensions : {std::make_pair(14, 10003), std::make_pair(257, 1029)}) {
        const auto [dof, rows] = dimensions;
        robot_identification::RawDataset original;
        for (int joint = 0; joint < dof; ++joint) original.joint_names.push_back("axis_" + std::to_string(joint));
        original.time_seconds.resize(rows);
        original.position.resize(rows, dof);
        original.torque.resize(rows, dof);
        for (int row = 0; row < rows; ++row) {
            original.time_seconds[row] = row * .015625;
            for (int joint = 0; joint < dof; ++joint) {
                original.position(row, joint) = row * .125 - joint * .25;
                original.torque(row, joint) = -row * .5 + joint * .0625;
            }
        }
        original.position(0, 0) = original.torque(0, 0) = -0.;
        robot_identification::WriteRawDatasetCsv(original, fixture.path);
        const auto actual = robot_identification::ReadRawDatasetCsv(fixture.path);
        const auto same_bits = [](const auto& left, const auto& right) {
            return left.rows() == right.rows() && left.cols() == right.cols() &&
                   std::memcmp(left.data(), right.data(), static_cast<std::size_t>(left.size()) * sizeof(double)) == 0;
        };
        if (actual.joint_names != original.joint_names || !same_bits(actual.time_seconds, original.time_seconds) ||
            !same_bits(actual.position, original.position) || !same_bits(actual.torque, original.torque)) {
            throw std::runtime_error("large raw CSV lost/reordered rows, columns, or signed zero");
        }
        {
            std::ofstream append(fixture.path, std::ios::app);
            append << "100000,nan";
            for (int column = 1; column < 2 * dof; ++column) append << ",0";
            append << '\n';
        }
        bool rejected = false;
        try {
            (void)robot_identification::ReadRawDatasetCsv(fixture.path);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("late malformed row bypassed raw CSV validation");
    }
}

}  // namespace

int main() {
    TestLargeRawStorageBoundaries();
    TestHeaderNamesBeforeRows();
    TestTimestampDiagnostics();
    TestNumericGrammar();
    TestFiniteRangeRoundTrip();
    TestLocaleIndependence();
    TestRangeErrorLocation();
    TestMultipleJoints();
#if defined(__unix__)
    TestFinalWriteFailure();
#endif
}
