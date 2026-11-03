#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "name_validation.h"
#include "robot_identification/identification.h"

namespace robot_identification {
namespace {

bool ReadCsvHeader(std::istream& input, std::string& line, char& separator) {
    line.clear();
    separator = '\n';
    // The header only needs byte delimiters. Use the existing file buffer
    // without constructing an istream sentry for every joint-name byte.
    auto* buffer = input.rdbuf();
    using Traits = std::char_traits<char>;
    while (true) {
        const auto value = buffer->sbumpc();
        if (Traits::eq_int_type(value, Traits::eof())) {
            input.setstate(std::ios::eofbit | std::ios::failbit);
            return !line.empty();
        }
        const char character = Traits::to_char_type(value);
        if (character == '\n') return true;
        if (character == '\r') {
            if (Traits::eq_int_type(buffer->sgetc(), Traits::to_int_type('\n'))) {
                buffer->sbumpc();
            } else {
                separator = '\r';
            }
            return true;
        }
        line.push_back(character);
    }
}

std::vector<std::string_view> SplitCsvHeader(const std::string& line) {
    std::vector<std::string_view> fields;
    // Views borrow the header until CsvInput finishes validating it. Keep
    // an empty field after a final comma.
    if (line.empty()) return fields;
    for (std::size_t begin = 0;;) {
        const auto end = line.find(',', begin);
        fields.emplace_back(line.data() + begin, end == std::string::npos ? line.size() - begin : end - begin);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return fields;
}

void ValidateCsvJointNames(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        if (name.empty() || name.find_first_of(",\r\n") != std::string::npos) {
            throw std::runtime_error("CSV joint names must be non-empty and contain no commas or newlines");
        }
    }
    if (!detail::HasUniqueNonemptyNames(names)) {
        throw std::runtime_error("CSV joint names must be unique");
    }
}

std::ofstream OpenCsvOutput(const std::string& path,
                            const std::vector<std::string>& joint_names,
                            std::initializer_list<const char*> prefixes,
                            const char* kind) {
    if (path.empty()) throw std::invalid_argument(std::string(kind) + " output path must not be empty");
    for (const auto& name : joint_names)
        if (name.find_first_of(",\r\n") != std::string::npos)
            throw std::invalid_argument("joint names in CSV must not contain commas or newlines");
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open " + std::string(kind) + " output: " + path);
    output.imbue(std::locale::classic());
    output << "time_s";
    for (const char* prefix : prefixes)
        for (const auto& name : joint_names) output << ',' << prefix << name;
    output << '\n' << std::setprecision(17);
    return output;
}

bool ParseCsvRow(std::string& line, std::vector<double>& values, std::size_t line_number, double& previous_time) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const char* cursor = line.data();
    const char* end = cursor + line.size();
    if (cursor == end) return false;

    // from_chars bounds conversion to each field without changing the buffer
    // or consulting the process locale. Preserve leading ASCII whitespace,
    // an explicit plus sign, and the existing 0x-prefixed hexadecimal syntax.
    std::size_t column = 0;
    while (true) {
        if (column == values.size()) {
            throw std::runtime_error("dataset row has the wrong field count at line " + std::to_string(line_number));
        }
        const char* field_end = std::find(cursor, end, ',');
        const char* number = cursor;
        while (number != field_end && (*number == ' ' || (*number >= '\t' && *number <= '\r'))) ++number;
        const bool negative = number != field_end && *number == '-';
        if (number != field_end && (*number == '+' || *number == '-')) ++number;
        auto format = std::chars_format::general;
        if (field_end - number >= 2 && number[0] == '0' && (number[1] == 'x' || number[1] == 'X')) {
            number += 2;
            format = std::chars_format::hex;
        }
        const auto location = [&]() {
            return " at line " + std::to_string(line_number) + ", column " + std::to_string(column + 1);
        };
        if (number == field_end || *number == '+' || *number == '-') {
            throw std::invalid_argument("CSV numeric field is empty or invalid" + location());
        }
        if (format == std::chars_format::hex) {
            auto exponent = std::find_if(number, field_end, [](char value) { return value == 'p' || value == 'P'; });
            if (exponent != field_end) {
                ++exponent;
                if (exponent != field_end && (*exponent == '+' || *exponent == '-')) ++exponent;
                // Some from_chars implementations accept a repeated exponent
                // sign such as p+-2. Require a digit after the optional sign.
                if (exponent == field_end || *exponent < '0' || *exponent > '9')
                    throw std::invalid_argument("CSV hexadecimal exponent is invalid" + location());
            }
        }
        double value = 0.0;
        const auto conversion = std::from_chars(number, field_end, value, format);
        if (conversion.ec == std::errc::invalid_argument) {
            throw std::invalid_argument("CSV numeric field is empty or invalid" + location());
        }
        if (conversion.ec == std::errc::result_out_of_range) {
            throw std::out_of_range("CSV numeric field is out of range" + location());
        }
        if (conversion.ptr != field_end || !std::isfinite(value)) {
            throw std::runtime_error("CSV numeric field is invalid" + location());
        }
        values[column++] = negative ? -value : value;
        if (field_end == end) break;
        cursor = field_end + 1;
    }
    if (column != values.size()) {
        throw std::runtime_error("dataset row has the wrong field count at line " + std::to_string(line_number));
    }
    if (values[0] <= previous_time) {
        throw std::runtime_error("CSV timestamp must be strictly increasing at line " + std::to_string(line_number));
    }
    previous_time = values[0];
    return true;
}

class CsvInput {
public:
    CsvInput(const std::string& path, std::initializer_list<const char*> following_prefixes, const char* kind)
        : input_(path), path_(path), kind_(kind) {
        if (path.empty()) throw std::invalid_argument(std::string(kind) + " input path must not be empty");
        if (!input_) throw std::runtime_error("cannot open " + std::string(kind) + " input: " + path);
        if (!ReadCsvHeader(input_, line_, separator_)) throw std::runtime_error(std::string(kind) + " CSV is empty");
        const auto header = SplitCsvHeader(line_);
        const auto groups = 1 + following_prefixes.size();
        if (header.size() <= groups || header[0] != "time_s" || (header.size() - 1) % groups != 0)
            throw std::runtime_error(std::string(kind) + " CSV header has an invalid shape");
        const auto dof = (header.size() - 1) / groups;
        const auto maximum_values = std::min(static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()),
                                             std::numeric_limits<std::size_t>::max() / sizeof(double));
        if (header.size() > maximum_values) throw std::overflow_error(std::string(kind) + " CSV has too many columns");
        joint_names.reserve(dof);
        for (std::size_t joint = 0; joint < dof; ++joint) {
            const auto& field = header[1 + joint];
            if (field.rfind("q/", 0) != 0 || field.size() <= 2) throw std::runtime_error("invalid q column name");
            joint_names.emplace_back(field.substr(2));
        }
        std::size_t column = 1 + dof;
        for (std::string_view prefix : following_prefixes)
            for (const auto& joint : joint_names) {
                const auto field = header[column++];
                if (field.substr(0, prefix.size()) != prefix || field.substr(prefix.size()) != joint)
                    throw std::runtime_error(std::string(kind) + " CSV columns do not share the same joint order");
            }
        ValidateCsvJointNames(joint_names);
        values.resize(header.size());
    }

    bool ReadRow() {
        while (std::getline(input_, line_, separator_)) {
            ++line_number_;
            if (ParseCsvRow(line_, values, line_number_, previous_time_)) return true;
        }
        if (input_.bad() || !input_.eof())
            throw std::runtime_error("failed while reading " + std::string(kind_) + ": " + path_);
        return false;
    }

    std::vector<std::string> joint_names;
    std::vector<double> values;

private:
    std::ifstream input_;
    // Each public read call keeps this reader local, within the path argument's lifetime.
    const std::string& path_;
    const char* kind_;
    std::string line_;
    char separator_;
    std::size_t line_number_{1};
    double previous_time_{-std::numeric_limits<double>::infinity()};
};

}  // namespace

void WriteRawDatasetCsv(const RawDataset& dataset, const std::string& path) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid raw dataset: " + reason);
    auto output = OpenCsvOutput(path, dataset.joint_names, {"q/", "tau/"}, "raw dataset");
    for (Eigen::Index row = 0; row < dataset.time_seconds.size(); ++row) {
        output << dataset.time_seconds[row];
        for (const auto* values : {&dataset.position, &dataset.torque})
            for (Eigen::Index joint = 0; joint < values->cols(); ++joint) output << ',' << (*values)(row, joint);
        output << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing raw dataset: " + path);
}

RawDataset ReadRawDatasetCsv(const std::string& path) {
    CsvInput input(path, {"tau/"}, "raw dataset");
    RawDataset dataset;
    dataset.joint_names = std::move(input.joint_names);
    const std::size_t dof = dataset.joint_names.size();
    const auto maximum_values = std::min(static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()),
                                         std::numeric_limits<std::size_t>::max() / sizeof(double));
    struct Chunk {
        Chunk(Eigen::Index capacity, Eigen::Index columns)
            : time(capacity), position(capacity, columns), torque(capacity, columns) {}
        Eigen::VectorXd time;
        Eigen::MatrixXd position, torque;
        Eigen::Index used{0};
    };
    std::vector<Chunk> chunks;
    // Small files start with eight rows. Larger chunks are capped by both row
    // count and coefficient count, bounding excess capacity without a file pass.
    const Eigen::Index maximum_chunk_rows =
        std::max<Eigen::Index>(1, std::min<Eigen::Index>(4096, 65536 / static_cast<Eigen::Index>(input.values.size())));
    const auto columns = static_cast<Eigen::Index>(dof);
    std::size_t row_count = 0;
    const auto& values = input.values;
    while (input.ReadRow()) {
        if (row_count >= maximum_values / values.size()) {
            throw std::overflow_error("raw dataset CSV contains too many values");
        }
        if (chunks.empty() || chunks.back().used == chunks.back().time.size()) {
            const Eigen::Index capacity =
                std::min<Eigen::Index>(maximum_chunk_rows, chunks.empty() ? 8 : 2 * chunks.back().time.size());
            chunks.emplace_back(capacity, columns);
        }
        auto& chunk = chunks.back();
        chunk.time[chunk.used] = values[0];
        chunk.position.row(chunk.used) = Eigen::Map<const Eigen::RowVectorXd>(values.data() + 1, columns);
        chunk.torque.row(chunk.used) = Eigen::Map<const Eigen::RowVectorXd>(values.data() + 1 + dof, columns);
        ++chunk.used;
        ++row_count;
    }
    // Header validation and ParseCsvRow already enforce names, dimensions,
    // finite fields and strict timestamps. Only the minimum row count remains.
    if (row_count < 3)
        throw std::runtime_error(
            "raw dataset CSV is invalid: raw dataset dimensions are inconsistent or values are non-finite");
    // Materialize one field at a time and release consumed chunks immediately.
    // This avoids retaining a complete flat parse buffer alongside all outputs.
    const auto samples = static_cast<Eigen::Index>(row_count);
    const auto materialize = [&](auto& output, auto field) {
        output.resize(samples, (chunks.front().*field).cols());
        Eigen::Index offset = 0;
        for (auto& chunk : chunks) {
            auto& values = chunk.*field;
            output.middleRows(offset, chunk.used) = values.topRows(chunk.used);
            offset += chunk.used;
            values.resize(0, values.cols());
        }
    };
    materialize(dataset.time_seconds, &Chunk::time);
    materialize(dataset.position, &Chunk::position);
    materialize(dataset.torque, &Chunk::torque);
    return dataset;
}

void WriteDatasetCsv(const Dataset& dataset, const std::string& path) {
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::invalid_argument("invalid dataset: " + reason);
    auto output = OpenCsvOutput(path, dataset.joint_names, {"q/", "dq/", "ddq/", "tau/"}, "dataset");
    for (const auto& sample : dataset.samples) {
        output << sample.time_seconds;
        for (const auto* values : {&sample.position, &sample.velocity, &sample.acceleration, &sample.torque}) {
            for (Eigen::Index index = 0; index < values->size(); ++index) output << ',' << (*values)[index];
        }
        output << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing dataset: " + path);
}

Dataset ReadDatasetCsv(const std::string& path) {
    CsvInput input(path, {"dq/", "ddq/", "tau/"}, "dataset");
    Dataset dataset;
    dataset.joint_names = std::move(input.joint_names);
    const std::size_t dof = dataset.dof();
    const auto& values = input.values;
    while (input.ReadRow()) {
        Sample sample;
        sample.time_seconds = values[0];
        std::array<Eigen::VectorXd*, 4> vectors{&sample.position, &sample.velocity, &sample.acceleration,
                                                &sample.torque};
        for (std::size_t group = 0; group < vectors.size(); ++group) {
            *vectors[group] =
                Eigen::Map<const Eigen::VectorXd>(values.data() + 1 + group * dof, static_cast<Eigen::Index>(dof));
        }
        dataset.samples.push_back(std::move(sample));
    }
    std::string reason;
    if (!dataset.Validate(&reason)) throw std::runtime_error("dataset CSV is invalid: " + reason);
    return dataset;
}

}  // namespace robot_identification
