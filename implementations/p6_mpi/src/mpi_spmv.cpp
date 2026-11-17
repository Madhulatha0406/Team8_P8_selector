#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Entry {
    int row;
    int col;
    double value;
};

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<Entry> entries;
};

struct Timings {
    double elapsed;
    double local_compute;
    double remote_compute;
    double comm_exposed;
    std::uint64_t sent_bytes;
    std::uint64_t received_bytes;
};

[[noreturn]] void fail(const std::string& message, int rank) {
    if (rank == 0) {
        std::cerr << "Error: " << message << '\n';
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

Matrix read_matrix_market(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open matrix: " + path);
    }

    std::string line;
    if (!std::getline(input, line) || line.rfind("%%MatrixMarket", 0) != 0) {
        throw std::runtime_error("expected a Matrix Market coordinate file");
    }
    std::istringstream header(line);
    std::string banner, object, format, field, symmetry;
    header >> banner >> object >> format >> field >> symmetry;
    std::transform(field.begin(), field.end(), field.begin(), ::tolower);
    std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::tolower);
    if (object != "matrix" || format != "coordinate" || field == "complex") {
        throw std::runtime_error("only real/integer/pattern coordinate matrices are supported");
    }

    Matrix matrix;
    int declared_nonzeros = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream dimensions(line);
        if (!(dimensions >> matrix.rows >> matrix.cols >> declared_nonzeros) ||
            matrix.rows <= 0 || matrix.cols <= 0 || declared_nonzeros < 0) {
            throw std::runtime_error("invalid Matrix Market dimensions");
        }
        matrix.entries.reserve(static_cast<std::size_t>(declared_nonzeros) *
                               (symmetry == "general" ? 1U : 2U));
        break;
    }
    if (matrix.rows == 0) {
        throw std::runtime_error("Matrix Market dimensions are missing");
    }

    for (int i = 0; i < declared_nonzeros; ++i) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("matrix ended before all declared entries were read");
        }
        if (line.empty() || line[0] == '%') {
            --i;
            continue;
        }
        std::istringstream row(line);
        int r = 0, c = 0;
        double value = 1.0;
        if (!(row >> r >> c) || (field != "pattern" && !(row >> value))) {
            throw std::runtime_error("invalid Matrix Market entry");
        }
        --r;
        --c;
        if (r < 0 || r >= matrix.rows || c < 0 || c >= matrix.cols) {
            throw std::runtime_error("matrix entry is outside declared dimensions");
        }
        matrix.entries.push_back({r, c, value});
        if (r != c && symmetry != "general") {
            matrix.entries.push_back(
                {c, r, symmetry == "skew-symmetric" ? -value : value});
        }
    }
    return matrix;
}

std::vector<double> read_vector(const std::string& path, int expected_size) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open vector: " + path);
    }
    std::vector<double> values;
    double value = 0.0;
    while (input >> value) {
        values.push_back(value);
    }
    if (!input.eof() || static_cast<int>(values.size()) != expected_size) {
        throw std::runtime_error("vector must contain exactly " +
                                 std::to_string(expected_size) + " numeric values");
    }
    return values;
}

std::vector<int> equal_cuts(int length, int parts) {
    std::vector<int> cuts(parts + 1);
    for (int i = 0; i <= parts; ++i) {
        cuts[i] = static_cast<int>((static_cast<long long>(length) * i) / parts);
    }
    return cuts;
}

std::vector<int> nnz_cuts(const std::vector<int>& row_counts, int parts) {
    const int rows = static_cast<int>(row_counts.size());
    std::vector<long long> prefix(rows + 1, 0);
    for (int i = 0; i < rows; ++i) {
        prefix[i + 1] = prefix[i] + row_counts[i];
    }
    std::vector<int> cuts(parts + 1, 0);
    cuts[parts] = rows;
    for (int part = 1; part < parts; ++part) {
        const long long target = prefix.back() * part / parts;
        auto it = std::lower_bound(prefix.begin(), prefix.end(), target);
        int cut = static_cast<int>(it - prefix.begin());
        if (cut > 0 && std::llabs(prefix[cut - 1] - target) <=
                           std::llabs(prefix[cut] - target)) {
            --cut;
        }
        cuts[part] = std::max(cuts[part - 1], std::min(cut, rows));
    }
    return cuts;
}

int owner_of(int index, const std::vector<int>& cuts) {
    auto it = std::upper_bound(cuts.begin() + 1, cuts.end(), index);
    return std::min(static_cast<int>(it - cuts.begin()) - 1,
                    static_cast<int>(cuts.size()) - 2);
}

std::string get_option(int argc, char** argv, const std::string& name,
                       const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int get_int_option(int argc, char** argv, const std::string& name, int fallback) {
    const std::string value = get_option(argc, argv, name, "");
    if (value.empty()) {
        return fallback;
    }
    std::size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size() || result < 0) {
        throw std::runtime_error("invalid value for " + name);
    }
    return result;
}

void append_csv_header(const std::string& path, const std::string& header) {
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot open output CSV: " + path);
    }
    output.seekp(0, std::ios::end);
    if (output.tellp() == 0) {
        output << header << '\n';
    }
}

void append_csv_row(const std::string& path, const std::string& row) {
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot write output CSV: " + path);
    }
    output << row << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        const std::string matrix_path = get_option(argc, argv, "--matrix", "");
        const std::string vector_path = get_option(argc, argv, "--vector", "");
        const std::string scheme = get_option(argc, argv, "--scheme", "row");
        const std::string partition = get_option(argc, argv, "--partition", "rows");
        const std::string output_path = get_option(argc, argv, "--output", "results/spmv.csv");
        const std::string rank_output_path =
            get_option(argc, argv, "--rank-output", "results/rank_metrics.csv");
        const int iterations = get_int_option(argc, argv, "--iterations", 5);
        const int warmup = get_int_option(argc, argv, "--warmup", 1);
        if (matrix_path.empty() || vector_path.empty() ||
            (scheme != "row" && scheme != "2d") ||
            (partition != "rows" && partition != "nnz") ||
            iterations < 1 || (scheme == "2d" && partition != "rows")) {
            fail("usage: --matrix FILE --vector FILE [--scheme row|2d] "
                 "[--partition rows|nnz] [--iterations N] [--warmup N] "
                 "[--output FILE] [--rank-output FILE]", rank);
        }

        Matrix global_matrix;
        std::vector<double> global_x;
        if (rank == 0) {
            global_matrix = read_matrix_market(matrix_path);
            if (global_matrix.rows != global_matrix.cols) {
                throw std::runtime_error("provided vector datasets require a square matrix");
            }
            global_x = read_vector(vector_path, global_matrix.cols);
        }

        int dimensions[2] = {0, 0};
        if (rank == 0) {
            dimensions[0] = global_matrix.rows;
            dimensions[1] = global_matrix.cols;
        }
        MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);
        const int rows = dimensions[0], cols = dimensions[1];

        int grid_rows = world_size, grid_cols = 1;
        if (scheme == "2d") {
            grid_rows = static_cast<int>(std::sqrt(world_size));
            while (grid_rows > 1 && world_size % grid_rows != 0) {
                --grid_rows;
            }
            grid_cols = world_size / grid_rows;
        }
        const int grid_row = scheme == "2d" ? rank / grid_cols : rank;
        const int grid_col = scheme == "2d" ? rank % grid_cols : 0;

        MPI_Comm row_comm = MPI_COMM_NULL;
        MPI_Comm col_comm = MPI_COMM_NULL;
        if (scheme == "2d") {
            MPI_Comm_split(MPI_COMM_WORLD, grid_row, grid_col, &row_comm);
            MPI_Comm_split(MPI_COMM_WORLD, grid_col, grid_row, &col_comm);
        } else {
            row_comm = MPI_COMM_WORLD;
            col_comm = MPI_COMM_WORLD;
        }
        int row_comm_rank = 0;
        MPI_Comm_rank(row_comm, &row_comm_rank);
        const int request_peers = scheme == "2d" ? grid_cols : world_size;
        const int x_owner = scheme == "2d" ? grid_col : rank;

        std::vector<int> row_cuts;
        if (scheme == "2d") {
            row_cuts = equal_cuts(rows, grid_rows);
        } else if (partition == "nnz") {
            std::vector<int> counts(rows, 0);
            if (rank == 0) {
                for (const Entry& entry : global_matrix.entries) {
                    ++counts[entry.row];
                }
            }
            MPI_Bcast(counts.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);
            row_cuts = nnz_cuts(counts, world_size);
        } else {
            row_cuts = equal_cuts(rows, world_size);
        }
        const std::vector<int> vector_cuts = equal_cuts(cols, request_peers);
        const std::vector<int> column_cuts =
            equal_cuts(cols, scheme == "2d" ? grid_cols : 1);
        const int local_row_start = row_cuts[grid_row];
        const int local_row_count = row_cuts[grid_row + 1] - local_row_start;
        const int local_x_start = vector_cuts[x_owner];
        const int local_x_count = vector_cuts[x_owner + 1] - local_x_start;

        std::vector<int> matrix_counts, matrix_displs;
        std::vector<int> matrix_rows, matrix_cols;
        std::vector<double> matrix_values;
        std::vector<int> vector_counts(world_size, 0), vector_displs(world_size, 0);
        std::vector<double> scatter_x;
        if (rank == 0) {
            matrix_counts.assign(world_size, 0);
            std::vector<std::vector<Entry>> buckets(world_size);
            for (const Entry& entry : global_matrix.entries) {
                const int dest = owner_of(entry.row, row_cuts) * grid_cols +
                                 (scheme == "2d" ? owner_of(entry.col, column_cuts) : 0);
                buckets[dest].push_back(entry);
            }
            matrix_displs.assign(world_size, 0);
            for (int dest = 0; dest < world_size; ++dest) {
                matrix_counts[dest] = static_cast<int>(buckets[dest].size());
                if (dest > 0) {
                    matrix_displs[dest] =
                        matrix_displs[dest - 1] + matrix_counts[dest - 1];
                }
                const auto& bucket = buckets[dest];
                for (const Entry& entry : bucket) {
                    matrix_rows.push_back(entry.row);
                    matrix_cols.push_back(entry.col);
                    matrix_values.push_back(entry.value);
                }
            }
        }
        int local_entry_count = 0;
        MPI_Scatter(rank == 0 ? matrix_counts.data() : nullptr, 1, MPI_INT,
                    &local_entry_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> local_entry_rows(local_entry_count);
        std::vector<int> local_entry_cols(local_entry_count);
        std::vector<double> local_entry_values(local_entry_count);
        MPI_Scatterv(rank == 0 ? matrix_rows.data() : nullptr,
                     rank == 0 ? matrix_counts.data() : nullptr,
                     rank == 0 ? matrix_displs.data() : nullptr, MPI_INT,
                     local_entry_rows.data(), local_entry_count, MPI_INT, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? matrix_cols.data() : nullptr,
                     rank == 0 ? matrix_counts.data() : nullptr,
                     rank == 0 ? matrix_displs.data() : nullptr, MPI_INT,
                     local_entry_cols.data(), local_entry_count, MPI_INT, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? matrix_values.data() : nullptr,
                     rank == 0 ? matrix_counts.data() : nullptr,
                     rank == 0 ? matrix_displs.data() : nullptr, MPI_DOUBLE,
                     local_entry_values.data(), local_entry_count, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);

        if (rank == 0) {
            for (int dest = 0; dest < world_size; ++dest) {
                const int owner = scheme == "2d" ? dest % grid_cols : dest;
                if (scheme != "2d" || dest / grid_cols == 0) {
                    vector_counts[dest] =
                        vector_cuts[owner + 1] - vector_cuts[owner];
                }
            }
        }
        if (rank == 0) {
            int offset = 0;
            for (int dest = 0; dest < world_size; ++dest) {
                vector_displs[dest] = offset;
                offset += vector_counts[dest];
            }
            scatter_x.resize(offset);
            for (int dest = 0; dest < world_size; ++dest) {
                const int owner = scheme == "2d" ? dest % grid_cols : dest;
                if (vector_counts[dest] > 0) {
                    std::copy_n(global_x.begin() + vector_cuts[owner],
                                vector_counts[dest],
                                scatter_x.begin() + vector_displs[dest]);
                }
            }
        }
        std::vector<double> local_x(local_x_count);
        MPI_Scatterv(rank == 0 ? scatter_x.data() : nullptr,
                     rank == 0 ? vector_counts.data() : nullptr,
                     rank == 0 ? vector_displs.data() : nullptr, MPI_DOUBLE,
                     local_x.data(), scheme == "2d" && grid_row != 0 ? 0 : local_x_count,
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (scheme == "2d") {
            MPI_Bcast(local_x.data(), local_x_count, MPI_DOUBLE, 0, col_comm);
        }
        std::vector<std::vector<int>> needed_by_owner(request_peers);
        for (int i = 0; i < local_entry_count; ++i) {
            const int owner = owner_of(local_entry_cols[i], vector_cuts);
            if (owner != x_owner) {
                needed_by_owner[owner].push_back(local_entry_cols[i]);
            }
        }
        std::vector<int> request_counts(request_peers, 0);
        std::vector<int> request_displs(request_peers, 0);
        std::vector<int> requested_columns;
        for (int owner = 0; owner < request_peers; ++owner) {
            auto& indices = needed_by_owner[owner];
            std::sort(indices.begin(), indices.end());
            indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
            request_displs[owner] = static_cast<int>(requested_columns.size());
            request_counts[owner] = static_cast<int>(indices.size());
            requested_columns.insert(requested_columns.end(), indices.begin(), indices.end());
        }
        std::vector<int> incoming_counts(request_peers, 0);
        std::vector<int> incoming_displs(request_peers, 0);
        MPI_Alltoall(request_counts.data(), 1, MPI_INT, incoming_counts.data(), 1,
                     MPI_INT, row_comm);
        int incoming_total = 0;
        for (int peer = 0; peer < request_peers; ++peer) {
            incoming_displs[peer] = incoming_total;
            incoming_total += incoming_counts[peer];
        }
        std::vector<int> incoming_columns(incoming_total);
        MPI_Alltoallv(requested_columns.data(), request_counts.data(),
                      request_displs.data(), MPI_INT, incoming_columns.data(),
                      incoming_counts.data(), incoming_displs.data(), MPI_INT,
                      row_comm);

        std::unordered_map<int, int> remote_slot;
        remote_slot.reserve(requested_columns.size());
        for (int i = 0; i < static_cast<int>(requested_columns.size()); ++i) {
            remote_slot.emplace(requested_columns[i], i);
        }
        std::vector<int> local_rows, remote_rows, remote_slots;
        std::vector<int> local_x_indices;
        std::vector<double> local_values, remote_values;
        for (int i = 0; i < local_entry_count; ++i) {
            const int owner = owner_of(local_entry_cols[i], vector_cuts);
            if (owner == x_owner) {
                local_rows.push_back(local_entry_rows[i] - local_row_start);
                local_x_indices.push_back(local_entry_cols[i] - local_x_start);
                local_values.push_back(local_entry_values[i]);
            } else {
                remote_rows.push_back(local_entry_rows[i] - local_row_start);
                remote_slots.push_back(remote_slot.at(local_entry_cols[i]));
                remote_values.push_back(local_entry_values[i]);
            }
        }

        if (rank == 0) {
            append_csv_header(output_path,
                              "matrix,scheme,partition,ranks,iteration,nnz,elapsed_s,"
                              "local_compute_s,remote_compute_s,comm_exposed_s");
            append_csv_header(rank_output_path,
                              "matrix,scheme,partition,ranks,iteration,rank,sent_bytes,"
                              "received_bytes,local_compute_s,remote_compute_s,"
                              "comm_exposed_s");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        const std::string matrix_name =
            matrix_path.substr(matrix_path.find_last_of("/\\") == std::string::npos
                                   ? 0
                                   : matrix_path.find_last_of("/\\") + 1);
        const int total_runs = warmup + iterations;
        for (int iteration = 0; iteration < total_runs; ++iteration) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double start = MPI_Wtime();
            std::vector<double> received_values(requested_columns.size(), 0.0);
            std::vector<double> values_to_send(incoming_total, 0.0);
            for (int i = 0; i < incoming_total; ++i) {
                values_to_send[i] = local_x[incoming_columns[i] - local_x_start];
            }
            std::vector<MPI_Request> requests;
            requests.reserve(2 * request_peers + 2 * grid_cols);
            for (int peer = 0; peer < request_peers; ++peer) {
                if (request_counts[peer] > 0) {
                    MPI_Request request;
                    MPI_Irecv(received_values.data() + request_displs[peer],
                              request_counts[peer], MPI_DOUBLE, peer, 701, row_comm,
                              &request);
                    requests.push_back(request);
                }
                if (incoming_counts[peer] > 0) {
                    MPI_Request request;
                    MPI_Isend(values_to_send.data() + incoming_displs[peer],
                              incoming_counts[peer], MPI_DOUBLE, peer, 701, row_comm,
                              &request);
                    requests.push_back(request);
                }
            }
            double sent = static_cast<double>(incoming_total) * sizeof(double);
            double received = static_cast<double>(requested_columns.size()) * sizeof(double);
            if (scheme == "2d") {
                std::vector<double> partial_y(local_row_count, 0.0);
                for (std::size_t i = 0; i < local_rows.size(); ++i) {
                    partial_y[local_rows[i]] +=
                        local_values[i] * local_x[local_x_indices[i]];
                }
                const double local_done = MPI_Wtime();
                if (!requests.empty()) {
                    MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                                MPI_STATUSES_IGNORE);
                }
                const double exchange_done = MPI_Wtime();
                for (std::size_t i = 0; i < remote_rows.size(); ++i) {
                    partial_y[remote_rows[i]] +=
                        remote_values[i] * received_values[remote_slots[i]];
                }
                const double remote_done = MPI_Wtime();

                std::vector<MPI_Request> reductions;
                if (grid_col == 0) {
                    std::vector<std::vector<double>> partials(grid_cols - 1);
                    for (int peer = 1; peer < grid_cols; ++peer) {
                        partials[peer - 1].resize(local_row_count);
                        MPI_Request request;
                        MPI_Irecv(partials[peer - 1].data(), local_row_count,
                                  MPI_DOUBLE, peer, 703, row_comm, &request);
                        reductions.push_back(request);
                    }
                    if (!reductions.empty()) {
                        MPI_Waitall(static_cast<int>(reductions.size()),
                                    reductions.data(), MPI_STATUSES_IGNORE);
                    }
                    for (const auto& part : partials) {
                        for (int r = 0; r < local_row_count; ++r) {
                            partial_y[r] += part[r];
                        }
                    }
                    received += static_cast<double>(local_row_count) *
                                sizeof(double) * (grid_cols - 1);
                } else {
                    MPI_Send(partial_y.data(), local_row_count, MPI_DOUBLE, 0, 703,
                             row_comm);
                    sent += static_cast<double>(local_row_count) * sizeof(double);
                }
                const double end = MPI_Wtime();
                const double local_seconds = local_done - start;
                const double remote_seconds = remote_done - exchange_done;
                const double elapsed = end - start;
                Timings timing{elapsed, local_seconds, remote_seconds,
                               std::max(0.0, elapsed - local_seconds - remote_seconds),
                               static_cast<std::uint64_t>(sent),
                               static_cast<std::uint64_t>(received)};
                // Keep the 2-D result only on column zero for the final gather.
                if (iteration == total_runs - 1) {
                    local_values.swap(partial_y);
                }
                double global_elapsed = 0.0;
                MPI_Reduce(&timing.elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                           0, MPI_COMM_WORLD);
                if (iteration >= warmup) {
                    double local_metrics[3] = {timing.local_compute,
                                               timing.remote_compute,
                                               timing.comm_exposed};
                    std::vector<double> all_metrics(rank == 0 ? world_size * 3 : 0);
                    std::uint64_t local_bytes[2] = {timing.sent_bytes,
                                                    timing.received_bytes};
                    std::vector<std::uint64_t> all_bytes(rank == 0 ? world_size * 2 : 0);
                    MPI_Gather(local_metrics, 3, MPI_DOUBLE,
                               rank == 0 ? all_metrics.data() : nullptr, 3, MPI_DOUBLE,
                               0, MPI_COMM_WORLD);
                    MPI_Gather(local_bytes, 2, MPI_UINT64_T,
                               rank == 0 ? all_bytes.data() : nullptr, 2, MPI_UINT64_T,
                               0, MPI_COMM_WORLD);
                    if (rank == 0) {
                        std::ostringstream line;
                        line << matrix_name << ",2d,rows," << world_size << ','
                             << (iteration - warmup) << ','
                             << global_matrix.entries.size() << ','
                             << std::setprecision(10) << global_elapsed << ','
                             << *std::max_element(all_metrics.begin(),
                                                  all_metrics.end(),
                                                  [](double a, double b) {
                                                      return a < b;
                                                  })
                             << ',' << 0 << ',' << 0;
                        // Summary phase columns use maximum phase times, not rank zero.
                        double max_local = 0.0, max_remote = 0.0, max_comm = 0.0;
                        for (int r = 0; r < world_size; ++r) {
                            max_local = std::max(max_local, all_metrics[r * 3]);
                            max_remote = std::max(max_remote, all_metrics[r * 3 + 1]);
                            max_comm = std::max(max_comm, all_metrics[r * 3 + 2]);
                        }
                        line.str("");
                        line.clear();
                        line << matrix_name << ",2d,rows," << world_size << ','
                             << (iteration - warmup) << ','
                             << global_matrix.entries.size() << ','
                             << std::setprecision(10) << global_elapsed << ','
                             << max_local << ',' << max_remote << ',' << max_comm;
                        append_csv_row(output_path, line.str());
                        for (int r = 0; r < world_size; ++r) {
                            std::ostringstream rank_line;
                            rank_line << matrix_name << ",2d,rows," << world_size << ','
                                      << (iteration - warmup) << ',' << r << ','
                                      << all_bytes[r * 2] << ',' << all_bytes[r * 2 + 1]
                                      << ',' << all_metrics[r * 3] << ','
                                      << all_metrics[r * 3 + 1] << ','
                                      << all_metrics[r * 3 + 2];
                            append_csv_row(rank_output_path, rank_line.str());
                        }
                    }
                }
            } else {
                std::vector<double> local_y(local_row_count, 0.0);
                for (std::size_t i = 0; i < local_rows.size(); ++i) {
                    local_y[local_rows[i]] +=
                        local_values[i] * local_x[local_x_indices[i]];
                }
                const double local_done = MPI_Wtime();
                if (!requests.empty()) {
                    MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                                MPI_STATUSES_IGNORE);
                }
                const double exchange_done = MPI_Wtime();
                for (std::size_t i = 0; i < remote_rows.size(); ++i) {
                    local_y[remote_rows[i]] +=
                        remote_values[i] * received_values[remote_slots[i]];
                }
                const double remote_done = MPI_Wtime();
                const double end = MPI_Wtime();
                Timings timing{end - start, local_done - start,
                               remote_done - exchange_done,
                               std::max(0.0, end - start - (local_done - start) -
                                                 (remote_done - exchange_done)),
                               static_cast<std::uint64_t>(sent),
                               static_cast<std::uint64_t>(received)};
                if (iteration == total_runs - 1) {
                    local_values.swap(local_y);
                }
                double global_elapsed = 0.0;
                MPI_Reduce(&timing.elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX,
                           0, MPI_COMM_WORLD);
                if (iteration >= warmup) {
                    double local_metrics[3] = {timing.local_compute,
                                               timing.remote_compute,
                                               timing.comm_exposed};
                    std::vector<double> all_metrics(rank == 0 ? world_size * 3 : 0);
                    std::uint64_t local_bytes[2] = {timing.sent_bytes,
                                                    timing.received_bytes};
                    std::vector<std::uint64_t> all_bytes(rank == 0 ? world_size * 2 : 0);
                    MPI_Gather(local_metrics, 3, MPI_DOUBLE,
                               rank == 0 ? all_metrics.data() : nullptr, 3, MPI_DOUBLE,
                               0, MPI_COMM_WORLD);
                    MPI_Gather(local_bytes, 2, MPI_UINT64_T,
                               rank == 0 ? all_bytes.data() : nullptr, 2, MPI_UINT64_T,
                               0, MPI_COMM_WORLD);
                    if (rank == 0) {
                        double max_local = 0.0, max_remote = 0.0, max_comm = 0.0;
                        for (int r = 0; r < world_size; ++r) {
                            max_local = std::max(max_local, all_metrics[r * 3]);
                            max_remote = std::max(max_remote, all_metrics[r * 3 + 1]);
                            max_comm = std::max(max_comm, all_metrics[r * 3 + 2]);
                        }
                        std::ostringstream line;
                        line << matrix_name << ",row," << partition << ','
                             << world_size << ',' << (iteration - warmup) << ','
                             << global_matrix.entries.size() << ','
                             << std::setprecision(10) << global_elapsed << ','
                             << max_local << ',' << max_remote << ',' << max_comm;
                        append_csv_row(output_path, line.str());
                        for (int r = 0; r < world_size; ++r) {
                            std::ostringstream rank_line;
                            rank_line << matrix_name << ",row," << partition << ','
                                      << world_size << ',' << (iteration - warmup) << ','
                                      << r << ',' << all_bytes[r * 2] << ','
                                      << all_bytes[r * 2 + 1] << ','
                                      << all_metrics[r * 3] << ','
                                      << all_metrics[r * 3 + 1] << ','
                                      << all_metrics[r * 3 + 2];
                            append_csv_row(rank_output_path, rank_line.str());
                        }
                    }
                }
            }
        }

        std::vector<int> gather_counts(world_size, 0), gather_displs(world_size, 0);
        for (int source = 0; source < world_size; ++source) {
            if (scheme != "2d" || source % grid_cols == 0) {
                const int source_row = scheme == "2d" ? source / grid_cols : source;
                gather_counts[source] =
                    row_cuts[source_row + 1] - row_cuts[source_row];
                gather_displs[source] = row_cuts[source_row];
            }
        }
        std::vector<double> result(rank == 0 ? rows : 0);
        const int local_result_count =
            scheme != "2d" || grid_col == 0 ? local_row_count : 0;
        MPI_Gatherv(local_result_count > 0 ? local_values.data() : nullptr,
                    local_result_count, MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr,
                    rank == 0 ? gather_counts.data() : nullptr,
                    rank == 0 ? gather_displs.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        int correct = 1;
        if (rank == 0) {
            std::vector<double> reference(rows, 0.0);
            for (const Entry& entry : global_matrix.entries) {
                reference[entry.row] += entry.value * global_x[entry.col];
            }
            for (int r = 0; r < rows; ++r) {
                if (std::abs(reference[r] - result[r]) >
                    1e-9 * std::max(1.0, std::abs(reference[r]))) {
                    correct = 0;
                    break;
                }
            }
        }
        MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Validation: " << (correct ? "PASS" : "FAIL")
                      << " (matrix=" << matrix_name << ", scheme=" << scheme
                      << ", ranks=" << world_size << ")\n";
        }
        if (!correct) {
            fail("distributed result differs from serial reference", rank);
        }

        if (scheme == "2d") {
            MPI_Comm_free(&row_comm);
            MPI_Comm_free(&col_comm);
        }
    } catch (const std::exception& error) {
        fail(error.what(), rank);
    }
    MPI_Finalize();
    return 0;
}
