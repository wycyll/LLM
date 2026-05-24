#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check_utils.cuh"
#include "gemm_common.cuh"
#include "timer.cuh"

struct ShapeRecord {
  std::string shape_id;
  int M = 0;
  int N = 0;
  int K = 0;
};

struct Args {
  std::string mode = "both";
  std::string shapes_path;
  std::string jsonl_output;
  int warmup = 10;
  int repeat = 100;
  float rtol = 1.0e-2f;
  float atol = 1.0e-2f;
  int seed = 1234;
};

static std::string read_text(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

static std::string extract_string_field(const std::string& object,
                                        const std::string& key,
                                        const std::string& fallback) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  std::smatch match;
  if (std::regex_search(object, match, pattern)) {
    return match[1].str();
  }
  return fallback;
}

static int extract_int_field(const std::string& object, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(object, match, pattern)) {
    throw std::runtime_error("missing shape field " + key + " in " + object);
  }
  return std::stoi(match[1].str());
}

static std::vector<ShapeRecord> load_shapes(const std::string& path) {
  const std::string text = read_text(path);
  const std::regex object_pattern("\\{[^\\}]*\\}");
  std::vector<ShapeRecord> shapes;
  int index = 0;
  for (auto it = std::sregex_iterator(text.begin(), text.end(), object_pattern);
       it != std::sregex_iterator(); ++it) {
    const std::string object = it->str();
    ShapeRecord shape;
    shape.shape_id = extract_string_field(object, "shape_id", "shape_" + std::to_string(index));
    shape.M = extract_int_field(object, "M");
    shape.N = extract_int_field(object, "N");
    shape.K = extract_int_field(object, "K");
    shapes.push_back(shape);
    ++index;
  }
  if (shapes.empty()) {
    throw std::runtime_error("no shapes parsed from " + path);
  }
  return shapes;
}

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
      }
      return argv[++i];
    };
    if (key == "--mode") {
      args.mode = next();
    } else if (key == "--shapes") {
      args.shapes_path = next();
    } else if (key == "--jsonl-output") {
      args.jsonl_output = next();
    } else if (key == "--warmup") {
      args.warmup = std::stoi(next());
    } else if (key == "--repeat") {
      args.repeat = std::stoi(next());
    } else if (key == "--rtol") {
      args.rtol = std::stof(next());
    } else if (key == "--atol") {
      args.atol = std::stof(next());
    } else if (key == "--seed") {
      args.seed = std::stoi(next());
    } else if (key == "--help") {
      std::cout << "gemm_runner --mode correctness|performance|both --shapes path "
                   "[--jsonl-output path] [--warmup n] [--repeat n]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument " + key);
    }
  }
  if (args.shapes_path.empty()) {
    throw std::runtime_error("--shapes is required");
  }
  args.warmup = std::max(args.warmup, 0);
  args.repeat = std::max(args.repeat, 1);
  return args;
}

static std::vector<half> make_input(size_t count, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  std::vector<half> values(count);
  for (size_t idx = 0; idx < count; ++idx) {
    values[idx] = __float2half(dist(rng));
  }
  return values;
}

static void print_json_string(std::ostream& os, const std::string& key,
                              const std::string& value, bool comma = true) {
  os << '"' << key << "\":\"";
  for (char ch : value) {
    if (ch == '"' || ch == '\\') {
      os << '\\';
    }
    os << ch;
  }
  os << '"';
  if (comma) {
    os << ',';
  }
}

static void print_json_number(std::ostream& os, const std::string& key,
                              double value, bool comma = true) {
  os << '"' << key << "\":" << std::setprecision(10) << value;
  if (comma) {
    os << ',';
  }
}

static void print_json_bool(std::ostream& os, const std::string& key, bool value,
                            bool comma = true) {
  os << '"' << key << "\":" << (value ? "true" : "false");
  if (comma) {
    os << ',';
  }
}

static std::vector<float> time_many(cudaStream_t stream, int warmup, int repeat,
                                    const std::function<void()>& fn) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<float> timings;
  timings.reserve(repeat);
  GpuTimer timer;
  for (int i = 0; i < repeat; ++i) {
    timer.record_start(stream);
    fn();
    timings.push_back(timer.record_stop_ms(stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return timings;
}

static double mean(const std::vector<float>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

static double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

static double stddev(const std::vector<float>& values, double avg) {
  double accum = 0.0;
  for (float value : values) {
    const double diff = value - avg;
    accum += diff * diff;
  }
  return std::sqrt(accum / static_cast<double>(values.size()));
}

static void run_shape(const ShapeRecord& shape, const Args& args,
                      cublasHandle_t handle, cudaStream_t stream,
                      std::ostream& output) {
  const size_t bytes_a = static_cast<size_t>(shape.M) * shape.K * sizeof(half);
  const size_t bytes_b = static_cast<size_t>(shape.K) * shape.N * sizeof(half);
  const size_t bytes_c = static_cast<size_t>(shape.M) * shape.N * sizeof(half);

  std::vector<half> host_a = make_input(static_cast<size_t>(shape.M) * shape.K,
                                        args.seed + shape.M + shape.K);
  std::vector<half> host_b = make_input(static_cast<size_t>(shape.K) * shape.N,
                                        args.seed + shape.N + shape.K + 17);

  half* device_a = nullptr;
  half* device_b = nullptr;
  half* device_ref = nullptr;
  half* device_test = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, bytes_a));
  CHECK_CUDA(cudaMalloc(&device_b, bytes_b));
  CHECK_CUDA(cudaMalloc(&device_ref, bytes_c));
  CHECK_CUDA(cudaMalloc(&device_test, bytes_c));
  CHECK_CUDA(cudaMemcpyAsync(device_a, host_a.data(), bytes_a,
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(device_b, host_b.data(), bytes_b,
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemsetAsync(device_ref, 0, bytes_c, stream));
  CHECK_CUDA(cudaMemsetAsync(device_test, 0, bytes_c, stream));

  bool run_success = true;
  std::string runtime_error;

  run_cublas_reference(handle, device_a, device_b, device_ref, shape.M, shape.N,
                       shape.K, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  launch_gemm_kernel(device_a, device_b, device_test, shape.M, shape.N, shape.K,
                     stream);
  cudaError_t launch_status = cudaGetLastError();
  cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (launch_status != cudaSuccess || sync_status != cudaSuccess) {
    run_success = false;
    runtime_error = std::string(cudaGetErrorString(launch_status != cudaSuccess
                                                       ? launch_status
                                                       : sync_status));
  }

  std::vector<half> host_ref(static_cast<size_t>(shape.M) * shape.N);
  std::vector<half> host_test(static_cast<size_t>(shape.M) * shape.N);
  if (run_success) {
    CHECK_CUDA(cudaMemcpyAsync(host_ref.data(), device_ref, bytes_c,
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(host_test.data(), device_test, bytes_c,
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double mean_abs_error = 0.0;
  int64_t nan_count = 0;
  int64_t inf_count = 0;
  bool correct = run_success;

  if (run_success) {
    for (size_t idx = 0; idx < host_ref.size(); ++idx) {
      const float ref = __half2float(host_ref[idx]);
      const float test = __half2float(host_test[idx]);
      if (std::isnan(test)) {
        ++nan_count;
      }
      if (std::isinf(test)) {
        ++inf_count;
      }
      const double abs_error = std::abs(static_cast<double>(test) - ref);
      const double rel_error_value = relative_error(test, ref);
      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error_value);
      mean_abs_error += abs_error;
      if (abs_error > args.atol && rel_error_value > args.rtol) {
        correct = false;
      }
    }
    mean_abs_error /= static_cast<double>(host_ref.size());
  }

  output << '{';
  print_json_string(output, "mode", "correctness");
  print_json_string(output, "shape_id", shape.shape_id);
  print_json_number(output, "M", shape.M);
  print_json_number(output, "N", shape.N);
  print_json_number(output, "K", shape.K);
  print_json_bool(output, "run_success", run_success);
  print_json_bool(output, "correct", correct);
  print_json_number(output, "max_abs_error", max_abs_error);
  print_json_number(output, "max_rel_error", max_rel_error);
  print_json_number(output, "mean_abs_error", mean_abs_error);
  print_json_number(output, "nan_count", nan_count);
  print_json_number(output, "inf_count", inf_count);
  print_json_string(output, "runtime_error", runtime_error, false);
  output << "}\n";

  if ((args.mode == "performance" || args.mode == "both") && run_success &&
      correct) {
    CHECK_CUDA(cudaMemsetAsync(device_ref, 0, bytes_c, stream));
    CHECK_CUDA(cudaMemsetAsync(device_test, 0, bytes_c, stream));
    const std::vector<float> cublas_times = time_many(stream, args.warmup,
                                                      args.repeat, [&]() {
                                                        run_cublas_reference(
                                                            handle, device_a,
                                                            device_b, device_ref,
                                                            shape.M, shape.N,
                                                            shape.K, stream);
                                                      });
    const std::vector<float> candidate_times = time_many(stream, args.warmup,
                                                         args.repeat, [&]() {
                                                           launch_gemm_kernel(
                                                               device_a, device_b,
                                                               device_test,
                                                               shape.M, shape.N,
                                                               shape.K, stream);
                                                         });
    const double candidate_mean = mean(candidate_times);
    const double candidate_median = median(candidate_times);
    const double candidate_min = *std::min_element(candidate_times.begin(),
                                                   candidate_times.end());
    const double candidate_std = stddev(candidate_times, candidate_mean);
    const double cublas_mean = mean(cublas_times);
    const double cublas_min = *std::min_element(cublas_times.begin(),
                                                cublas_times.end());

    output << '{';
    print_json_string(output, "mode", "performance");
    print_json_string(output, "shape_id", shape.shape_id);
    print_json_number(output, "M", shape.M);
    print_json_number(output, "N", shape.N);
    print_json_number(output, "K", shape.K);
    print_json_number(output, "mean_ms", candidate_mean);
    print_json_number(output, "median_ms", candidate_median);
    print_json_number(output, "min_ms", candidate_min);
    print_json_number(output, "std_ms", candidate_std);
    print_json_number(output, "tflops_mean", tflops_from_ms(shape.M, shape.N, shape.K, candidate_mean));
    print_json_number(output, "tflops_max", tflops_from_ms(shape.M, shape.N, shape.K, candidate_min));
    print_json_number(output, "cublas_mean_ms", cublas_mean);
    print_json_number(output, "cublas_min_ms", cublas_min);
    print_json_number(output, "cublas_tflops", tflops_from_ms(shape.M, shape.N, shape.K, cublas_mean));
    print_json_number(output, "speedup_vs_cublas", cublas_mean / candidate_mean);
    print_json_bool(output, "performance_valid", true, false);
    output << "}\n";
  }

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_ref);
  cudaFree(device_test);
}

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    const std::vector<ShapeRecord> shapes = load_shapes(args.shapes_path);

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::ofstream file_output;
    std::ostream* output = &std::cout;
    if (!args.jsonl_output.empty()) {
      file_output.open(args.jsonl_output);
      if (!file_output) {
        throw std::runtime_error("failed to open output " + args.jsonl_output);
      }
      output = &file_output;
    }

    for (const ShapeRecord& shape : shapes) {
      run_shape(shape, args, handle, stream, *output);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "gemm_runner error: " << error.what() << "\n";
    return 1;
  }
}
