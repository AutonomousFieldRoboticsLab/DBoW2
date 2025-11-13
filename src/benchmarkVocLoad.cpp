/**
 * @file benchmarkVocLoad.cpp
 * @brief Benchmark vocabulary loading speed with different formats
 * @author Assistant
 */

#include <iostream>
#include <chrono>
#include <iomanip>

// DBoW2
#include "DBoW2.h"
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>

typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK> FBriskVocabulary;

void benchmark_load(const std::string &filename, int iterations = 3) {
  std::cout << "\n=== Benchmarking: " << filename << " ===" << std::endl;
  
  std::vector<double> load_times;
  
  for (int i = 0; i < iterations; ++i) {
    std::cout << "Iteration " << (i+1) << "/" << iterations << "... " << std::flush;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    FBriskVocabulary voc;
    try {
      voc.load(filename);
      
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      double seconds = duration.count() / 1000.0;
      load_times.push_back(seconds);
      
      std::cout << seconds << " s";
      if (i == 0) {
        std::cout << " (" << voc.size() << " words)";
      }
      std::cout << std::endl;
      
    } catch (const std::exception &e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return;
    }
  }
  
  // Calculate statistics
  double sum = 0.0;
  double min_time = load_times[0];
  double max_time = load_times[0];
  
  for (double t : load_times) {
    sum += t;
    if (t < min_time) min_time = t;
    if (t > max_time) max_time = t;
  }
  
  double avg = sum / load_times.size();
  
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nResults:" << std::endl;
  std::cout << "  Min:     " << min_time << " s" << std::endl;
  std::cout << "  Max:     " << max_time << " s" << std::endl;
  std::cout << "  Average: " << avg << " s" << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <vocabulary_file> [iterations]" << std::endl;
    std::cout << "Benchmarks vocabulary loading speed." << std::endl;
    std::cout << "  iterations: number of times to load (default: 3)" << std::endl;
    return -1;
  }

  std::string vocab_file = argv[1];
  int iterations = 3;
  
  if (argc >= 3) {
    iterations = std::atoi(argv[2]);
    if (iterations < 1) iterations = 1;
  }

  std::cout << "Vocabulary Loading Benchmark" << std::endl;
  std::cout << "============================" << std::endl;
  
  benchmark_load(vocab_file, iterations);
  
  return 0;
}
