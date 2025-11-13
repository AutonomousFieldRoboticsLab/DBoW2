/**
 * @file benchmarkParallelLoad.cpp
 * @brief Benchmark parallel descriptor parsing during vocabulary load
 * 
 * This tests different approaches to parallelizing vocabulary loading:
 * 1. Sequential (baseline)
 * 2. Parallel descriptor parsing
 * 3. Parallel with different thread counts
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <thread>
#include <future>

// DBoW2
#include "DBoW2.h"
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>

typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK> FBriskVocabulary;

// Test parallel descriptor parsing
double test_parallel_descriptor_parsing(const std::string& filename, int num_threads) {
  auto start = std::chrono::high_resolution_clock::now();
  
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if(!fs.isOpened()) {
    throw std::string("Could not open file");
  }

  cv::FileNode fvoc = fs["vocabulary"];
  cv::FileNode fn_nodes = fvoc["nodes"];
  
  // First pass: collect descriptor strings
  std::vector<std::pair<DBoW2::NodeId, std::string>> descriptor_strings;
  descriptor_strings.reserve(fn_nodes.size());
  
  for(cv::FileNodeIterator it = fn_nodes.begin(); it != fn_nodes.end(); ++it) {
    DBoW2::NodeId nid = (int)(*it)["nodeId"];
    std::string d = (std::string)(*it)["descriptor"];
    descriptor_strings.emplace_back(nid, d);
  }
  
  auto collect_end = std::chrono::high_resolution_clock::now();
  
  // Second pass: parse descriptors in parallel
  std::vector<DBoW2::FBRISK::TDescriptor> descriptors(descriptor_strings.size());
  
  auto parse_chunk = [&](size_t start_idx, size_t end_idx) {
    for (size_t i = start_idx; i < end_idx; ++i) {
      DBoW2::FBRISK::fromString(descriptors[i], descriptor_strings[i].second);
    }
  };
  
  std::vector<std::thread> threads;
  size_t chunk_size = (descriptor_strings.size() + num_threads - 1) / num_threads;
  
  for (int t = 0; t < num_threads; ++t) {
    size_t start_idx = t * chunk_size;
    size_t end_idx = std::min(start_idx + chunk_size, descriptor_strings.size());
    if (start_idx < descriptor_strings.size()) {
      threads.emplace_back(parse_chunk, start_idx, end_idx);
    }
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  
  fs.release();
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto collect_duration = std::chrono::duration_cast<std::chrono::milliseconds>(collect_end - start);
  auto parse_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - collect_end);
  
  std::cout << "  Collect phase: " << (collect_duration.count() / 1000.0) << "s" << std::endl;
  std::cout << "  Parse phase: " << (parse_duration.count() / 1000.0) << "s" << std::endl;
  std::cout << "  Total: " << (duration.count() / 1000.0) << "s" << std::endl;
  
  return duration.count() / 1000.0;
}

// Standard sequential load
double test_sequential_load(const std::string& filename) {
  auto start = std::chrono::high_resolution_clock::now();
  
  FBriskVocabulary voc;
  voc.load(filename);
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  return duration.count() / 1000.0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <vocabulary_file>" << std::endl;
    std::cout << "Benchmarks sequential vs parallel descriptor parsing." << std::endl;
    return -1;
  }

  std::string vocab_file = argv[1];
  int hw_threads = std::thread::hardware_concurrency();

  std::cout << "Parallel Loading Benchmark" << std::endl;
  std::cout << "==========================" << std::endl;
  std::cout << "File: " << vocab_file << std::endl;
  std::cout << "Hardware threads: " << hw_threads << std::endl;
  std::cout << std::endl;

  try {
    // Test 1: Sequential baseline
    std::cout << "Test 1: Sequential (baseline)" << std::endl;
    double seq_time = test_sequential_load(vocab_file);
    std::cout << "Time: " << std::fixed << std::setprecision(3) << seq_time << "s" << std::endl;
    std::cout << std::endl;
    
    // Test 2: Parallel descriptor parsing with different thread counts
    std::vector<int> thread_counts = {2, 4, hw_threads};
    if (hw_threads > 8) thread_counts.push_back(hw_threads / 2);
    
    // Remove duplicates and sort
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()), thread_counts.end());
    
    for (int num_threads : thread_counts) {
      std::cout << "Test: Parallel with " << num_threads << " threads" << std::endl;
      double par_time = test_parallel_descriptor_parsing(vocab_file, num_threads);
      double speedup = seq_time / par_time;
      double improvement = ((seq_time - par_time) / seq_time) * 100.0;
      
      std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x";
      if (improvement > 0) {
        std::cout << " (" << improvement << "% faster)";
      } else {
        std::cout << " (" << -improvement << "% slower)";
      }
      std::cout << std::endl << std::endl;
    }
    
    std::cout << "==========================" << std::endl;
    std::cout << "Note: The parallel approach only parallelizes descriptor parsing." << std::endl;
    std::cout << "OpenCV FileNode iteration is sequential (not thread-safe)." << std::endl;
    std::cout << "For true parallel loading, a custom binary format would be needed." << std::endl;
    
  } catch(const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 4;
  } catch(const std::string &e) {
    std::cerr << "Exception: " << e << std::endl;
    return 4;
  }

  return 0;
}
