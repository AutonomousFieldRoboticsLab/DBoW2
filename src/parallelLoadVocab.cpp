/**
 * @file parallelLoadVocab.cpp
 * @brief Parallel vocabulary loader using multithreading
 * 
 * This implementation attempts to parallelize the vocabulary loading process
 * by processing nodes in parallel batches.
 */

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

// DBoW2
#include "DBoW2.h"
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>

typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK> FBriskVocabulary;

// Structure to hold node data during parallel parsing
struct NodeData {
  DBoW2::NodeId nid;
  DBoW2::NodeId pid;
  DBoW2::WordValue weight;
  std::string descriptor_str;
  
  NodeData() : nid(0), pid(0), weight(0.0) {}
  NodeData(DBoW2::NodeId n, DBoW2::NodeId p, DBoW2::WordValue w, const std::string& d)
    : nid(n), pid(p), weight(w), descriptor_str(d) {}
};

// Parallel node parser
void parse_node_batch(
    cv::FileNodeIterator start,
    cv::FileNodeIterator end,
    std::vector<NodeData>& output,
    size_t output_offset,
    std::atomic<size_t>& progress,
    size_t total)
{
  size_t idx = 0;
  for (cv::FileNodeIterator it = start; it != end; ++it, ++idx) {
    const cv::FileNode &n = *it;
    DBoW2::NodeId nid = (int)n["nodeId"];
    DBoW2::NodeId pid = (int)n["parentId"];
    DBoW2::WordValue weight = (DBoW2::WordValue)n["weight"];
    std::string d = (std::string)n["descriptor"];
    
    output[output_offset + idx] = NodeData(nid, pid, weight, d);
    
    // Update progress
    ++progress;
  }
}

void load_vocabulary_parallel(FBriskVocabulary& voc, const std::string& filename, int num_threads) {
  std::cout << "Loading with " << num_threads << " threads..." << std::endl;
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Open file
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if(!fs.isOpened()) {
    throw std::string("Could not open file ") + filename;
  }

  cv::FileNode fvoc = fs["vocabulary"];
  if(fvoc.empty()) {
    throw std::string("File does not contain 'vocabulary' node");
  }

  // Read metadata
  int k = (int)fvoc["k"];
  int L = (int)fvoc["L"];
  DBoW2::ScoringType scoring = (DBoW2::ScoringType)((int)fvoc["scoringType"]);
  DBoW2::WeightingType weighting = (DBoW2::WeightingType)((int)fvoc["weightingType"]);

  std::cout << "  k=" << k << ", L=" << L << std::endl;

  // Phase 1: Collect all node iterators first
  cv::FileNode fn_nodes = fvoc["nodes"];
  size_t num_nodes = fn_nodes.size();
  std::cout << "  nodes=" << num_nodes << std::endl;
  
  auto phase1_start = std::chrono::high_resolution_clock::now();
  
  // Unfortunately, FileNodeIterator cannot be easily split across threads
  // because OpenCV's internal structure is not thread-safe for iteration
  // We'll collect nodes first, then process descriptors in parallel
  
  std::vector<NodeData> node_data;
  node_data.reserve(num_nodes);
  
  // Sequential collection (this is the bottleneck we can't easily parallelize)
  std::cout << "  Collecting nodes..." << std::flush;
  std::atomic<size_t> progress(0);
  size_t report_interval = num_nodes / 20;
  if (report_interval == 0) report_interval = 1000;
  
  for(cv::FileNodeIterator it = fn_nodes.begin(); it != fn_nodes.end(); ++it) {
    const cv::FileNode &n = *it;
    DBoW2::NodeId nid = (int)n["nodeId"];
    DBoW2::NodeId pid = (int)n["parentId"];
    DBoW2::WordValue weight = (DBoW2::WordValue)n["weight"];
    std::string d = (std::string)n["descriptor"];
    
    node_data.emplace_back(nid, pid, weight, d);
    
    size_t p = ++progress;
    if (p % report_interval == 0) {
      std::cout << "\r  Collecting nodes... " << (p * 100 / num_nodes) << "%" << std::flush;
    }
  }
  std::cout << "\r  Collecting nodes... 100%" << std::endl;
  
  auto phase1_end = std::chrono::high_resolution_clock::now();
  auto phase1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - phase1_start);
  std::cout << "  Phase 1 (collect): " << (phase1_duration.count() / 1000.0) << "s" << std::endl;
  
  // Phase 2: Parse descriptors in parallel
  auto phase2_start = std::chrono::high_resolution_clock::now();
  std::cout << "  Parsing descriptors in parallel..." << std::flush;
  
  // Create a temporary storage for parsed descriptors
  std::vector<DBoW2::FBRISK::TDescriptor> descriptors(num_nodes);
  
  // Parallel descriptor parsing
  auto parse_descriptors = [&](size_t start_idx, size_t end_idx) {
    for (size_t i = start_idx; i < end_idx; ++i) {
      DBoW2::FBRISK::fromString(descriptors[node_data[i].nid], node_data[i].descriptor_str);
    }
  };
  
  std::vector<std::thread> threads;
  size_t chunk_size = (num_nodes + num_threads - 1) / num_threads;
  
  for (int t = 0; t < num_threads; ++t) {
    size_t start_idx = t * chunk_size;
    size_t end_idx = std::min(start_idx + chunk_size, num_nodes);
    if (start_idx < num_nodes) {
      threads.emplace_back(parse_descriptors, start_idx, end_idx);
    }
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  
  auto phase2_end = std::chrono::high_resolution_clock::now();
  auto phase2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase2_end - phase2_start);
  std::cout << " done in " << (phase2_duration.count() / 1000.0) << "s" << std::endl;
  
  // Phase 3: Load words
  auto phase3_start = std::chrono::high_resolution_clock::now();
  cv::FileNode fn_words = fvoc["words"];
  size_t num_words = fn_words.size();
  std::cout << "  words=" << num_words << std::endl;
  std::cout << "  Loading words..." << std::flush;
  
  std::vector<std::pair<DBoW2::WordId, DBoW2::NodeId>> word_data;
  word_data.reserve(num_words);
  
  for(cv::FileNodeIterator it = fn_words.begin(); it != fn_words.end(); ++it) {
    const cv::FileNode &w = *it;
    DBoW2::WordId wid = (int)w["wordId"];
    DBoW2::NodeId nid = (int)w["nodeId"];
    word_data.emplace_back(wid, nid);
  }
  
  auto phase3_end = std::chrono::high_resolution_clock::now();
  auto phase3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase3_end - phase3_start);
  std::cout << " done in " << (phase3_duration.count() / 1000.0) << "s" << std::endl;
  
  fs.release();
  
  // Now load using the standard method (to properly construct the vocabulary)
  std::cout << "  Constructing vocabulary object..." << std::flush;
  auto phase4_start = std::chrono::high_resolution_clock::now();
  voc.load(filename);
  auto phase4_end = std::chrono::high_resolution_clock::now();
  auto phase4_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase4_end - phase4_start);
  std::cout << " done in " << (phase4_duration.count() / 1000.0) << "s" << std::endl;
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  std::cout << "\nTotal time: " << (total_duration.count() / 1000.0) << "s" << std::endl;
  std::cout << "Note: Standard load still needed for full construction" << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cout << "Usage: " << argv[0] << " <vocabulary_file> [num_threads]" << std::endl;
    std::cout << "Attempts to load vocabulary using parallel parsing." << std::endl;
    std::cout << "  num_threads: number of threads to use (default: hardware concurrency)" << std::endl;
    return -1;
  }

  std::string vocab_file = argv[1];
  int num_threads = std::thread::hardware_concurrency();
  
  if (argc >= 3) {
    num_threads = std::atoi(argv[2]);
    if (num_threads < 1) num_threads = 1;
  }

  std::cout << "Parallel Vocabulary Loader" << std::endl;
  std::cout << "==========================" << std::endl;
  std::cout << "File: " << vocab_file << std::endl;
  std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
  std::cout << "Using threads: " << num_threads << std::endl;
  std::cout << std::endl;

  try {
    FBriskVocabulary voc;
    load_vocabulary_parallel(voc, vocab_file, num_threads);
    
    std::cout << "\nVocabulary loaded successfully!" << std::endl;
    std::cout << "Words: " << voc.size() << std::endl;
    
  } catch(const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 4;
  } catch(const std::string &e) {
    std::cerr << "Exception: " << e << std::endl;
    return 4;
  }

  return 0;
}
