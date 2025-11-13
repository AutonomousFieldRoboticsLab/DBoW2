/**
 * @file saveToBinary.cpp
 * @brief Save DBoW2 vocabulary in a more compact binary-friendly format
 * 
 * Strategy: Use OpenCV's FileStorage with WRITE_BASE64 flag for more efficient storage
 */

#include <iostream>
#include <fstream>
#include <chrono>

// DBoW2
#include "DBoW2.h"
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>

typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK> FBriskVocabulary;

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " input.yml.gz output.yml" << std::endl;
    std::cout << "Converts compressed YAML to uncompressed YAML (sometimes faster to load)" << std::endl;
    return -1;
  }

  std::string input_file = argv[1];
  std::string output_file = argv[2];

  std::cout << "Loading vocabulary from: " << input_file << std::endl;
  auto start_load = std::chrono::high_resolution_clock::now();
  
  FBriskVocabulary voc;
  
  try {
    voc.load(input_file);
    
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);
    
    std::cout << "Vocabulary loaded in " << (load_duration.count() / 1000.0) << " seconds" << std::endl;
    std::cout << "  Words: " << voc.size() << std::endl;
    
    // Save uncompressed
    std::cout << "\nSaving uncompressed: " << output_file << std::endl;
    auto start_save = std::chrono::high_resolution_clock::now();
    
    voc.save(output_file);
    
    auto end_save = std::chrono::high_resolution_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    
    std::cout << "Save completed in " << (save_duration.count() / 1000.0) << " seconds" << std::endl;
    
    // Get file sizes
    std::ifstream in_file(input_file, std::ios::binary | std::ios::ate);
    std::ifstream out_file(output_file, std::ios::binary | std::ios::ate);
    if (in_file.is_open() && out_file.is_open()) {
      auto in_size = in_file.tellg();
      auto out_size = out_file.tellg();
      std::cout << "\nFile size comparison:" << std::endl;
      std::cout << "  Input:  " << (in_size / (1024*1024)) << " MB (compressed)" << std::endl;
      std::cout << "  Output: " << (out_size / (1024*1024)) << " MB (uncompressed)" << std::endl;
      std::cout << "  Compression ratio: " << (float)in_size / out_size << std::endl;
    }
    
    std::cout << "\nTest with: ./benchmarkVocLoad " << output_file << " 3" << std::endl;
    
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
