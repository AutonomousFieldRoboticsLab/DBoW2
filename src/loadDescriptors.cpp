/**
 * @file loadDescriptors.cpp
 * @brief Load and inspect saved BRISK descriptors from file
 */

#include <iostream>
#include <vector>
#include <string>

// DBoW2
#include "DBoW2.h"
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>

using namespace std;

void loadDescriptors(const string &filename,
                    vector<vector<vector<unsigned char> > > &features,
                    vector<string> &image_names);

void changeStructure(const cv::Mat& mat, vector<vector<unsigned char> > &out, int L);

int main(int argc, char **argv)
{
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <descriptors.yml.gz>" << std::endl;
    std::cout << "Loads and displays information about saved BRISK descriptors." << std::endl;
    return -1;
  }

  std::string descriptor_file = argv[1];
  
  std::cout << "Loading descriptors from: " << descriptor_file << std::endl;
  
  vector<vector<vector<unsigned char> > > features;
  vector<string> image_names;
  
  try {
    loadDescriptors(descriptor_file, features, image_names);
    
    std::cout << "\n=== Descriptor File Summary ===" << std::endl;
    std::cout << "Total images: " << features.size() << std::endl;
    std::cout << "Total descriptors: ";
    
    size_t total_descriptors = 0;
    size_t min_desc = SIZE_MAX;
    size_t max_desc = 0;
    
    for (const auto& img_features : features) {
      total_descriptors += img_features.size();
      if (img_features.size() < min_desc) min_desc = img_features.size();
      if (img_features.size() > max_desc) max_desc = img_features.size();
    }
    
    std::cout << total_descriptors << std::endl;
    std::cout << "Avg descriptors per image: " << (total_descriptors / features.size()) << std::endl;
    std::cout << "Min descriptors: " << min_desc << std::endl;
    std::cout << "Max descriptors: " << max_desc << std::endl;
    std::cout << "Descriptor length: 48 bytes (BRISK)" << std::endl;
    
    // Show first 10 image names
    std::cout << "\n=== Image Names (first 10) ===" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), image_names.size()); ++i) {
      std::cout << i << ": " << image_names[i] 
                << " (" << features[i].size() << " descriptors)" << std::endl;
    }
    
    if (image_names.size() > 10) {
      std::cout << "... and " << (image_names.size() - 10) << " more images" << std::endl;
    }
    
    // Show first descriptor of first image (sample)
    if (!features.empty() && !features[0].empty()) {
      std::cout << "\n=== Sample Descriptor (Image 0, Descriptor 0) ===" << std::endl;
      std::cout << "First 16 bytes: ";
      for (size_t i = 0; i < std::min(size_t(16), features[0][0].size()); ++i) {
        printf("%02x ", features[0][0][i]);
      }
      std::cout << std::endl;
    }
    
    std::cout << "\n=== Memory Usage ===" << std::endl;
    size_t memory_bytes = total_descriptors * 48;
    double memory_mb = memory_bytes / (1024.0 * 1024.0);
    std::cout << "Descriptors in memory: " << memory_mb << " MB" << std::endl;
    
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  } catch (const std::string &e) {
    std::cerr << "Error: " << e << std::endl;
    return -1;
  }

  return 0;
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat& mat, vector<vector<unsigned char> > &out, int L)
{
  out.resize(size_t(mat.rows));

  unsigned int j = 0;
  for(int i = 0; i < mat.rows*mat.cols; i += L, ++j)
  {
    out[j].resize(size_t(L));
    std::copy(mat.data + i, mat.data + i + L, out[j].begin());
  }
}

// ----------------------------------------------------------------------------

void loadDescriptors(const string &filename,
                    vector<vector<vector<unsigned char> > > &features,
                    vector<string> &image_names)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  
  if (!fs.isOpened()) {
    throw std::string("Could not open file ") + filename + " for reading";
  }

  int num_images = (int)fs["num_images"];
  std::cout << "Reading " << num_images << " images from file..." << std::endl;
  
  // Read image names
  cv::FileNode fn_names = fs["image_names"];
  image_names.clear();
  for (cv::FileNodeIterator it = fn_names.begin(); it != fn_names.end(); ++it) {
    image_names.push_back((string)*it);
  }
  
  // Read features
  features.clear();
  features.reserve(num_images);
  
  cv::FileNode fn_features = fs["features"];
  int progress = 0;
  for (cv::FileNodeIterator it = fn_features.begin(); it != fn_features.end(); ++it) {
    cv::Mat descriptors_mat;
    (*it)["descriptors"] >> descriptors_mat;
    
    features.push_back(vector<vector<unsigned char> >());
    changeStructure(descriptors_mat, features.back(), 48);
    
    progress++;
    if (progress % 1000 == 0 || progress == num_images) {
      std::cout << "\rLoaded " << progress << "/" << num_images << " images..." << std::flush;
    }
  }
  std::cout << std::endl;
  
  fs.release();
  
  std::cout << "Successfully loaded " << features.size() << " images" << std::endl;
}
