/**
 * @file saveBRISKDescriptors.cpp
 * @brief Save BRISK descriptors extracted from images to a yml.gz file
 *
 * This program extracts BRISK descriptors from all images in a directory
 * and saves them to a compressed yml.gz file for later use in vocabulary training.
 *
 * License: BSD, see https://github.com/dorian3d/DBoW2/blob/master/LICENSE.txt
 */

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// BRISK
#include <brisk/brisk.h>

// Boost filesystem (used for directory iteration)
#include <boost/filesystem.hpp>

// Timing
#include <chrono>
#include <iomanip>

using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// \brief Extract BRISK descriptors from images in path and save to file.
/// \param image_path Path containing images.
/// \param output_file Output yml.gz file path.
void extractAndSaveDescriptors(const string &image_path, const string &output_file);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 3) {
    std::cout << "Usage: " << argv[0] << " <dataset-folder> [output-file]" << std::endl;
    std::cout << "  dataset-folder: Directory containing images" << std::endl;
    std::cout << "  output-file: Output yml.gz file (default: brisk_descriptors.yml.gz)" << std::endl;
    return -1;
  }

  std::string image_path(argv[1]);
  std::string output_file = "brisk_descriptors.yml.gz";
  
  if (argc == 3) {
    output_file = argv[2];
    // Ensure it has .yml.gz extension
    if (output_file.find(".yml.gz") == std::string::npos) {
      output_file += ".yml.gz";
    }
  }

  std::cout << "Extracting BRISK descriptors from: " << image_path << std::endl;
  std::cout << "Output file: " << output_file << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  extractAndSaveDescriptors(image_path, output_file);
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const auto total_ms = duration.count();
  const auto minutes = total_ms / 60000;
  const double seconds_rem = (total_ms % 60000) / 1000.0;
  
  cout << "\nTotal time: " << total_ms << " ms ("
       << fixed << setprecision(3) << (total_ms / 1000.0) << " s, "
       << minutes << " min " << setprecision(3) << seconds_rem << " s)" << endl;
  cout.unsetf(ios::fixed);

  return 0;
}

// ----------------------------------------------------------------------------

void extractAndSaveDescriptors(const string &image_path, const string &output_file)
{
  // BRISK detector and extractor parameters (matching trainBRISK.cpp)
  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> briskDetector(36, 0, 100, 700);
  brisk::BriskDescriptorExtractor briskDescriptorExtractor(false, false);

  // Count total images
  size_t total_images = size_t(std::count_if(
    boost::filesystem::directory_iterator(image_path),
    boost::filesystem::directory_iterator(),
    static_cast<bool(*)(const boost::filesystem::path&)>(
      boost::filesystem::is_regular_file)));

  if (total_images == 0) {
    cerr << "Error: No images found in directory: " << image_path << endl;
    return;
  }

  cout << "Found " << total_images << " images" << endl;
  cout << "Extracting BRISK descriptors..." << endl;

  // Store all descriptors - one cv::Mat per image
  vector<cv::Mat> all_descriptors;
  vector<string> image_filenames;
  all_descriptors.reserve(total_images);
  image_filenames.reserve(total_images);

  int processed = 0;
  for (auto it = boost::filesystem::directory_iterator(image_path);
       it != boost::filesystem::directory_iterator(); ++it) {
    
    if (!boost::filesystem::is_directory(it->path())) {
      std::string filename = it->path().filename().string();
      std::string full_path = image_path + "/" + filename;
      
      // Progress indicator
      std::cout << "\r " << int(double(processed) / double(total_images) * 100.0) 
                << "%, processing " << filename << std::flush;

      // Read image in grayscale
      cv::Mat image = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
      
      if (image.empty()) {
        cerr << "\nWarning: Could not read image: " << filename << endl;
        continue;
      }

      // Extract keypoints and descriptors
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      
      briskDetector.detect(image, keypoints);
      briskDescriptorExtractor.compute(image, keypoints, descriptors);

      if (!descriptors.empty()) {
        all_descriptors.push_back(descriptors.clone());
        image_filenames.push_back(filename);
      } else {
        cerr << "\nWarning: No descriptors extracted from: " << filename << endl;
      }

      processed++;
    }
  }
  
  std::cout << std::endl;
  std::cout << "Extracted descriptors from " << all_descriptors.size() << " images" << std::endl;

  // Save all descriptors to file
  std::cout << "Saving descriptors to " << output_file << "..." << std::endl;
  
  cv::FileStorage fs(output_file, cv::FileStorage::WRITE);
  
  if (!fs.isOpened()) {
    cerr << "Error: Could not open file for writing: " << output_file << endl;
    return;
  }

  // Write metadata
  fs << "num_images" << static_cast<int>(all_descriptors.size());
  fs << "descriptor_type" << "BRISK";
  fs << "descriptor_size" << (all_descriptors.empty() ? 0 : all_descriptors[0].cols);
  
  // Write each image's descriptors as a separate entry
  for (size_t i = 0; i < all_descriptors.size(); i++) {
    std::stringstream ss;
    ss << "image_" << std::setw(6) << std::setfill('0') << i;
    fs << ss.str() << all_descriptors[i];
  }
  
  fs.release();
  
  std::cout << "Successfully saved descriptors!" << std::endl;
  
  // Print statistics
  int total_features = 0;
  for (const auto& desc : all_descriptors) {
    total_features += desc.rows;
  }
  
  std::cout << "\nStatistics:" << std::endl;
  std::cout << "  Total images: " << all_descriptors.size() << std::endl;
  std::cout << "  Total features: " << total_features << std::endl;
  std::cout << "  Average features per image: " 
            << (all_descriptors.empty() ? 0 : total_features / all_descriptors.size()) << std::endl;
}

// ----------------------------------------------------------------------------
