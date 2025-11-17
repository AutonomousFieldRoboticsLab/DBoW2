/**
 * @file trainBRISK_saveDescriptors.cpp
 * @brief Training application for BRISK descriptors with DBoW2 - saves descriptors to disk
 *
 * This version saves all extracted BRISK descriptors during training for later use.
 */

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

// DBoW2
#include "DBoW2.h" // defines core DBoW2 types
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// BRISK
#include <brisk/brisk.h>

// Boost filesystem (used below for directory iteration)
#include <boost/filesystem.hpp>

// Timing
#include <chrono>
#include <iomanip>

using namespace DBoW2;
using namespace std;

// Output filenames (can be overridden via CLI)
static std::string g_vocab_file = "small_voc.yml.gz";
static std::string g_db_file    = "small_db.yml.gz";
static std::string g_descriptors_file = "descriptors.yml.gz";

// \brief BRISK vocabulary.
typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskVocabulary;

/// \brief BRISK database.
typedef DBoW2::TemplatedDatabase<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskDatabase;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// \brief Load features from path.
void loadFeatures(const string &path, vector<vector<vector<unsigned char> > > &features,
                 vector<string> &image_names);

/// \brief Convert data structure.
void changeStructure(const cv::Mat &mat, vector<vector<unsigned char> > &out, int L);

/// \brief Save descriptors to file
void saveDescriptors(const string &filename, 
                    const vector<vector<vector<unsigned char> > > &features,
                    const vector<string> &image_names);

/// \brief Load descriptors from file
void loadDescriptors(const string &filename,
                    vector<vector<vector<unsigned char> > > &features,
                    vector<string> &image_names);

/// \brief Test vocabulary creation.
void testVocCreation(const vector<vector<vector<unsigned char> > > &features);

/// \brief Test database.
void testDatabase(const vector<vector<vector<unsigned char> > > &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

const int NIMAGES = 4; ///< number of training images
const int TESTIMAGES = 10; ///< number of test images

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 3) {
    std::cout << "Usage: ./" << argv[0] << " <dataset-folder> [base-name]" << std::endl;
    return -1;
  }

  std::string path(argv[1]);

  // Optional base name (no extension) for output files
  if (argc == 3) {
    const std::string base = argv[2];
    g_vocab_file = base + "_voc.yml.gz";
    g_db_file    = base + "_db.yml.gz";
    g_descriptors_file = base + "_descriptors.yml.gz";
  }

  vector<vector<vector<unsigned char> > > features;ave
  vector<string> image_names;
  
  // Check if descriptors already exist
  if (boost::filesystem::exists(g_descriptors_file)) {
    std::cout << "Found existing descriptors file: " << g_descriptors_file << std::endl;
    std::cout << "Loading descriptors from file..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    loadDescriptors(g_descriptors_file, features, image_names);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Loaded " << features.size() << " image descriptors in " 
              << (duration.count() / 1000.0) << " seconds" << std::endl;
  } else {
    // Extract features from images
    std::cout << "Extracting features from: " << path << std::endl;
    loadFeatures(path, features, image_names);
    
    std::cout << "Loaded " << features.size() << " images from '" << path << "'" << std::endl;
    
    // Save descriptors for future use
    std::cout << "Saving descriptors to: " << g_descriptors_file << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    saveDescriptors(g_descriptors_file, features, image_names);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Descriptors saved in " << (duration.count() / 1000.0) << " seconds" << std::endl;
  }

  testVocCreation(features);
  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(const string &path, vector<vector<vector<unsigned char> > > &features,
                 vector<string> &image_names)
{
  features.clear();
  image_names.clear();
  features.reserve(NIMAGES);
  image_names.reserve(NIMAGES);

  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> briskDetector(36, 0, 100, 700);
  brisk::BriskDescriptorExtractor briskDescriptorExtractor(false, false);

  size_t cnt = size_t(std::count_if(
          boost::filesystem::directory_iterator(path),
          boost::filesystem::directory_iterator(),
          static_cast<bool(*)(const boost::filesystem::path&)>(
                          boost::filesystem::is_regular_file)));

  cout << "Extracting BRISK features from " << cnt << " images..." << endl;
  int ctr = 0;
  for (auto it = boost::filesystem::directory_iterator(path);
      it != boost::filesystem::directory_iterator(); it++) {
    if (!boost::filesystem::is_directory(it->path())) {
      std::cout << "\r " << int(double(ctr)/double(cnt)*100.0) << "%, processing "
                << it->path().filename().string() << std::flush;
      
      cv::Mat image = cv::imread(path + "/" + it->path().filename().string(), cv::IMREAD_GRAYSCALE);
      
      if (image.empty()) {
        std::cerr << "\nWarning: Could not load image: " << it->path().filename().string() << std::endl;
        continue;
      }
      
      // Keypoints and descriptors
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      // Detect and compute descriptors
      briskDetector.detect(image, keypoints);
      briskDescriptorExtractor.compute(image, keypoints, descriptors);

      if (descriptors.empty()) {
        std::cerr << "\nWarning: No features found in: " << it->path().filename().string() << std::endl;
        continue;
      }

      // Convert and store
      features.push_back(vector<vector<unsigned char> >());
      changeStructure(descriptors, features.back(), 48);
      image_names.push_back(it->path().filename().string());

      ctr++;
    }
  }
  std::cout << std::endl;
  std::cout << "Total images processed: " << features.size() << std::endl;
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

void saveDescriptors(const string &filename,
                    const vector<vector<vector<unsigned char> > > &features,
                    const vector<string> &image_names)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  
  if (!fs.isOpened()) {
    throw std::string("Could not open file ") + filename + " for writing";
  }

  // Write metadata
  fs << "num_images" << (int)features.size();
  fs << "descriptor_length" << 48; // BRISK descriptor length
  
  // Write image names
  fs << "image_names" << "[";
  for (const auto& name : image_names) {
    fs << name;
  }
  fs << "]";
  
  // Write features for each image
  fs << "features" << "[";
  for (size_t img_idx = 0; img_idx < features.size(); ++img_idx) {
    fs << "{";
    fs << "image_id" << (int)img_idx;
    fs << "num_descriptors" << (int)features[img_idx].size();
    
    // Write descriptors as a matrix for this image
    if (!features[img_idx].empty()) {
      cv::Mat descriptors_mat(features[img_idx].size(), 48, CV_8U);
      for (size_t desc_idx = 0; desc_idx < features[img_idx].size(); ++desc_idx) {
        memcpy(descriptors_mat.ptr(desc_idx), features[img_idx][desc_idx].data(), 48);
      }
      fs << "descriptors" << descriptors_mat;
    }
    
    fs << "}";
  }
  fs << "]";
  
  fs.release();
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
  for (cv::FileNodeIterator it = fn_features.begin(); it != fn_features.end(); ++it) {
    cv::Mat descriptors_mat;
    (*it)["descriptors"] >> descriptors_mat;
    
    features.push_back(vector<vector<unsigned char> >());
    changeStructure(descriptors_mat, features.back(), 48);
  }
  
  fs.release();
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<vector<unsigned char> > > &features)
{
  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  FBriskVocabulary voc(k, L, weight, score);

  cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
  auto start = std::chrono::high_resolution_clock::now();
  voc.create(features);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const auto total_ms = duration.count();
  const auto minutes = total_ms / 60000;
  const double seconds_rem = (total_ms % 60000) / 1000.0;
  cout << "... done! Training took " << total_ms << " ms ("
       << fixed << setprecision(3) << (total_ms / 1000.0) << " s, "
       << minutes << " min " << setprecision(3) << seconds_rem << " s)" << endl;
  cout.unsetf(ios::fixed);

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // Test vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(size_t i = 0; i < TESTIMAGES && i < features.size(); i++)
  {
    voc.transform(features[i], v1);
    for(size_t j = 0; j < TESTIMAGES && j < features.size(); j++)
    {
      voc.transform(features[j], v2);
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // Save vocabulary
  cout << endl << "Saving vocabulary to '" << g_vocab_file << "'..." << endl;
  voc.save(g_vocab_file);
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<unsigned char> > > &features)
{
  cout << "Creating a small database..." << endl;

  // Load the vocabulary from disk
  FBriskVocabulary voc(g_vocab_file);

  // Create database
  FBriskDatabase db(voc, false, 0);

  // Add images to the database
  for(size_t i = 0; i < TESTIMAGES && i < features.size(); i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;
  cout << "Database information: " << endl << db << endl;

  // Query the database
  cout << "Querying the database: " << endl;
  QueryResults ret;
  for(size_t i = 0; i < TESTIMAGES && i < features.size(); i++)
  {
    db.query(features[i], ret, -1);
    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // Save database
  cout << "Saving database to '" << g_db_file << "'..." << endl;
  db.save(g_db_file);
  cout << "... done!" << endl;

  // Test reload
  cout << "Retrieving database once again..." << endl;
  FBriskDatabase db2(g_db_file);
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------
