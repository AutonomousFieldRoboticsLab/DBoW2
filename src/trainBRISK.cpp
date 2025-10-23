/**
 * @file dbow2_trainbrisk.cpp
 * @brief Training application for BRISK descriptors with DBoW2
 *
 * License: BSD, see https://github.com/dorian3d/DBoW2/blob/master/LICENSE.txt
 * @author Dorian Galvez-Lopez
 * @author Stefan Leutenegger
 *
 */

#include <iostream>
#include <vector>
#include <sstream>

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

using namespace DBoW2;
using namespace std;

// Output filenames (can be overridden via CLI)
static std::string g_vocab_file = "small_voc.yml.gz";
static std::string g_db_file    = "small_db.yml.gz";

// \brief BRISK vocabulary.
typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskVocabulary;

/// \brief BRISK database.
typedef DBoW2::TemplatedDatabase<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskDatabase;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// \brief Load features from path.
/// \param path Path.
/// @param[out] features The loaded features.
void loadFeatures(const string &path, vector<vector<vector<unsigned char> > > &features);

/// \brief Convert data structure.
/// \param mat cv::Mat format.
/// @param[out] out DBoW format.
/// @param[in] L No. descriptor bytes.
void changeStructure(const cv::Mat &mat, vector<vector<unsigned char> > &out,
  int L);

/// \brief Test vocabulary creation.
/// \param features Features.
void testVocCreation(const vector<vector<vector<unsigned char> > > &features);

/// \brief Test database.
/// \param features Features.
void testDatabase(const vector<vector<vector<unsigned char> > > &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

const int NIMAGES = 4; ///< number of training images

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
const int TESTIMAGES = 5; ///< number of test images

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// \brief Main
/// \param argc argc.
/// \param argv argv.
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
  }

  vector<vector<vector<unsigned char> > > features;
  loadFeatures(path, features);

  // Set NIMAGES from the number of loaded images
  // NIMAGES = static_cast<int>(features.size());
  // NIMAGES = 5000;
  std::cout << "Loaded " << NIMAGES << " images from '" << path << "'" << std::endl;

  testVocCreation(features);

  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(const string &path, vector<vector<vector<unsigned char> > > &features)
{
  features.clear();
  features.reserve(NIMAGES);
  // Reserve capacity after counting files below

  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> briskDetector(36, 0, 100,700);
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
    if (!boost::filesystem::is_directory(it->path())) {  //we eliminate directories
      std::cout << "\r " << int(double(ctr)/double(cnt)*100.0) << "%, processing "
                << it->path().filename().string();
      cv::Mat image = cv::imread(path + "/" + it->path().filename().string(), cv::IMREAD_GRAYSCALE);
      
      // Keypoints and descriptors
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      // Load keypoints and descriptors
      briskDetector.detect(image,keypoints);
      briskDescriptorExtractor.compute(image,keypoints,descriptors);

      // 
      features.push_back(vector<vector<unsigned char> >());
      changeStructure(descriptors, features.back(), 48);

      ctr++;
    }
  }
  std::cout  << std::endl;
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat& mat, vector<vector<unsigned char> > &out,
  int L)
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

void testVocCreation(const vector<vector<vector<unsigned char> > > &features)
{
  // branching factor and depth levels 
  // Total no. of words = k^L = 10^6 = 1 million
  const int k = 8; // 9
  const int L = 3; // 3
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  FBriskVocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(size_t i = 0; i < TESTIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(size_t j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary to '" << g_vocab_file << "'..." << endl;
  voc.save(g_vocab_file);
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<unsigned char> > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  FBriskVocabulary voc(g_vocab_file);

  // Create a copy of vocabulary into database
  FBriskDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(size_t i = 0; i < TESTIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(size_t i = 0; i < TESTIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database to '" << g_db_file << "'..." << endl;
  db.save(g_db_file);
  cout << "... done!" << endl;

  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  FBriskDatabase db2(g_db_file);
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------

