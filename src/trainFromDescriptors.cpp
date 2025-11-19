/**
 * @file trainFromDescriptors.cpp
 * @brief Train BRISK vocabulary from pre-extracted descriptors saved in yml.gz file
 *
 * This program loads BRISK descriptors from a yml.gz file (created by saveBRISKDescriptors)
 * and trains a vocabulary, avoiding the time-consuming feature extraction step.
 *
 * License: BSD, see https://github.com/dorian3d/DBoW2/blob/master/LICENSE.txt
 */

#include <iostream>
#include <vector>
#include <string>

// DBoW2
#include "DBoW2.h"
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>

// Timing
#include <chrono>
#include <iomanip>

using namespace DBoW2;
using namespace std;

// Output filenames (can be overridden via CLI)
static std::string g_vocab_file = "small_voc.yml.gz";
static std::string g_db_file    = "small_db.yml.gz";

// BRISK vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskVocabulary;

// BRISK database
typedef DBoW2::TemplatedDatabase<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskDatabase;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// \brief Load features from yml.gz file.
/// \param descriptor_file Path to yml.gz descriptor file.
/// @param[out] features The loaded features.
bool loadDescriptorsFromFile(const string &descriptor_file, 
                              vector<vector<vector<unsigned char> > > &features);

/// \brief Convert data structure.
/// \param mat cv::Mat format.
/// @param[out] out DBoW format.
/// @param[in] L No. descriptor bytes.
void changeStructure(const cv::Mat &mat, vector<vector<unsigned char> > &out, int L);

/// \brief Test vocabulary creation.
/// \param features Features.
void testVocCreation(const vector<vector<vector<unsigned char> > > &features);

/// \brief Test database.
/// \param features Features.
void testDatabase(const vector<vector<vector<unsigned char> > > &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

const int TESTIMAGES = 10; ///< number of test images

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 3) {
    std::cout << "Usage: " << argv[0] << " <descriptors.yml.gz> [base-name]" << std::endl;
    std::cout << "  descriptors.yml.gz: Input file with pre-extracted BRISK descriptors" << std::endl;
    std::cout << "  base-name: Optional base name for output vocabulary and database files" << std::endl;
    return -1;
  }

  std::string descriptor_file(argv[1]);

  // Optional base name (no extension) for output files
  if (argc == 3) {
    const std::string base = argv[2];
    g_vocab_file = base + "_voc.yml.gz";
    g_db_file    = base + "_db.yml.gz";
  }

  vector<vector<vector<unsigned char> > > features;
  
  std::cout << "Loading descriptors from: " << descriptor_file << std::endl;
  auto load_start = std::chrono::high_resolution_clock::now();
  
  if (!loadDescriptorsFromFile(descriptor_file, features)) {
    std::cerr << "Failed to load descriptors from file!" << std::endl;
    return -1;
  }
  
  auto load_end = std::chrono::high_resolution_clock::now();
  auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
  std::cout << "Loaded " << features.size() << " images in " << load_duration.count() << " ms" << std::endl;

  testVocCreation(features);

  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

bool loadDescriptorsFromFile(const string &descriptor_file, 
                              vector<vector<vector<unsigned char> > > &features)
{
  features.clear();

  cv::FileStorage fs(descriptor_file, cv::FileStorage::READ);
  
  if (!fs.isOpened()) {
    cerr << "Error: Could not open descriptor file: " << descriptor_file << endl;
    return false;
  }

  // Read metadata
  int num_images = (int)fs["num_images"];
  string descriptor_type = (string)fs["descriptor_type"];
  int descriptor_size = (int)fs["descriptor_size"];
  
  std::cout << "File metadata:" << std::endl;
  std::cout << "  Number of images: " << num_images << std::endl;
  std::cout << "  Descriptor type: " << descriptor_type << std::endl;
  std::cout << "  Descriptor size: " << descriptor_size << std::endl;

  if (descriptor_type != "BRISK") {
    cerr << "Warning: Descriptor type is not BRISK!" << endl;
  }

  features.reserve(num_images);

  // Read descriptors for each image
  int total_features = 0;
  for (int i = 0; i < num_images; i++) {
    std::stringstream ss;
    ss << "image_" << std::setw(6) << std::setfill('0') << i;
    
    cv::Mat desc_mat;
    fs[ss.str()] >> desc_mat;
    
    if (!desc_mat.empty()) {
      features.push_back(vector<vector<unsigned char> >());
      changeStructure(desc_mat, features.back(), descriptor_size);
      total_features += desc_mat.rows;
    } else {
      cerr << "Warning: No descriptors found for " << ss.str() << endl;
    }
  }

  fs.release();
  
  std::cout << "Successfully loaded descriptors!" << std::endl;
  std::cout << "  Total features: " << total_features << std::endl;
  std::cout << "  Average features per image: " 
            << (features.empty() ? 0 : total_features / features.size()) << std::endl;

  return true;
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

void testVocCreation(const vector<vector<vector<unsigned char> > > &features)
{
  // branching factor and depth levels 
  // Total no. of words = k^L = 10^6 = 1 million
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

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  int test_images = min(TESTIMAGES, (int)features.size());
  for(int i = 0; i < test_images; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < test_images; j++)
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
  int test_images = min(TESTIMAGES, (int)features.size());
  for(int i = 0; i < test_images; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < test_images; i++)
  {
    db.query(features[i], ret, -1); // max 10 results

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
