/**
 * File: trainBRISK.cpp
 * Date: October 2025
 * Author: Modified from Demo.cpp by Dorian Galvez-Lopez
 * Description: Training application for BRISK descriptors with DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <sstream>

// DBoW2
#include "DBoW2.h" // defines BriskVocabulary and BriskDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// BRISK
#include <brisk/brisk.h>

using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeaturesBRISK(vector<vector<FBRISK::TDescriptor> > &features);
void changeStructureBRISK(const cv::Mat &plain, vector<FBRISK::TDescriptor> &out);
void testVocCreationBRISK(const vector<vector<FBRISK::TDescriptor> > &features);
void testDatabaseBRISK(const vector<vector<FBRISK::TDescriptor> > &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 10000;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
  vector<vector<FBRISK::TDescriptor> > features;
  loadFeaturesBRISK(features);

  testVocCreationBRISK(features);

  wait();

  testDatabaseBRISK(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeaturesBRISK(vector<vector<FBRISK::TDescriptor> > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  // Create 3rdparty BRISK detector and descriptor extractor
  // Parameters: threshold, octaves, suppressScaleNonmaxima
  brisk::BriskFeatureDetector detector(60, 3, true);
  // Parameters: rotationInvariant, scaleInvariant, version
  brisk::BriskDescriptorExtractor extractor(true, true, brisk::BriskDescriptorExtractor::briskV2);

  cout << "Extracting BRISK features using 3rdparty library..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "/media/cmb/T71/singularity_data/dataset/DBOW2/Pamir2/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    if(image.empty())
    {
      cout << "Could not load image: " << ss.str() << endl;
      continue;
    }

    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Detect keypoints using 3rdparty BRISK detector
    detector.detect(image, keypoints, mask);
    
    // Compute descriptors using 3rdparty BRISK descriptor extractor
    extractor.compute(image, keypoints, descriptors);

    cout << "Image " << i << ": " << keypoints.size() << " keypoints, " 
         << descriptors.rows << " descriptors" << endl;

    if(!descriptors.empty())
    {
      features.push_back(vector<FBRISK::TDescriptor>());
      changeStructureBRISK(descriptors, features.back());
    }
  }
  
  cout << "Loaded " << features.size() << " images with features" << endl;
}

// ----------------------------------------------------------------------------

void changeStructureBRISK(const cv::Mat &plain, vector<FBRISK::TDescriptor> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i].resize(FBRISK::L);  // BRISK descriptor length is 48 bytes
    
    // Copy descriptor data
    const unsigned char* desc_data = plain.ptr<unsigned char>(i);
    for(int j = 0; j < FBRISK::L; ++j)
    {
      out[i][j] = desc_data[j];
    }
  }
}

// ----------------------------------------------------------------------------

void testVocCreationBRISK(const vector<vector<FBRISK::TDescriptor> > &features)
{
  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  BriskVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small BRISK " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "BRISK Vocabulary information: " << endl
  << voc << endl << endl;

  // Test vocabulary by matching images against themselves
  // cout << "Matching images against themselves with BRISK (0 low, 1 high): " << endl;
  // BowVector v1, v2;
  // for(size_t i = 0; i < features.size(); i++)
  // {
  //   voc.transform(features[i], v1);
  //   for(size_t j = 0; j < features.size(); j++)
  //   {
  //     voc.transform(features[j], v2);
      
  //     double score = voc.score(v1, v2);
  //     cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  // }

  // save the vocabulary to disk
  cout << endl << "Saving BRISK vocabulary..." << endl;
  voc.save("brisk_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabaseBRISK(const vector<vector<FBRISK::TDescriptor> > &features)
{
  cout << "Creating a small BRISK database..." << endl;

  // load the vocabulary from disk
  BriskVocabulary voc("brisk_voc.yml.gz");
  
  BriskDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(size_t i = 0; i < features.size(); i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "BRISK Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the BRISK database: " << endl;

  QueryResults ret;
  for(size_t i = 0; i < features.size(); i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving BRISK database..." << endl;
  db.save("brisk_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving BRISK database once again..." << endl;
  BriskDatabase db2("brisk_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------
