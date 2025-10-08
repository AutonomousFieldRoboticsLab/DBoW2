/**
 * File: FBRISK.cpp
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for BRISK descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

// #include <stdint.h>
// #include <limits.h>

#include <brisk/brisk.h>
#include "FBRISK.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FBRISK::meanValue(const std::vector<FBRISK::pDescriptor> &descriptors,
  FBRISK::TDescriptor &mean)
{
  mean.resize(0);
  mean.resize(L, 0);

  uint64_t s = descriptors.size()/2;

  std::vector<uint64_t> sum(L*8,0);

  // sum
  vector<FBRISK::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FBRISK::TDescriptor &desc = **it;
    for(size_t i = 0; i < L; i++)
    {
      for(size_t b=0; b<8; ++b)
      {
        sum[i*8+b] += (desc[i]&(0x01<<b)) ? 1 : 0;
      }
    }
  }

  // average
  for(size_t i = 0; i < L; i++)
  {
    for(int b=0; b<8; ++b)
    {
      if(sum[i*8+size_t(b)]>s)
      {
        mean[i] |= (0x01<<b);
      }
    }
  }
}

// // --------------------------------------------------------------------------
  
double FBRISK::distance(const FBRISK::TDescriptor &a, const FBRISK::TDescriptor &b)
{
  return double(brisk::Hamming::PopcntofXORed(&a.front(),&b.front(),L/16));
}

// // --------------------------------------------------------------------------
  
std::string FBRISK::toString(const FBRISK::TDescriptor &a)
{
  stringstream ss;
  for(size_t i = 0; i < L; ++i)
  {
    ss << int(a[i]) << " ";
  }
  return ss.str();
}

// // --------------------------------------------------------------------------
  
void FBRISK::fromString(FBRISK::TDescriptor &a, const std::string &s)
{
  a.resize(L);

  stringstream ss(s);
  for(size_t i = 0; i < L; ++i)
  {
    int tmp;
    ss >> tmp;
    a[i] = uint8_t(tmp);
  }
}

// // --------------------------------------------------------------------------

void FBRISK::toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const int N = int(descriptors.size());

  mat.create(N, L*8, CV_32F); // 8bit per byte

  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[size_t(i)];
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < L*8; j+=8, p+=8)
    {
      for(int b=0; b<8; ++b)
      {
        *(p+b) = (desc[size_t(j)]&(0x01<<b));
      }
    }
  }
}

// --------------------------------------------------------------------------

} // namespace DBoW2

