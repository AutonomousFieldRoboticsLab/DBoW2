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

// \brief BRISK vocabulary.
typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskVocabulary;

/// \brief BRISK database.
typedef DBoW2::TemplatedDatabase<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK>
  FBriskDatabase;


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <vocabulary_file>" << std::endl;
        return 1;
    }
    std::cout << "Loading vocabulary from file: " << argv[1] << std::endl;

    std::string g_vocab_file = argv[1];

    // Fast pre-check: open file and inspect sizes to avoid long/blocking loads
    try {
      cv::FileStorage fs(g_vocab_file, cv::FileStorage::READ);
      if(!fs.isOpened()) {
        std::cerr << "Error: could not open vocabulary file '" << g_vocab_file << "'" << std::endl;
        return 2;
      }

      cv::FileNode fvoc = fs["vocabulary"];
      if(fvoc.empty()) {
        std::cerr << "Error: file does not contain a 'vocabulary' node or is not a valid DBoW2 vocabulary." << std::endl;
        return 3;
      }

      int k = (int)fvoc["k"];
      int L = (int)fvoc["L"];
      size_t nodes = (size_t)fvoc["nodes"].size();
      size_t words = (size_t)fvoc["words"].size();

      // crude memory estimate (bytes): assume ~48 bytes per word descriptor + overhead per node
      const size_t est_bytes = words * 64 + nodes * 64;
      const double est_mb = double(est_bytes) / (1024.0*1024.0);

      std::cout << "Vocabulary file: '" << g_vocab_file << "'" << std::endl
                << "  k=" << k << ", L=" << L << ", nodes=" << nodes << ", words=" << words << std::endl
                << "  Estimated memory to load: ~" << int(est_mb) << " MB (rough)" << std::endl;

      // require explicit confirmation to proceed for very large vocabs
      const size_t WARN_THRESHOLD_WORDS = 200000; // arbitrary threshold
      bool force = false;
      for(int i = 2; i < argc; ++i) {
        if(std::string(argv[i]) == std::string("--force") || std::string(argv[i]) == std::string("-f")) force = true;
      }
      if(words > WARN_THRESHOLD_WORDS && !force) {
        std::cout << "The vocabulary appears large (> " << WARN_THRESHOLD_WORDS << " words)." << std::endl
                  << "Re-run with '--force' (or '-f') to actually load it and print details." << std::endl;
        return 0;
      }

      // proceed to load (this may take time)
      std::cout << "Loading vocabulary (this may take some time)..." << std::endl;
    }
    catch(const std::exception &e) {
      std::cerr << "Exception while inspecting vocabulary: " << e.what() << std::endl;
      return 4;
    }

    // Note: FBriskVocabulary constructor calls TemplatedVocabulary::load which may be very slow
    // for large YAML files due to O(n²) FileNode access patterns in OpenCV.
    FBriskVocabulary voc(g_vocab_file);
    std::cout << "Vocabulary loaded from file: " << g_vocab_file << std::endl;
    std::cout << "Vocabulary info: " << std::endl << voc << std::endl;
    
    return 0;
}