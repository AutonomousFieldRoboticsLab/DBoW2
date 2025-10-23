/**
 * Fast vocabulary loader that uses FileNodeIterator to avoid O(n²) access
 * This is a workaround for the slow TemplatedVocabulary::load in DBoW2
 * when loading very large vocabulary files.
 */

#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>

// DBoW2
#include "DBoW2.h"
#include "DBoW2/FBRISK.h"

// OpenCV
#include <opencv2/core.hpp>

using namespace DBoW2;

typedef DBoW2::TemplatedVocabulary<DBoW2::FBRISK::TDescriptor, DBoW2::FBRISK> FBriskVocabulary;

// Custom fast loader using iterator-based access
class FastVocabularyLoader {
public:
    static void load(FBriskVocabulary &voc, const std::string &filename) {
        auto start = std::chrono::high_resolution_clock::now();
        
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if(!fs.isOpened()) {
            throw std::string("Could not open file ") + filename;
        }

        cv::FileNode fvoc = fs["vocabulary"];
        if(fvoc.empty()) {
            throw std::string("File does not contain 'vocabulary' node");
        }

        int k = (int)fvoc["k"];
        int L = (int)fvoc["L"];
        ScoringType scoring = (ScoringType)((int)fvoc["scoringType"]);
        WeightingType weighting = (WeightingType)((int)fvoc["weightingType"]);

        std::cout << "Vocabulary parameters: k=" << k << ", L=" << L << std::endl;

        // Load nodes using iterator (O(n) instead of O(n²))
        cv::FileNode fn_nodes = fvoc["nodes"];
        size_t num_nodes = fn_nodes.size();
        std::cout << "Loading " << num_nodes << " nodes..." << std::flush;

        std::vector<NodeData> node_data;
        node_data.reserve(num_nodes + 1); // +1 for root
        
        // Root node
        node_data.push_back({0, 0, 0.0, ""});

        size_t count = 0;
        size_t report_interval = num_nodes / 20; // Report every 5%
        if(report_interval == 0) report_interval = 1000;

        for(cv::FileNodeIterator it = fn_nodes.begin(); it != fn_nodes.end(); ++it, ++count) {
            NodeId nid = (int)(*it)["nodeId"];
            NodeId pid = (int)(*it)["parentId"];
            WordValue weight = (WordValue)(*it)["weight"];
            std::string descriptor = (std::string)(*it)["descriptor"];

            if(nid >= node_data.size()) {
                node_data.resize(nid + 1);
            }
            node_data[nid] = {nid, pid, weight, descriptor};

            if(count % report_interval == 0) {
                std::cout << "\rLoading " << num_nodes << " nodes... " 
                          << (count * 100 / num_nodes) << "%" << std::flush;
            }
        }
        std::cout << "\rLoading " << num_nodes << " nodes... 100%" << std::endl;

        // Load words using iterator
        cv::FileNode fn_words = fvoc["words"];
        size_t num_words = fn_words.size();
        std::cout << "Loading " << num_words << " words..." << std::flush;

        std::vector<std::pair<WordId, NodeId>> word_data;
        word_data.reserve(num_words);

        count = 0;
        report_interval = num_words / 20;
        if(report_interval == 0) report_interval = 1000;

        for(cv::FileNodeIterator it = fn_words.begin(); it != fn_words.end(); ++it, ++count) {
            WordId wid = (int)(*it)["wordId"];
            NodeId nid = (int)(*it)["nodeId"];
            word_data.push_back({wid, nid});

            if(count % report_interval == 0) {
                std::cout << "\rLoading " << num_words << " words... " 
                          << (count * 100 / num_words) << "%" << std::flush;
            }
        }
        std::cout << "\rLoading " << num_words << " words... 100%" << std::endl;

        fs.release();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Vocabulary loaded in " << (duration.count() / 1000.0) << " seconds" << std::endl;
        std::cout << "  Nodes: " << node_data.size() << std::endl;
        std::cout << "  Words: " << word_data.size() << std::endl;
    }

private:
    struct NodeData {
        NodeId id;
        NodeId parent;
        WordValue weight;
        std::string descriptor_str;
    };
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <vocabulary_file>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "This is a fast vocabulary inspector that uses iterator-based" << std::endl;
        std::cerr << "FileNode access to avoid O(n²) performance with large YAML files." << std::endl;
        return 1;
    }

    std::string vocab_file = argv[1];
    std::cout << "Fast vocabulary loader" << std::endl;
    std::cout << "File: " << vocab_file << std::endl << std::endl;

    try {
        // Quick pre-check
        cv::FileStorage fs(vocab_file, cv::FileStorage::READ);
        if(!fs.isOpened()) {
            std::cerr << "Error: could not open file '" << vocab_file << "'" << std::endl;
            return 2;
        }

        cv::FileNode fvoc = fs["vocabulary"];
        if(fvoc.empty()) {
            std::cerr << "Error: file does not contain 'vocabulary' node" << std::endl;
            return 3;
        }

        int k = (int)fvoc["k"];
        int L = (int)fvoc["L"];
        size_t nodes = (size_t)fvoc["nodes"].size();
        size_t words = (size_t)fvoc["words"].size();

        std::cout << "Quick inspection:" << std::endl;
        std::cout << "  k=" << k << ", L=" << L << std::endl;
        std::cout << "  nodes=" << nodes << ", words=" << words << std::endl;
        std::cout << std::endl;

        fs.release();

        // Load using fast iterator-based access
        FBriskVocabulary voc;
        FastVocabularyLoader::load(voc, vocab_file);

        std::cout << std::endl << "Successfully inspected vocabulary structure." << std::endl;
        std::cout << "Note: Full TemplatedVocabulary construction not implemented in this tool." << std::endl;
        std::cout << "      This is a diagnostic/inspection tool only." << std::endl;

    } catch(const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 4;
    } catch(const std::string &e) {
        std::cerr << "Exception: " << e << std::endl;
        return 4;
    }

    return 0;
}
