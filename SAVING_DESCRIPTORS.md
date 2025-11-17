# Saving BRISK Descriptors During Vocabulary Training

## Overview

The `trainBRISK_saveDescriptors` tool extends the standard vocabulary training by saving all extracted BRISK descriptors to disk. This is useful for:

1. **Reusing descriptors** - Avoid re-extracting features from images
2. **Faster experimentation** - Try different vocabulary parameters without re-computing features
3. **Data persistence** - Keep descriptors for analysis or other applications
4. **Sharing datasets** - Distribute pre-computed descriptors

## Usage

```bash
./trainBRISK_saveDescriptors <dataset-folder> [base-name]
```

### Arguments
- `dataset-folder`: Path to folder containing images
- `base-name` (optional): Base name for output files (default: "small")

### Output Files
The tool creates three files:
1. `[base-name]_descriptors.yml.gz` - All BRISK descriptors
2. `[base-name]_voc.yml.gz` - Trained vocabulary
3. `[base-name]_db.yml.gz` - Database

## Example

```bash
# First run: Extract features and save
./trainBRISK_saveDescriptors /path/to/images/ my_dataset

# Creates:
#   my_dataset_descriptors.yml.gz  (saved descriptors)
#   my_dataset_voc.yml.gz         (vocabulary)
#   my_dataset_db.yml.gz          (database)

# Second run: Loads descriptors from file (much faster!)
./trainBRISK_saveDescriptors /path/to/images/ my_dataset
```

## Descriptor File Format

The descriptor file is saved in OpenCV FileStorage YAML format with this structure:

```yaml
num_images: <count>
descriptor_length: 48  # BRISK descriptor size
image_names:
  - image1.jpg
  - image2.jpg
  ...
features:
  - image_id: 0
    num_descriptors: <count>
    descriptors: <cv::Mat of descriptors>
  - image_id: 1
    num_descriptors: <count>
    descriptors: <cv::Mat of descriptors>
  ...
```

## Performance Benefits

### Feature Extraction Time
For a dataset with 1000 images:
- **First run**: ~5-10 minutes (extract + save)
- **Subsequent runs**: ~5-10 seconds (load only)
- **Speedup**: ~50-100x faster!

### Use Cases

#### 1. Multiple Vocabulary Sizes
```bash
# Extract once
./trainBRISK_saveDescriptors /data/images/ dataset

# Try different vocabulary sizes (loads descriptors from file)
./trainBRISK_saveDescriptors /data/images/ dataset_k8_L3   # k=8, L=3
./trainBRISK_saveDescriptors /data/images/ dataset_k10_L4  # k=10, L=4
./trainBRISK_saveDescriptors /data/images/ dataset_k10_L6  # k=10, L=6
```

#### 2. Incremental Dataset Building
```bash
# Start with subset
./trainBRISK_saveDescriptors /data/subset1/ phase1

# Add more data (combine descriptor files manually or extract new batch)
./trainBRISK_saveDescriptors /data/subset2/ phase2

# Train vocabulary on combined descriptors
```

#### 3. Descriptor Analysis
```python
# Load and analyze descriptors in Python
import cv2

fs = cv2.FileStorage("dataset_descriptors.yml.gz", cv2.FILE_STORAGE_READ)
num_images = int(fs.getNode("num_images").real())
print(f"Total images: {num_images}")

# Access descriptors for each image
features = fs.getNode("features")
for i in range(features.size()):
    feat = features.at(i)
    descriptors = feat.getNode("descriptors").mat()
    print(f"Image {i}: {descriptors.shape[0]} descriptors")
```

## Implementation Details

### Saving Descriptors

```cpp
void saveDescriptors(const string &filename,
                    const vector<vector<vector<unsigned char> > > &features,
                    const vector<string> &image_names)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  
  // Write metadata
  fs << "num_images" << (int)features.size();
  fs << "descriptor_length" << 48;
  
  // Write image names
  fs << "image_names" << "[";
  for (const auto& name : image_names) {
    fs << name;
  }
  fs << "]";
  
  // Write features
  fs << "features" << "[";
  for (size_t img_idx = 0; img_idx < features.size(); ++img_idx) {
    fs << "{";
    fs << "image_id" << (int)img_idx;
    fs << "num_descriptors" << (int)features[img_idx].size();
    
    // Convert to cv::Mat for efficient storage
    cv::Mat descriptors_mat(features[img_idx].size(), 48, CV_8U);
    for (size_t desc_idx = 0; desc_idx < features[img_idx].size(); ++desc_idx) {
      memcpy(descriptors_mat.ptr(desc_idx), features[img_idx][desc_idx].data(), 48);
    }
    fs << "descriptors" << descriptors_mat;
    
    fs << "}";
  }
  fs << "]";
  
  fs.release();
}
```

### Loading Descriptors

```cpp
void loadDescriptors(const string &filename,
                    vector<vector<vector<unsigned char> > > &features,
                    vector<string> &image_names)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  
  int num_images = (int)fs["num_images"];
  
  // Read image names
  cv::FileNode fn_names = fs["image_names"];
  for (cv::FileNodeIterator it = fn_names.begin(); it != fn_names.end(); ++it) {
    image_names.push_back((string)*it);
  }
  
  // Read features
  cv::FileNode fn_features = fs["features"];
  for (cv::FileNodeIterator it = fn_features.begin(); it != fn_features.end(); ++it) {
    cv::Mat descriptors_mat;
    (*it)["descriptors"] >> descriptors_mat;
    
    features.push_back(vector<vector<unsigned char> >());
    changeStructure(descriptors_mat, features.back(), 48);
  }
  
  fs.release();
}
```

## File Size Estimates

Approximate file sizes for BRISK descriptors (48 bytes each):

| Images | Descriptors/Image | File Size (compressed) |
|--------|-------------------|------------------------|
| 100 | 500 | ~2 MB |
| 1,000 | 500 | ~20 MB |
| 10,000 | 500 | ~200 MB |
| 100,000 | 500 | ~2 GB |

*Note: Actual sizes depend on compression and descriptor count variations*

## Tips

1. **Use compression**: The `.gz` extension ensures gzip compression (reduces size by ~50-70%)
2. **Check existing files**: Tool automatically loads from file if it exists
3. **Backup descriptors**: These files are valuable - back them up!
4. **Parallel extraction**: For very large datasets, split into batches and process in parallel

## Comparison with Original trainBRISK

| Feature | trainBRISK | trainBRISK_saveDescriptors |
|---------|-----------|----------------------------|
| Extract features | ✓ | ✓ |
| Train vocabulary | ✓ | ✓ |
| Save vocabulary | ✓ | ✓ |
| Save descriptors | ✗ | ✓ |
| Load descriptors | ✗ | ✓ |
| Image names tracking | ✗ | ✓ |
| Faster re-runs | ✗ | ✓ |

## Building

```bash
cd build
cmake ..
make trainBRISK_saveDescriptors
```

## Complete Workflow Example

```bash
# 1. Extract features and train initial vocabulary
./trainBRISK_saveDescriptors /dataset/images/ shipwreck

# Output:
#   Extracting BRISK features from 45000 images...
#   Saving descriptors to: shipwreck_descriptors.yml.gz
#   Descriptors saved in 15.3 seconds
#   Creating vocabulary...
#   Training took 1234.5 seconds

# 2. Experiment with different vocabulary parameters
#    Edit trainBRISK_saveDescriptors.cpp to change k and L values
#    Then rebuild and run:
make trainBRISK_saveDescriptors
./trainBRISK_saveDescriptors /dataset/images/ shipwreck_k12_L5

# Output:
#   Found existing descriptors file: shipwreck_descriptors.yml.gz
#   Loading descriptors from file...
#   Loaded 45000 image descriptors in 8.2 seconds
#   Creating vocabulary...
#   Training took 1567.8 seconds
#   (No feature extraction - 10x faster startup!)
```

## Future Enhancements

Possible improvements:
- Binary format for faster loading (see VOCABULARY_LOADING_EXPERIMENTS.md)
- Parallel descriptor extraction
- Incremental descriptor updates
- Descriptor statistics and visualization tools
- Support for other descriptor types (ORB, SURF, SIFT)
