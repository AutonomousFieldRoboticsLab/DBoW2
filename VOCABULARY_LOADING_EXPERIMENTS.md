# Vocabulary Loading Speed Experiments

## Summary
Experiments to improve DBoW2 vocabulary loading speed using different file formats and approaches.

## Test Dataset
- File: `shipwreck_10_6_voc.yml.gz`
- Words: 999,609
- Original size: 77 MB (gzipped)
- Parameters: k=10, L=6

## Results

### Loading Time Comparison

| Format | Avg Load Time | vs Baseline | File Size | Notes |
|--------|---------------|-------------|-----------|-------|
| YAML (gzipped) | 21.447s | baseline | 77 MB | Original format |
| YAML (uncompressed) | 20.205s | **-5.8%** ✓ | 312 MB | Slightly faster |
| XML | 22.291s | +3.9% ✗ | 388 MB | Slower than YAML |

### Key Findings

1. **Uncompressed YAML is fastest** (~1.2s improvement)
   - Avoids gzip decompression overhead
   - Trade-off: 4x larger file size (77MB → 312MB)
   - Good for local fast storage (SSD)

2. **XML format is slower**
   - OpenCV's XML format is actually slower to parse
   - Much larger file size (388 MB)
   - Not recommended

3. **Iterator-based loading (already implemented)**
   - The commit 7d4e5f8 already optimized from O(n²) to O(n) 
   - This was the biggest improvement possible at the parsing level

## Tools Created

### 1. `benchmarkVocLoad`
Benchmark vocabulary loading speed with multiple iterations.

```bash
./benchmarkVocLoad <vocab_file> [iterations]
```

Example:
```bash
./benchmarkVocLoad shipwreck_10_6_voc.yml.gz 3
```

### 2. `convertVocToBinary`
Convert between YAML and XML formats.

```bash
./convertVocToBinary input.yml.gz output.xml
```

### 3. `saveToBinary`
Create uncompressed YAML from compressed version.

```bash
./saveToBinary input.yml.gz output.yml
```

Example:
```bash
./saveToBinary shipwreck_10_6_voc.yml.gz shipwreck_fast.yml
```

## Recommendations

### For Maximum Speed:
1. **Use uncompressed YAML** if you have disk space
   - ~6% faster loading
   - 4x larger files
   - Best for SSD storage

### For Balanced Performance:
1. **Keep compressed YAML** for storage/distribution
2. **Decompress to temp location** on first use
3. **Cache uncompressed version** on local SSD

### Future Optimization Ideas

#### 1. Memory-Mapped Binary Format
Create a true binary format that can be memory-mapped:
- Requires modifying DBoW2 internals to expose node structure
- Could achieve near-instant "loading" (just mmap the file)
- Implementation: Add binary serialization to TemplatedVocabulary

#### 2. Database Storage (LMDB/SQLite)
Store vocabulary in an embedded database:
- LMDB: Very fast memory-mapped key-value store
- SQLite: More portable, decent performance
- Allows partial loading of vocabulary

#### 3. Lazy Loading
Only load vocabulary nodes as needed:
- Load tree structure first (fast)
- Load descriptors on-demand
- Requires architecture changes

#### 4. Pre-compiled Binary Blob
Serialize the exact memory layout:
- Direct memory dump of vocabulary structure
- Fastest possible loading (just memcpy)
- Platform-specific (not portable)

#### 5. Parallel Loading
Use multiple threads to parse YAML:
- Split file into chunks
- Parse nodes in parallel
- Limited by YAML parsing library

## Code Example: Using Uncompressed Format

```cpp
// Convert once (offline)
FBriskVocabulary voc;
voc.load("original.yml.gz");
voc.save("fast_loading.yml");  // Uncompressed

// Use in production
FBriskVocabulary voc_fast;
voc_fast.load("fast_loading.yml");  // ~6% faster
```

## Benchmark Commands

```bash
# Baseline (compressed YAML)
./benchmarkVocLoad shipwreck_10_6_voc.yml.gz 3

# Create uncompressed version
./saveToBinary shipwreck_10_6_voc.yml.gz shipwreck_fast.yml

# Test uncompressed
./benchmarkVocLoad shipwreck_fast.yml 3

# Test XML (not recommended)
./convertVocToBinary shipwreck_10_6_voc.yml.gz shipwreck.xml
./benchmarkVocLoad shipwreck.xml 3
```

## Conclusion

For your 1M word vocabulary:
- **Quick win**: Use uncompressed YAML (~6% faster, 312MB vs 77MB)
- **Best approach**: Keep both versions (compressed for storage, uncompressed for loading)
- **Future work**: Implement true binary format with DBoW2 modifications

The current iterator-based loading (from commit 7d4e5f8) is already well-optimized for YAML parsing. Further significant improvements would require changing the file format entirely or modifying DBoW2's internal architecture.
