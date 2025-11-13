cmake_minimum_required(VERSION 3.0)
project(DBoW2)
include(ExternalProject)

option(BUILD_DBoW2   "Build DBoW2"            ON)
option(BUILD_Demo    "Build demo application" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(HDRS
  include/DBoW2/BowVector.h           include/DBoW2/FBrief.h
  include/DBoW2/QueryResults.h        include/DBoW2/TemplatedDatabase.h   include/DBoW2/FORB.h
  include/DBoW2/DBoW2.h               include/DBoW2/FClass.h              include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       include/DBoW2/TemplatedVocabulary.h include/DBoW2/FBRISK.h)
set(SRCS 
  src/BowVector.cpp     src/FBrief.cpp        src/FORB.cpp
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp src/FBRISK.cpp)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

# Add BRISK subdirectory (disable its demo to avoid conflicts)
set(BRISK_BUILD_DEMO OFF CACHE BOOL "Disable BRISK demo")
add_subdirectory(3rdparty/brisk)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(BUILD_DBoW2)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
  target_include_directories(${PROJECT_NAME} PUBLIC 
    include/DBoW2/ 
    include/
    3rdparty/brisk/include
    ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} brisk)
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
endif(BUILD_DBoW2)

if(BUILD_Demo)
  add_executable(dbow2_demo demo/demo.cpp)
  target_link_libraries(dbow2_demo ${PROJECT_NAME} ${OpenCV_LIBS} brisk)
  set_target_properties(dbow2_demo PROPERTIES CXX_STANDARD 11)
  file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)
endif(BUILD_Demo)

# Add BRISK training executable
add_executable(trainBRISK src/trainBRISK.cpp)
find_package(Boost COMPONENTS filesystem system REQUIRED)
target_link_libraries(trainBRISK ${PROJECT_NAME} ${OpenCV_LIBS} brisk Boost::filesystem Boost::system)
target_include_directories(trainBRISK PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(trainBRISK PROPERTIES CXX_STANDARD 11)


# Add BRISK loading executable
add_executable(loadBRISK src/loadBRISK.cpp)
find_package(Boost COMPONENTS filesystem system REQUIRED)
target_link_libraries(loadBRISK ${PROJECT_NAME} ${OpenCV_LIBS} brisk Boost::filesystem Boost::system)
target_include_directories(loadBRISK PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(loadBRISK PROPERTIES CXX_STANDARD 11)

# Add fast vocabulary inspector (uses iterator-based FileNode access)
add_executable(fastLoadBRISK src/fastLoadBRISK.cpp)
target_link_libraries(fastLoadBRISK ${PROJECT_NAME} ${OpenCV_LIBS} brisk)
target_include_directories(fastLoadBRISK PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(fastLoadBRISK PROPERTIES CXX_STANDARD 11)

# Add vocabulary format converter (YAML to binary/XML)
add_executable(convertVocToBinary src/convertVocToBinary.cpp)
target_link_libraries(convertVocToBinary ${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(convertVocToBinary PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(convertVocToBinary PROPERTIES CXX_STANDARD 11)

# Add vocabulary loading benchmark tool
add_executable(benchmarkVocLoad src/benchmarkVocLoad.cpp)
target_link_libraries(benchmarkVocLoad ${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(benchmarkVocLoad PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(benchmarkVocLoad PROPERTIES CXX_STANDARD 11)

# Add uncompressed vocabulary saver
add_executable(saveToBinary src/saveToBinary.cpp)
target_link_libraries(saveToBinary ${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(saveToBinary PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(saveToBinary PROPERTIES CXX_STANDARD 11)

# Add parallel vocabulary loader (experimental)
find_package(Threads REQUIRED)
add_executable(parallelLoadVocab src/parallelLoadVocab.cpp)
target_link_libraries(parallelLoadVocab ${PROJECT_NAME} ${OpenCV_LIBS} Threads::Threads)
target_include_directories(parallelLoadVocab PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(parallelLoadVocab PROPERTIES CXX_STANDARD 11)

# Add parallel loading benchmark
add_executable(benchmarkParallelLoad src/benchmarkParallelLoad.cpp)
target_link_libraries(benchmarkParallelLoad ${PROJECT_NAME} ${OpenCV_LIBS} Threads::Threads)
target_include_directories(benchmarkParallelLoad PRIVATE 
  include/DBoW2/ 
  include/
  3rdparty/brisk/include
  ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brisk/include)
set_target_properties(benchmarkParallelLoad PROPERTIES CXX_STANDARD 11)


configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DBoW2)
  install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)

