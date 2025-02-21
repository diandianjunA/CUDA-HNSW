project(cuda-hnsw LANGUAGES CXX)

# required cmake version
cmake_minimum_required(VERSION 3.2)


add_subdirectory(src/cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)

file(GLOB CUDA_HNSW_SOURCES src/cuda_hnsw_index.h src/cuda_hnsw_index.cpp src/hnswlib/*.h)


ADD_EXECUTABLE(cuda_hnsw src/main.cpp ${CUDA_HNSW_SOURCES})


target_link_libraries (cuda_hnsw ${EXTRA_LIBS})
