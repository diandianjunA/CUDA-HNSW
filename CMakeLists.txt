project(cuda-hnsw LANGUAGES CXX)

# required cmake version
cmake_minimum_required(VERSION 3.2)


add_subdirectory(src/cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu faiss)

file(GLOB CUDA_HNSW_SOURCES src/cuda_hnsw_index.h src/cuda_hnsw_index.cpp src/hnswlib/*.h)

include_directories("/usr/local/include/")
include_directories("/usr/local/cuda-12.2/targets/x86_64-linux/include")
link_directories("./ThirdParty/OpenCV/lib")
link_directories("/usr/lib/x86_64-linux-gnu")
link_directories("/usr/local/lib")
link_directories("~/anaconda3/lib")


ADD_EXECUTABLE(cuda_hnsw src/main.cpp ${CUDA_HNSW_SOURCES})


target_link_libraries (cuda_hnsw ${EXTRA_LIBS} faiss_gpu cuvs cudart cublas cuda openblas)
