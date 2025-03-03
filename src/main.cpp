#include "cuda_hnsw_index.h"
#include <chrono>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <iostream>
#include <stdio.h>

#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/random.h>

void test1() {
  int num_vectors = 10000;
  int dim = 128;

  CUDAHNSWIndex *index = new CUDAHNSWIndex(dim, num_vectors, 16, 200);

  std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(dim));

  // 用随机数据填充向量
  for (int i = 0; i < num_vectors; ++i) {
    for (int j = 0; j < dim; ++j) {
      vectors[i][j] = static_cast<float>(rand()) / RAND_MAX; // 随机数填充
    }
  }

  std::cout << "generate vectors done" << std::endl;

  // 批量插入向量
  for (int i = 0; i < num_vectors; ++i) {
    index->insert_vectors(vectors[i].data(), i); // 向量和对应的ID
  }

  std::cout << "insert vectors done" << std::endl;
  // index->check();

  // 测试查询

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    std::vector<float> query(dim);
    for (int j = 0; j < dim; ++j) {
      query[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    std::pair<std::vector<long>, std::vector<float>> result =
        index->search_vectors(query, 5, 20); // 找5个最近邻
  }
  auto end = std::chrono::high_resolution_clock::now();
  // 输出搜索时间，单位为纳秒
  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count()
            << "ns" << std::endl;

  std::cout << "------------------" << std::endl;

  index->init_gpu(); // 初始化GPU

  auto start_gpu = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    std::vector<float> query(dim);
    for (int j = 0; j < dim; ++j) {
      query[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    std::pair<std::vector<long>, std::vector<float>> result =
        index->search_vectors_gpu(query, 5, 20); // 找5个最近邻
  }
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu -
                                                                    start_gpu)
                   .count()
            << "ns" << std::endl;
}

// 生成指定大小的随机128维浮点向量
void generate_random_vectors(size_t num_vectors, size_t dimension,
                             std::vector<float> &vectors,
                             std::vector<faiss::idx_t> &ids) {
  // 随机数生成器
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0); // 随机数范围 -1 to 1

  // 生成随机向量和ID
  vectors.resize(num_vectors * dimension);
  ids.resize(num_vectors);

  for (size_t i = 0; i < num_vectors; ++i) {
    ids[i] = i; // ID设置为序号
    for (size_t j = 0; j < dimension; ++j) {
      vectors[i * dimension + j] = dis(gen); // 每个维度生成一个随机数
    }
  }
}

void test2() {
  size_t num_vectors = 10000;
  size_t dimension = 1000;

  auto index_ = new faiss::IndexHNSWFlat(dimension, 16);
  index_->hnsw.efConstruction = 200;
  index_->hnsw.efSearch = 20;
  faiss::Index *index = new faiss::IndexIDMap(index_);

  std::vector<float> vectors;
  std::vector<faiss::idx_t> ids;
  generate_random_vectors(num_vectors, dimension, vectors, ids);

  index->add_with_ids(num_vectors, vectors.data(), ids.data());

  int num_query = 1000;
  std::vector<float> query(num_query * dimension);
  for (int j = 0; j < num_query; ++j) {
    for (int k = 0; k < dimension; ++k) {
      query[j * dimension + k] = static_cast<float>(rand()) / RAND_MAX;
    }
  }
  std::vector<faiss::idx_t> I(5 * num_query);
  std::vector<float> D(5 * num_query);
  // 测试查询
  auto start = std::chrono::high_resolution_clock::now();
  index->search(num_query, query.data(), 5, D.data(), I.data());
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count()
            << "ns" << std::endl;

  delete index;

  // 使用GPU
  CUDAHNSWIndex *cuindex = new CUDAHNSWIndex(dimension, num_vectors, 16, 200);

  std::vector<size_t> size_t_ids(ids.begin(), ids.end());
  cuindex->insert_vectors_batch(vectors, size_t_ids);

  cuindex->init_gpu();
  auto start_gpu = std::chrono::high_resolution_clock::now();
  cuindex->search_vectors_batch_gpu(query, 5, 20, true);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu -
                                                                    start_gpu)
                   .count()
            << "ns" << std::endl;
}

void test3() {
  int num_vectors = 1000;
  int dim = 128;

  CUDAHNSWIndex *index = new CUDAHNSWIndex(dim, num_vectors, 16, 200);

  std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(dim));

  // 用随机数据填充向量
  for (int i = 0; i < num_vectors; ++i) {
    for (int j = 0; j < dim; ++j) {
      vectors[i][j] = static_cast<float>(rand()) / RAND_MAX; // 随机数填充
    }
  }

  std::cout << "generate vectors done" << std::endl;

  // 批量插入向量
  for (int i = 0; i < num_vectors; ++i) {
    index->insert_vectors(vectors[i].data(), i); // 向量和对应的ID
  }

  std::cout << "insert vectors done" << std::endl;

  // 测试查询

  index->init_gpu();

  int num_query = 1000;
  std::vector<float> query(num_query * dim);
  for (int j = 0; j < num_query; ++j) {
    for (int k = 0; k < dim; ++k) {
      query[j * dim + k] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  index->search_vectors_batch_gpu(query, 5, 20);
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count()
            << "ns" << std::endl;
}

void test4() {
  int num_vectors = 1000;
  int dim = 128;

  CUDAHNSWIndex *index = new CUDAHNSWIndex(dim, num_vectors, 16, 200);

  std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(dim));

  // 用随机数据填充向量
  for (int i = 0; i < num_vectors; ++i) {
    for (int j = 0; j < dim; ++j) {
      vectors[i][j] = static_cast<float>(rand()) / RAND_MAX; // 随机数填充
    }
  }

  std::cout << "generate vectors done" << std::endl;

  // 批量插入向量
  for (int i = 0; i < num_vectors; ++i) {
    index->insert_vectors(vectors[i].data(), i); // 向量和对应的ID
  }

  std::cout << "insert vectors done" << std::endl;

  // 测试查询
  std::vector<float> query(dim);
  for (int j = 0; j < dim; ++j) {
    query[j] = static_cast<float>(rand()) / RAND_MAX;
  }

  auto result = index->search_vectors(query, 5, 20);
  for (int i = 0; i < 5; ++i) {
    std::cout << "id: " << result.first[i] << ", distance: " << result.second[i]
              << std::endl;
  }

  index->init_gpu();
  auto result_gpu = index->search_vectors_gpu(query, 5, 20);
  for (int i = 0; i < 5; ++i) {
    std::cout << "id: " << result_gpu.first[i]
              << ", distance: " << result_gpu.second[i] << std::endl;
  }
}

void test5() {
  int num_vectors = 10000;
  int dim = 128;

  CUDAHNSWIndex *index = new CUDAHNSWIndex(dim, num_vectors, 16, 200);

  std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(dim));

  // 用随机数据填充向量
  for (int i = 0; i < num_vectors; ++i) {
    for (int j = 0; j < dim; ++j) {
      vectors[i][j] = static_cast<float>(rand()) / RAND_MAX; // 随机数填充
    }
  }

  std::cout << "generate vectors done" << std::endl;

  // 批量插入向量
  for (int i = 0; i < num_vectors; ++i) {
    index->insert_vectors(vectors[i].data(), i); // 向量和对应的ID
  }

  std::cout << "insert vectors done" << std::endl;

  // 测试查询

  index->init_gpu();

  int num_query = 1000;
  std::vector<float> query(num_query * dim);
  for (int j = 0; j < num_query; ++j) {
    for (int k = 0; k < dim; ++k) {
      query[j * dim + k] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  index->search_vectors_batch_gpu(query, 5, 20);
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count()
            << "ns" << std::endl;

  std::cout << "------------------" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  index->search_vectors_batch_gpu(query, 5, 20, false);
  end = std::chrono::high_resolution_clock::now();

  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count()
            << "ns" << std::endl;
}

size_t num_vectors = 1000000;
size_t dimension = 128;

void test6() {
  auto index_ = new faiss::IndexHNSWFlat(dimension, 16);
  index_->hnsw.efConstruction = 200;
  index_->hnsw.efSearch = 20;
  faiss::Index *index = new faiss::IndexIDMap(index_);

  std::vector<float> vectors;
  std::vector<faiss::idx_t> ids;
  generate_random_vectors(num_vectors, dimension, vectors, ids);

  index->add_with_ids(num_vectors, vectors.data(), ids.data());

  int num_query = 1000;
  std::vector<float> query(num_query * dimension);
  for (int j = 0; j < num_query; ++j) {
    for (int k = 0; k < dimension; ++k) {
      query[j * dimension + k] = static_cast<float>(rand()) / RAND_MAX;
    }
  }
  std::vector<faiss::idx_t> I(5 * num_query);
  std::vector<float> D(5 * num_query);
  // 测试查询
  auto start = std::chrono::high_resolution_clock::now();
  index->search(num_query, query.data(), 5, D.data(), I.data());
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count()
            << "ns" << std::endl;

  delete index;
}

void test7() {
  // 使用GPU
  CUDAHNSWIndex *cuindex = new CUDAHNSWIndex(dimension, num_vectors, 16, 200);

  std::vector<float> vectors;
  std::vector<faiss::idx_t> ids;
  generate_random_vectors(num_vectors, dimension, vectors, ids);

  std::vector<size_t> size_t_ids(ids.begin(), ids.end());
  cuindex->insert_vectors_batch(vectors, size_t_ids);

  int num_query = 1000;
  std::vector<float> query(num_query * dimension);
  for (int j = 0; j < num_query; ++j) {
    for (int k = 0; k < dimension; ++k) {
      query[j * dimension + k] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  cuindex->init_gpu();
  auto start_gpu = std::chrono::high_resolution_clock::now();
  cuindex->search_vectors_batch_gpu(query, 5, 20, true);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::cout << "search time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu -
                                                                    start_gpu)
                   .count()
            << "ns" << std::endl;
}

std::vector<float> randVecs(size_t num, size_t dim) {
  std::vector<float> v(num * dim);
  auto seed = static_cast<unsigned>(std::time(0));

  faiss::float_rand(v.data(), v.size(), seed);

  return v;
}

void test8() {
  try {
    faiss::gpu::StandardGpuResources res;
    faiss::MetricType metric = faiss::METRIC_L2;

    faiss::gpu::GpuIndexCagraConfig config;
    config.device = 1;
    config.graph_degree = 32;
    config.intermediate_graph_degree = 64;
    config.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    faiss::gpu::GpuIndexCagra *gpu_index =
        new faiss::gpu::GpuIndexCagra(&res, dimension, metric, config);

    std::cout << "GpuIndexCagra initialized successfully with cuVS support!"
              << std::endl;

    int num_train = num_vectors;
    std::vector<float> train_vec = randVecs(num_train, dimension);
    const float *ptr = train_vec.data();
    gpu_index->train(num_train, ptr);

    int num_query = 1000;
    std::vector<float> query_vec = randVecs(num_query, dimension);
    int k = 5;
    std::vector<long> indices(num_query * k);
    std::vector<float> distances(num_query * k);

    auto start = std::chrono::high_resolution_clock::now();
    gpu_index->search(num_query, query_vec.data(), k, distances.data(),
                      indices.data());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "search time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      start)
                     .count()
              << "ns" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

int main() {
  std::cout << "Hello C++" << std::endl;

  test7();

  return 0;
}