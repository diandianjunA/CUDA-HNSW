#include <stdio.h>
#include <iostream>
#include "cuda_hnsw_index.h"
#include <chrono>

void test1() {
    CUDAHNSWIndex* index = new CUDAHNSWIndex(128, 100000, 16, 200);

    int num_vectors = 10000;
    int dim = 128;
    std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(dim));

    // 用随机数据填充向量
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < dim; ++j) {
            vectors[i][j] = static_cast<float>(rand()) / RAND_MAX;  // 随机数填充
        }
    }

    std::cout << "generate vectors done" << std::endl;

    // 批量插入向量
    for (int i = 0; i < num_vectors; ++i) {
        index->insert_vectors(vectors[i], i);  // 向量和对应的ID
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
        std::pair<std::vector<long>, std::vector<float>> result = index->search_vectors(query, 5, 20);  // 找5个最近邻
    }
    auto end = std::chrono::high_resolution_clock::now();
    // 输出搜索时间，单位为纳秒
    std::cout << "search time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns" << std::endl;

    std::cout << "------------------" << std::endl;

    index->init_gpu();  // 初始化GPU

    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        std::vector<float> query(dim);
        for (int j = 0; j < dim; ++j) {
            query[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        std::pair<std::vector<long>, std::vector<float>> result = index->search_vectors_gpu(query, 5, 20);  // 找5个最近邻
    }
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "search time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - start_gpu).count() << "ns" << std::endl;
}

void test2() {
    
}

int main() {
    std::cout<<"Hello C++"<<std::endl;

    

    return 0;
}