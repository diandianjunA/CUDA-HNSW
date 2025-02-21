#include <stdio.h>
#include <iostream>
#include "cuda_hnsw_index.h"

extern "C"
void useCUDA();

int main() {
    std::cout<<"Hello C++"<<std::endl;
    useCUDA();

    CUDAHNSWIndex* index = new CUDAHNSWIndex(128, 1000000, 16, 200);

    int num_vectors = 100;
    int dim = 128;
    std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(dim));

    // 用随机数据填充向量
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < dim; ++j) {
            vectors[i][j] = static_cast<float>(rand()) / RAND_MAX;  // 随机数填充
        }
    }

    // 批量插入向量
    for (int i = 0; i < num_vectors; ++i) {
        index->insert_vectors(vectors[i], i);  // 向量和对应的ID
    }

    // 测试查询
    std::vector<float> query(dim, 0.5f);  // 查询向量
    std::pair<std::vector<long>, std::vector<float>> result = index->search_vectors(query, 5, 20);  // 找5个最近邻

    // 输出搜索结果
    for (int i = 0; i < result.first.size(); i++) {
        std::cout << "ID: " << result.first[i] << ", Distance: " << result.second[i] << std::endl;
    }

    return 0;
}