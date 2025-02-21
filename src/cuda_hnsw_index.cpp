#include "cuda_hnsw_index.h"
#include <vector>

CUDAHNSWIndex::CUDAHNSWIndex(int dim, int num_data, int M, int ef_construction) : dim(dim) { // 将MetricType参数修改为第三个参数
    bool normalize = false;
    space = new hnswlib::L2Space(dim);
    index = new hnswlib::HierarchicalNSW<float>(space, num_data, M, ef_construction);
}

void CUDAHNSWIndex::insert_vectors(const std::vector<float>& data, uint64_t label) {
    index->addPoint(data.data(), label);
}

std::pair<std::vector<long>, std::vector<float>> CUDAHNSWIndex::search_vectors(const std::vector<float>& query, int k, int ef_search) { // 修改返回类型
    index->setEf(ef_search);
    auto result = index->searchKnn(query.data(), k);

    std::vector<long> indices(k);
    std::vector<float> distances(k);
    for (int j = 0; j < k; j++) {
        auto item = result.top();
        indices[j] = item.second;
        distances[j] = item.first;
        result.pop();
    }

    return {indices, distances};
}