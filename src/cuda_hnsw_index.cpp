#include "cuda_hnsw_index.h"
#include <vector>
#include "cuda/search_kernel.cuh"

CUDAHNSWIndex::CUDAHNSWIndex(int dim, int num_data, int M, int ef_construction) : dim(dim) { // 将MetricType参数修改为第三个参数
    bool normalize = false;
    space = new hnswlib::L2Space(dim);
    index = new hnswlib::HierarchicalNSW<float>(space, num_data, M, ef_construction);
}

void CUDAHNSWIndex::insert_vectors(const float* data, uint64_t label) {
    index->addPoint(data, label);
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

void CUDAHNSWIndex::check() {
    for (int i = 0; i < index->cur_element_count; i++) {
        char* data_ptr = index->getDataByInternalId(i);
        float* data = (float*) data_ptr;
        std::cout << "data[" << i << "] = [";
        for (int j = 0; j < dim; j++) {
            std::cout << data[j] << ", ";
        }
        std::cout << "]" << std::endl;
    }

    for (int i = 0; i < index->cur_element_count; i++) {
        unsigned int *linklist = index->get_linklist0(i);
        int deg = index->getListCount(linklist);
        printf("linklist[%d] = [", i);
        for (int j = 1; j <= deg; j++) {
          printf("%d, ", *(linklist + j));
        }
        printf("]\n");
    }
}

void CUDAHNSWIndex::init_gpu() {
    // for (int i = 0; i < index->cur_element_count; i++) {
    //     std::cout << "element_levels_[" << i << "] = " << index->element_levels_[i] << std::endl;
    // }
    // for (int i = 0; i < index->cur_element_count; i++) {
    //     for (int l = 0; l < index->element_levels_[i]; l++) {
    //       unsigned int *linklist = index->get_linklist_at_level(i, l);
    //       int deg = index->getListCount(linklist);
    //       printf("linklist[%d][%d] = [", i, l);
    //       for (int j = 1; j <= deg; j++) {
    //         printf("%d, ", *(linklist + j));
    //       }
    //       printf("]\n");
    //     }
    // }
    cuda_init(dim, index->data_level0_memory_, index->size_data_per_element_, index->offsetData_, index->maxM0_, index->ef_, index->cur_element_count, index->data_size_, index->offsetLevel0_, index->linkLists_, index->element_levels_.data(), index->size_links_per_element_, index->maxlevel_);
}

std::pair<std::vector<long>, std::vector<float>> CUDAHNSWIndex::search_vectors_gpu(const std::vector<float>& query, int k, int ef_search, bool use_hierarchy) {
    std::vector<int> inner_index(k);
    std::vector<long> indices(k);
    std::vector<float> distances(k);
    int fount_cnt = 0;
    if (use_hierarchy) {
        cuda_search_hierarchical(index->enterpoint_node_, query.data(), 1, ef_search, k, inner_index.data(), distances.data(), &fount_cnt);
    } else {
        cuda_search(index->enterpoint_node_, query.data(), 1, ef_search, k, inner_index.data(), distances.data(), &fount_cnt);
    }
    for (int i = 0; i < fount_cnt; i++) {
        indices[i] = index->getExternalLabel(inner_index[i]);
    }
    return {indices, distances};
}

// GPU批量查询
std::vector<std::pair<std::vector<long>, std::vector<float>>> CUDAHNSWIndex::search_vectors_batch_gpu(const std::vector<float>& query, int k, int ef_search, bool use_hierarchy) {
    std::vector<std::pair<std::vector<long>, std::vector<float>>> results;
    int num_query = query.size() / dim;
    std::vector<int> inner_index(k * num_query);
    std::vector<float> distances(k * num_query);
    std::vector<int> found_cnt(num_query);

    if (use_hierarchy) {
        cuda_search_hierarchical(index->enterpoint_node_, query.data(), num_query, ef_search, k, inner_index.data(), distances.data(), found_cnt.data());
    } else {
        cuda_search(index->enterpoint_node_, query.data(), num_query, ef_search, k, inner_index.data(), distances.data(), found_cnt.data());
    }
    
    for (int i = 0; i < num_query; i++) {
        std::vector<long> indices(k);
        std::vector<float> dists(k);
        for (int j = 0; j < found_cnt[i]; j++) {
            indices[j] = index->getExternalLabel(inner_index[i * k + j]);
            dists[j] = distances[i * k + j];
        }
        results.push_back({indices, dists});
    }

    return results;
}