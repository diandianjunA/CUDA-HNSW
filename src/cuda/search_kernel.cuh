#pragma once

void cuda_search(int dims, int ef_search, int entry_node, int num_data_, const float *query_data, size_t data_size_, const char* data, size_t size_data_per_element_, size_t offsetData_, size_t k, int max_m0_, const std::vector<int>& graph_vec, const std::vector<int>& deg, int* nns, float* distances, int* found_cnt);