#pragma once

void cuda_init(int dims_, char *data_, size_t size_data_per_element_,
               size_t offsetData_, int max_m_, int ef_search_, int num_data_,
               size_t data_size_, size_t offsetLevel0_, char **linkLists_,
               int *element_levels_, size_t size_links_per_element_,
               int max_level_);

void cuda_search(int entry_node, const float *query_data, int num_query,
                 int ef_search_, int k, int *nns, float *distances,
                 int *found_cnt);

void cuda_search_hierarchical(int entry_node, const float *query_data,
                              int num_query, int ef_search_, int k, int *nns,
                              float *distances, int *found_cnt);