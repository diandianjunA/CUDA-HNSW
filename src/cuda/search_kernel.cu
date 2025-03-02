#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#include "priority_queue.cuh"
#include "search_kernel.cuh"

#define CHECK(res)                                                             \
  {                                                                            \
    if (res != cudaSuccess) {                                                  \
      printf("Error ：%s:%d , ", __FILE__, __LINE__);                          \
      printf("code : %d , reason : %s \n", res, cudaGetErrorString(res));      \
      exit(-1);                                                                \
    }                                                                          \
  }

int max_level;

__inline__ __device__ unsigned int *get_linklist0(unsigned int internal_id) {
  return (unsigned int *)(data + internal_id * size_data_per_element +
                          offsetLevel0);
}

__inline__ __device__ unsigned short int getListCount(unsigned int *ptr) {
  return *((unsigned short int *)ptr);
}

__inline__ __device__ unsigned int *get_linklist(unsigned int internal_id,
                                                 int level) {
  return (unsigned int *)(linkLists[internal_id] +
                          (level - 1) * size_links_per_element);
}

__inline__ __device__ unsigned int *
get_linklist_at_level(unsigned int internal_id, int level) {
  return level == 0 ? get_linklist0(internal_id)
                    : get_linklist(internal_id, level);
}

__global__ void search_kernel(const float *query_data, int num_query, int k,
                              const int *entry_node, Node *device_pq,
                              bool *visited_table, int *global_candidate_nodes,
                              float *global_candidate_distances, int *found_cnt,
                              int *nns, float *distances) {

  static __shared__ int size;

  // int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  Node *ef_search_pq = device_pq + ef_search * blockIdx.x;
  int *candidate_nodes = global_candidate_nodes + ef_search * blockIdx.x;
  float *candidate_distances =
      global_candidate_distances + ef_search * blockIdx.x;

  bool *_visited_table = visited_table + num_data * blockIdx.x;

  for (int i = blockIdx.x; i < num_query; i += gridDim.x) {
    if (threadIdx.x == 0) {
      size = 0;
    }
    __syncthreads();

    const float *src_vec = query_data + i * dims;
    PushNodeToSearchPq(ef_search_pq, &size, query_data, entry_node[i]);

    if (CheckVisited(_visited_table, entry_node[i])) {
      continue;
    }
    __syncthreads();

    int idx = GetCand(ef_search_pq, size);
    while (idx >= 0) {
      __syncthreads();
      if (threadIdx.x == 0)
        ef_search_pq[idx].checked = true;
      int entry = ef_search_pq[idx].nodeid;
      __syncthreads();

      unsigned int *entry_neighbor_ptr = get_linklist0(entry);
      int deg = getListCount(entry_neighbor_ptr);

      for (int j = 1; j <= deg; ++j) {
        int dstid = *(entry_neighbor_ptr + j);

        if (CheckVisited(_visited_table, dstid)) {
          continue;
        }
        __syncthreads();

        PushNodeToSearchPq(ef_search_pq, &size, src_vec, dstid);
      }
      __syncthreads();
      idx = GetCand(ef_search_pq, size);
    }
    __syncthreads();

    for (int j = threadIdx.x; j < num_data; j += blockDim.x) {
      _visited_table[j] = false;
    }
    __syncthreads();
    // get sorted neighbors
    if (threadIdx.x == 0) {
      int size2 = size;
      while (size > 0) {
        candidate_nodes[size - 1] = ef_search_pq[0].nodeid;
        candidate_distances[size - 1] = ef_search_pq[0].distance;
        PqPop(ef_search_pq, &size);
      }
      found_cnt[i] = size2 < k ? size2 : k;
      for (int j = 0; j < found_cnt[i]; ++j) {
        nns[j + i * k] = candidate_nodes[j];
        distances[j + i * k] = out_scalar(candidate_distances[j]);
      }
    }
    __syncthreads();
  }
}

__global__ void kernel_check() {
  printf("Hello from kernel\n");

  for (int i = 0; i < num_data; i++) {
    float *data = getDataByInternalId(i);
    printf("data[%d] = [", i);
    for (int j = 0; j < dims; j++) {
      printf("%f, ", data[j]);
    }
    printf("]\n");
  }

  for (int i = 0; i < num_data; i++) {
    unsigned int *linklist = get_linklist0(i);
    int deg = getListCount(linklist);
    printf("linklist[%d] = [", i);
    for (int j = 1; j <= deg; j++) {
      printf("%d, ", *(linklist + j));
    }
    printf("]\n");
  }
}

__global__ void kernel_check2() {
  printf("Hello from kernel\n");

  for (int i = 0; i < num_data; i++) {
    for (int l = 0; l < element_levels[i]; l++) {
      unsigned int *linklist = get_linklist(i, l);
      printf("linklist: %u\n", *((unsigned short int *)linklist));
      int deg = getListCount(linklist);
      printf("linklist[%d][%d] = [", i, l);
      for (int j = 1; j <= deg; j++) {
        printf("%d, ", *(linklist + j));
      }
      printf("]\n");
    }
  }
}

__global__ void get_entry_points_kernel(const float *qdata, bool *visited_table,
                                        int *entries, int level,
                                        int num_query) {

  bool *_visited_table = visited_table + num_data * blockIdx.x;

  for (int i = blockIdx.x; i < num_query; i += gridDim.x) {
    __syncthreads();
    float entry_dist = 0;
    int entryid = entries[i];
    const float *src_vec = qdata + dims * i;
    {
      const float *dst_vec = getDataByInternalId(entryid);
      entry_dist = GetDistanceByVec(src_vec, dst_vec, dims);
    }
    __syncthreads();
    bool updated = true;
    while (updated) {
      // initialize entries as neighbors
      unsigned int *entry_neighbor_ptr = get_linklist_at_level(entryid, level);
      int deg = getListCount(entry_neighbor_ptr);

      updated = false;
      for (int j = 1; j <= deg; ++j) {
        int candid = *(entry_neighbor_ptr + j);

        if (_visited_table[candid]) {
          continue;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          _visited_table[candid] = true;
        }
        __syncthreads();
        const float *dst_vec = getDataByInternalId(candid);
        float dist = GetDistanceByVec(src_vec, dst_vec, dims);
        if (dist < entry_dist) {
          entry_dist = dist;
          entryid = candid;
          updated = true;
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        entries[i] = entryid;
      }
      __syncthreads();
    }
    __syncthreads();
    for (int j = threadIdx.x; j < num_data; j += blockDim.x) {
      _visited_table[j] = false;
    }
    __syncthreads();
  }
}

void get_entry_points(std::vector<int> &entries, const float *query_data,
                      int level, int block_cnt) {
  // process input data for kernel
  int num_data_ = 0;
  cudaMemcpyFromSymbol(&num_data_, num_data, sizeof(int));

  // copy to gpu mem
  thrust::device_vector<int> dev_entries(entries.size());
  thrust::device_vector<bool> dev_visited_table(num_data_ * block_cnt, false);
  // run kernel
  for (int l = max_level; l > level; l--) {
    thrust::copy(entries.begin(), entries.end(), dev_entries.begin());

    get_entry_points_kernel<<<block_cnt, 32>>>(
        query_data, thrust::raw_pointer_cast(dev_visited_table.data()),
        thrust::raw_pointer_cast(dev_entries.data()), l, entries.size());
    CHECK(cudaDeviceSynchronize());
    thrust::copy(dev_entries.begin(), dev_entries.begin() + entries.size(),
                 entries.begin());
    CHECK(cudaDeviceSynchronize());
  }
}

void cuda_search(int entry_node, const float *query_data, int num_query,
                 int ef_search_, int k, int *nns, float *distances,
                 int *found_cnt) {
  int block_cnt_ = 1024;
  cudaMemcpyToSymbol(ef_search, &ef_search_, sizeof(int));
  thrust::device_vector<Node> device_pq(ef_search_ * block_cnt_);
  thrust::device_vector<int> global_candidate_nodes(ef_search_ * block_cnt_);
  thrust::device_vector<float> global_candidate_distances(ef_search_ *
                                                          block_cnt_);
  int num_data_ = 0;
  cudaMemcpyFromSymbol(&num_data_, num_data, sizeof(int));
  thrust::device_vector<bool> device_visited_table(num_data_ * block_cnt_,
                                                   false);
  thrust::device_vector<int> device_found_cnt(num_query);
  thrust::device_vector<int> device_nns(k * num_query);
  thrust::device_vector<float> device_distances(k * num_query);

  int dims_;
  cudaMemcpyFromSymbol(&dims_, dims, sizeof(int));
  thrust::device_vector<float> device_qdata_(num_query * dims_);
  thrust::copy(query_data, query_data + num_query * dims_,
               device_qdata_.begin());
  thrust::device_vector<int> dev_entries(num_query, entry_node);

  search_kernel<<<block_cnt_, 512>>>(
      thrust::raw_pointer_cast(device_qdata_.data()), num_query, k,
      thrust::raw_pointer_cast(dev_entries.data()),
      thrust::raw_pointer_cast(device_pq.data()),
      thrust::raw_pointer_cast(device_visited_table.data()),
      thrust::raw_pointer_cast(global_candidate_nodes.data()),
      thrust::raw_pointer_cast(global_candidate_distances.data()),
      thrust::raw_pointer_cast(device_found_cnt.data()),
      thrust::raw_pointer_cast(device_nns.data()),
      thrust::raw_pointer_cast(device_distances.data()));
  CHECK(cudaDeviceSynchronize());
  thrust::copy(device_nns.begin(), device_nns.end(), nns);
  thrust::copy(device_distances.begin(), device_distances.end(), distances);
  thrust::copy(device_found_cnt.begin(), device_found_cnt.end(), found_cnt);
  CHECK(cudaDeviceSynchronize());
}

void cuda_search_hierarchical(int entry_node, const float *query_data,
                              int num_query, int ef_search_, int k, int *nns,
                              float *distances, int *found_cnt) {
  int block_cnt_ = 1024;
  cudaMemcpyToSymbol(ef_search, &ef_search_, sizeof(int));
  thrust::device_vector<Node> device_pq(ef_search_ * block_cnt_);
  thrust::device_vector<int> global_candidate_nodes(ef_search_ * block_cnt_);
  thrust::device_vector<float> global_candidate_distances(ef_search_ *
                                                          block_cnt_);
  int num_data_ = 0;
  cudaMemcpyFromSymbol(&num_data_, num_data, sizeof(int));
  thrust::device_vector<bool> device_visited_table(num_data_ * block_cnt_,
                                                   false);
  thrust::device_vector<int> device_found_cnt(num_query);
  thrust::device_vector<int> device_nns(k * num_query);
  thrust::device_vector<float> device_distances(k * num_query);

  int dims_;
  cudaMemcpyFromSymbol(&dims_, dims, sizeof(int));
  thrust::device_vector<float> device_qdata_(num_query * dims_);
  thrust::copy(query_data, query_data + num_query * dims_,
               device_qdata_.begin());
  std::vector<int> entries(num_query, entry_node);
  get_entry_points(entries, query_data, 0, block_cnt_);
  thrust::device_vector<int> dev_entries(num_query, entry_node);
  thrust::copy(entries.begin(), entries.end(), dev_entries.begin());

  search_kernel<<<block_cnt_, 32>>>(
      thrust::raw_pointer_cast(device_qdata_.data()), num_query, k,
      thrust::raw_pointer_cast(dev_entries.data()),
      thrust::raw_pointer_cast(device_pq.data()),
      thrust::raw_pointer_cast(device_visited_table.data()),
      thrust::raw_pointer_cast(global_candidate_nodes.data()),
      thrust::raw_pointer_cast(global_candidate_distances.data()),
      thrust::raw_pointer_cast(device_found_cnt.data()),
      thrust::raw_pointer_cast(device_nns.data()),
      thrust::raw_pointer_cast(device_distances.data()));
  CHECK(cudaDeviceSynchronize());
  thrust::copy(device_nns.begin(), device_nns.end(), nns);
  thrust::copy(device_distances.begin(), device_distances.end(), distances);
  thrust::copy(device_found_cnt.begin(), device_found_cnt.end(), found_cnt);
  CHECK(cudaDeviceSynchronize());
}

void cuda_init(int dims_, char *data_, size_t size_data_per_element_,
               size_t offsetData_, int max_m_, int ef_search_, int num_data_,
               size_t data_size_, size_t offsetLevel0_, char **linkLists_,
               int *element_levels_, size_t size_links_per_element_,
               int max_level_) {
  cudaMemcpyToSymbol(dims, &dims_, sizeof(int));
  cudaMemcpyToSymbol(size_data_per_element, &size_data_per_element_,
                     sizeof(size_t));
  cudaMemcpyToSymbol(offsetData, &offsetData_, sizeof(size_t));
  cudaMemcpyToSymbol(ef_search, &ef_search_, sizeof(int));
  cudaMemcpyToSymbol(num_data, &num_data_, sizeof(int));
  cudaMemcpyToSymbol(data_size, &data_size_, sizeof(size_t));
  cudaMemcpyToSymbol(offsetLevel0, &offsetLevel0_, sizeof(size_t));
  cudaMemcpyToSymbol(size_links_per_element, &size_links_per_element_,
                     sizeof(size_t));

  cudaMemcpyToSymbol(max_m, &max_m_, sizeof(int));
  max_level = max_level_;
  int *deviceData = nullptr;
  CHECK(cudaMalloc(&deviceData, num_data_ * size_data_per_element_));
  CHECK(cudaMemcpy(deviceData, data_, num_data_ * size_data_per_element_,
                   cudaMemcpyHostToDevice));
  cudaMemcpyToSymbol(data, &deviceData, sizeof(int *));
  CHECK(cudaDeviceSynchronize());

  char **deviceLinkLists = nullptr;
  CHECK(cudaMalloc(&deviceLinkLists, num_data_ * sizeof(void *)));
  for (int i = 0; i < num_data_; i++) {
    unsigned int *linklist = (unsigned int *)(linkLists_[i]);
    unsigned int *deviceLinkList = nullptr;
    if (element_levels_[i] != 0) {
      CHECK(cudaMalloc(&deviceLinkList,
                       element_levels_[i] * size_links_per_element_));
      CHECK(cudaMemcpy(deviceLinkList, linklist,
                       element_levels_[i] * size_links_per_element_,
                       cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(deviceLinkLists + i, &deviceLinkList, sizeof(void *),
                     cudaMemcpyHostToDevice));
  }
  cudaMemcpyToSymbol(linkLists, &deviceLinkLists, sizeof(char **));

  int *deviceElementLevels = nullptr;
  CHECK(cudaMalloc(&deviceElementLevels, num_data_ * sizeof(int)));
  CHECK(cudaMemcpy(deviceElementLevels, element_levels_,
                   num_data_ * sizeof(int), cudaMemcpyHostToDevice));
  cudaMemcpyToSymbol(element_levels, &deviceElementLevels, sizeof(int *));
  CHECK(cudaDeviceSynchronize());

  //   kernel_check<<<1, 1>>>();
  //   CHECK(cudaDeviceSynchronize());
  // kernel_check2<<<1, 1>>>();
  // CHECK(cudaDeviceSynchronize());
}
