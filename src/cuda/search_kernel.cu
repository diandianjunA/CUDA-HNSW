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

#define visited_table_size_ 100
#define visited_list_size_ 50

__inline__ __device__ unsigned int *get_linklist0(unsigned int internal_id) {
  return (unsigned int *)(data + internal_id * size_data_per_element +
                          offsetLevel0);
}

__inline__ __device__ unsigned short int getListCount(unsigned int *ptr) {
  return *((unsigned short int *)ptr);
}

__inline__ __device__ bool CheckVisited(int *visited_table, int *visited_list,
                                        int &visited_cnt, int target,
                                        const int visited_table_size,
                                        const int visited_list_size) {
  __syncthreads();
  bool ret = false;
  if (visited_cnt < visited_list_size) {
    int idx = target % visited_table_size;
    if (visited_table[idx] != target) {
      __syncthreads();
      if (threadIdx.x == 0) {
        if (visited_table[idx] == -1) {
          visited_table[idx] = target;
          visited_list[visited_cnt++] = idx;
        }
      }
    } else {
      ret = true;
    }
  }
  __syncthreads();
  return ret;
}

__inline__ __device__ int warp_reduce_cand(const Node *pq, int cand) {
#if __CUDACC_VER_MAJOR__ >= 9
  unsigned int active = __activemask();
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    int _cand = __shfl_down_sync(active, cand, offset);
    if (_cand >= 0) {
      if (cand == -1) {
        cand = _cand;
      } else {
        bool update = gt(pq[cand].distance, pq[_cand].distance);
        if (update)
          cand = _cand;
      }
    }
  }
#else
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    int _cand = __shfl_down(cand, offset);
    if (_cand >= 0) {
      if (cand == -1) {
        cand = _cand;
      } else {
        bool update = gt(pq[cand].distance, pq[_cand.distance]);
        if (update)
          cand = _cand;
      }
    }
  }
#endif
  return cand;
}

__inline__ __device__ int GetCand(const Node *pq, const int size) {
  __syncthreads();

  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  static __shared__ int shared[WARP_SIZE];
  float dist = INFINITY;
  int cand = -1;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (not pq[i].checked) {
      bool update = gt(dist, pq[i].distance);
      if (update) {
        cand = i;
        dist = pq[i].distance;
      }
    }
  }
  cand = warp_reduce_cand(pq, cand);

  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = cand;
  }
  __syncthreads();

  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  }

  // otherwise reduce again in the first warp
  cand = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -1;
  if (warp == 0) {
    cand = warp_reduce_cand(pq, cand);
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
      shared[0] = cand;
    }
  }
  __syncthreads();
  return shared[0];
}

__global__ void search_kernel(const float *query_data, int num_query, int k, int entry_node,
                              Node *device_pq, int *visited_table,
                              int *visited_list, int *global_candidate_nodes,
                              float *global_candidate_distances, int *found_cnt,
                              int *nns, float *distances) {

  static __shared__ int size;

  // int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  Node *ef_search_pq = device_pq + ef_search * blockIdx.x;
  int *candidate_nodes = global_candidate_nodes + ef_search * blockIdx.x;
  float *candidate_distances =
      global_candidate_distances + ef_search * blockIdx.x;

  static __shared__ int visited_cnt;
  int *_visited_table = visited_table + visited_table_size_ * blockIdx.x;
  int *_visited_list = visited_list + visited_list_size_ * blockIdx.x;

  for (int i = blockIdx.x; i < num_qnodes; i += gridDim.x) {
    if (threadIdx.x == 0) {
      size = 0;
      visited_cnt = 0;
    }
    __syncthreads();
  
    const float *src_vec = query_data + i * dims;
    PushNodeToSearchPq(ef_search_pq, &size, query_data, entry_node);
  
    if (CheckVisited(_visited_table, _visited_list, visited_cnt, entry_node,
                     visited_table_size_, visited_list_size_)) {
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
  
        if (CheckVisited(_visited_table, _visited_list, visited_cnt, dstid,
                         visited_table_size_, visited_list_size_)) {
          continue;
        }
        __syncthreads();
  
        PushNodeToSearchPq(ef_search_pq, &size, src_vec, dstid);
      }
      __syncthreads();
      idx = GetCand(ef_search_pq, size);
    }
    __syncthreads();
  
    for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
      _visited_table[_visited_list[j]] = -1;
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
      *found_cnt = size2 < k ? size2 : k;
      for (int j = 0; j < found_cnt[i]; ++j) {
        nns[j + i * topk] = cand_nodes[j];
        distances[j + i * topk] = out_scalar(cand_distances[j]);
      }
    }
    __syncthreads();
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
}

void cuda_search(int entry_node, const float *query_data, int num_query, int ef_search_,
                       int k, int *nns, float *distances, int *found_cnt) {
  int block_cnt_ = num_query;
  cudaMemcpyToSymbol(ef_search, &ef_search_, sizeof(int));
  thrust::device_vector<Node> device_pq(ef_search_ * block_cnt_);
  thrust::device_vector<int> global_candidate_nodes(ef_search_ * block_cnt_);
  thrust::device_vector<float> global_candidate_distances(ef_search_ *
                                                          block_cnt_);
  thrust::device_vector<int> device_visited_table(
      visited_table_size_ * block_cnt_, -1);
  thrust::device_vector<int> device_visited_list(visited_list_size_ *
                                                 block_cnt_);
  thrust::device_vector<int> device_found_cnt(block_cnt_);
  thrust::device_vector<int> device_nns(k * block_cnt_);
  thrust::device_vector<float> device_distances(k * block_cnt_);

  search_kernel<<<block_cnt_, 32>>>(
      query_data, k, entry_node, thrust::raw_pointer_cast(device_pq.data()),
      thrust::raw_pointer_cast(device_visited_table.data()),
      thrust::raw_pointer_cast(device_visited_list.data()),
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
               size_t data_size_, size_t offsetLevel0_) {
  cudaMemcpyToSymbol(dims, &dims_, sizeof(int));
  cudaMemcpyToSymbol(size_data_per_element, &size_data_per_element_,
                     sizeof(size_t));
  cudaMemcpyToSymbol(offsetData, &offsetData_, sizeof(size_t));
  cudaMemcpyToSymbol(max_m, &max_m_, sizeof(size_t));
  cudaMemcpyToSymbol(ef_search, &ef_search_, sizeof(int));
  cudaMemcpyToSymbol(num_data, &num_data_, sizeof(int));
  cudaMemcpyToSymbol(data_size, &data_size_, sizeof(size_t));
  cudaMemcpyToSymbol(offsetLevel0, &offsetLevel0_, sizeof(size_t));

  int *deviceData = nullptr;
  CHECK(cudaMalloc(&deviceData, num_data_ * size_data_per_element_));
  CHECK(cudaMemcpy(deviceData, data_, num_data_ * size_data_per_element_,
                   cudaMemcpyHostToDevice));
  cudaMemcpyToSymbol(data, &deviceData, sizeof(int *));
  CHECK(cudaDeviceSynchronize());

  //   kernel_check<<<1, 1>>>();
  //   CHECK(cudaDeviceSynchronize());
}