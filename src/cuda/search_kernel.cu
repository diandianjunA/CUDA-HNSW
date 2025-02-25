#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "search_kernel.cuh"
#include "priority_queue.cuh"

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}

#define visited_table_size_ 100
#define visited_list_size_ 50

__inline__ __device__
bool CheckVisited(int* visited_table, int* visited_list, int& visited_cnt, int target, const int visited_table_size, const int visited_list_size) {
  __syncthreads();
  bool ret = false;
  if (visited_cnt < visited_list_size ){
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

__inline__ __device__
int warp_reduce_cand(const Node* pq, int cand) {
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
        if (update) cand = _cand;
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
        if (update) cand = _cand;
      }
    }
  }
  #endif
  return cand;
}

__inline__ __device__
int GetCand(const Node* pq, const int size) {
  __syncthreads();
  
  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  static __shared__ int shared[WARP_SIZE];
  // pick the closest neighbor with checked = false if reverse = false and vice versa 
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

__global__ void search_kernel(
    int dims, const float *query_data, const float* data, int entry_node,
    Node* device_pq, int* visited_table, int* visited_list, 
    int ef_search, int* global_candidate_nodes, float* global_candidate_distances, 
    int * found_cnt, int max_m, int topk, int* nns, float* distances, const int* graph, const int* deg) {

    static __shared__ int size;

    Node* ef_search_pq = device_pq + ef_search * blockIdx.x;
    int* candidate_nodes = global_candidate_nodes + ef_search * blockIdx.x;
    float* candidate_distances = global_candidate_distances + ef_search * blockIdx.x;

    static __shared__ int visited_cnt;
    int* _visited_table = visited_table + visited_table_size_ * blockIdx.x;
    int* _visited_list = visited_list + visited_list_size_ * blockIdx.x;

    const float* src_vec = query_data;
    PushNodeToSearchPq(ef_search_pq, &size, ef_search, data, dims, query_data, entry_node);

    if (CheckVisited(_visited_table, _visited_list, visited_cnt, entry_node, visited_table_size_, visited_list_size_)) {
        return;
    }

    int idx = GetCand(ef_search_pq, size);
    while (idx >= 0) {
      __syncthreads();
      if (threadIdx.x == 0) ef_search_pq[idx].checked = true;
      int entry = ef_search_pq[idx].nodeid;
      __syncthreads();

      for (int j = max_m * entry; j < max_m * entry + deg[entry]; ++j) {
        int dstid = graph[j];

        if (CheckVisited(_visited_table, _visited_list, visited_cnt, dstid, 
              visited_table_size_, visited_list_size_)) 
          continue;
        __syncthreads();

        const float* dst_vec = data + dims * dstid;
        float dist = GetDistanceByVec(src_vec, dst_vec, dims);

        PushNodeToSearchPq(ef_search_pq, &size, ef_search,
            data, dims, src_vec, dstid);
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
      *found_cnt = size2 < topk? size2: topk;
      for (int j = 0; j < *found_cnt; ++j) {
        nns[j] = candidate_nodes[j];
        distances[j] = out_scalar(candidate_distances[j]);
      }
    }
}


void cuda_search(int dims, int ef_search, int entry_node, int num_data_, const float *query_data, const float* data, size_t k, int max_m0_, const std::vector<int>& graph_vec, const std::vector<int>& deg, int* nns, float* distances, int* found_cnt) {
    int block_cnt_ = 16;
    thrust::device_vector<Node> device_pq(ef_search * block_cnt_);
    thrust::device_vector<int> global_candidate_nodes(ef_search * block_cnt_);
    thrust::device_vector<float> global_candidate_distances(ef_search * block_cnt_);
    thrust::device_vector<int> device_visited_table(visited_table_size_ * block_cnt_, -1);
    thrust::device_vector<int> device_visited_list(visited_list_size_ * block_cnt_);
    thrust::device_vector<int> device_graph(max_m0_ * num_data_);
    thrust::device_vector<int> device_deg(num_data_);
    thrust::device_vector<int> device_found_cnt(1);
    thrust::device_vector<int> device_nns(k);
    thrust::device_vector<float> device_distances(k);
    thrust::device_vector<float> device_data(num_data_);

    thrust::copy(graph_vec.begin(), graph_vec.end(), device_graph.begin());
    thrust::copy(deg.begin(), deg.end(), device_deg.begin());

    search_kernel<<<1, 5>>>(
      dims, query_data, data, entry_node,
      thrust::raw_pointer_cast(device_pq.data()),
      thrust::raw_pointer_cast(device_visited_table.data()),
      thrust::raw_pointer_cast(device_visited_list.data()),
      ef_search,
      thrust::raw_pointer_cast(global_candidate_nodes.data()),
      thrust::raw_pointer_cast(global_candidate_distances.data()),
      thrust::raw_pointer_cast(device_found_cnt.data()),
      max_m0_, k, 
      thrust::raw_pointer_cast(device_nns.data()),
      thrust::raw_pointer_cast(device_distances.data()),
      thrust::raw_pointer_cast(device_graph.data()),
      thrust::raw_pointer_cast(device_deg.data())
    );
    CHECK(cudaDeviceSynchronize());
    thrust::copy(device_nns.begin(), device_nns.end(), nns);
    thrust::copy(device_distances.begin(), device_distances.end(), distances);
    thrust::copy(device_found_cnt.begin(), device_found_cnt.end(), found_cnt);
    CHECK(cudaDeviceSynchronize());
}