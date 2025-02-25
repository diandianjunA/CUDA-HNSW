#include "dist_calculate.cuh"

struct Node {
  float distance;
  int nodeid;
  bool checked;
};

__inline__ __device__
void PqPop(Node* pq, int* size) {
  if (threadIdx.x != 0) return;
  if (*size == 0) return;
  (*size)--;
  if (*size == 0) return;
  float tail_dist = pq[*size].distance;
  int p = 0, r = 1;
  while (r < *size) {
    if (r < (*size) - 1 and gt(pq[r + 1].distance, pq[r].distance))
      r++;
    if (ge(tail_dist, pq[r].distance))
      break;
    pq[p] = pq[r];
    p = r;
    r = 2 * p + 1;
  }
  pq[p] = pq[*size];
}

__inline__ __device__
void PqPush(Node* pq, int* size, float dist, int nodeid, bool check) {
  if (threadIdx.x != 0) return;
  int idx = *size;
  while (idx > 0) {
    int nidx = (idx + 1) / 2 - 1;
    if (ge(pq[nidx].distance, dist))
      break;
    pq[idx] = pq[nidx];
    idx = nidx;
  }
  pq[idx].distance = dist;
  pq[idx].nodeid = nodeid;
  pq[idx].checked = check;
  (*size)++;
}

__inline__ __device__
bool CheckAlreadyExists(const Node* pq, const int size, const int nodeid) {
  __syncthreads();
  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  
  static __shared__ bool shared[WARP_SIZE];
  bool exists = false;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (pq[i].nodeid == nodeid) {
      exists = true;
    }
  }
  
  #if __CUDACC_VER_MAJOR__ >= 9
  unsigned int active = __activemask();
  exists = __any_sync(active, exists);
  #else
  exists = __any(exists);
  #endif
  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = exists;
  }
  
  __syncthreads();
  
  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  } 


  // otherwise reduce again in the first warp
  exists = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : false;
  if (warp == 0) {
    #if __CUDACC_VER_MAJOR__ >= 9
    active = __activemask();
    exists = __any_sync(active, exists);
    #else
    exists = __any(exists);
    #endif
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
        shared[0] = exists;
    }
  }
  __syncthreads();
  return shared[0];



}

__inline__ __device__
void PushNodeToSearchPq(Node* pq, int* size, const int max_size,
    const float* data, const int dims,
    const float* src_vec, const int dstid) {
  if (CheckAlreadyExists(pq, *size, dstid)) return;
  const float* dst_vec = data + dims * dstid;
  float dist = GetDistanceByVec(src_vec, dst_vec, dims);
  __syncthreads();
  if (*size < max_size) {
    PqPush(pq, size, dist, dstid, false);
  } else if (gt(pq[0].distance, dist)) {
    PqPop(pq, size);
    PqPush(pq, size, dist, dstid, false);
  }
  __syncthreads();
}