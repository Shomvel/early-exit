// SPDX-License-Identifier: Apache-2.0
// Turn marked, captured PyTorch layer regions into CUDA IF body graphs.
#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static thread_local std::string last_error;
static void check_cuda(cudaError_t e, const char *operation) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(e));
}
#define check(call) check_cuda((call), #call)
static void driver_check(CUresult e) {
  if (e != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(e, &message);
    throw std::runtime_error(std::string("CUDA driver graph clone: ") +
                             message);
  }
}
__global__ void region_marker(int code) { asm volatile(""); }
__global__ void pdl_increment(int *counter) {
  cudaGridDependencySynchronize();
  ++counter[0];
  cudaTriggerProgrammaticLaunchCompletion();
}
extern "C" int ee_pdl_increment(uint64_t stream, uint64_t counter) {
  try {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(1);
    config.stream = reinterpret_cast<cudaStream_t>(stream);
    config.attrs = &attribute;
    config.numAttrs = 1;
    check(cudaLaunchKernelEx(&config, pdl_increment,
                             reinterpret_cast<int *>(counter)));
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
__global__ void set_condition(cudaGraphConditionalHandle h, const int *action,
                              int width) {
  cudaGraphSetConditional(h,
                          width > 0 ? action[0] < width : action[0] >= -width);
}
extern "C" const char *ee_error() { return last_error.c_str(); }
extern "C" int ee_dump(uint64_t graph, const char *path) {
  try {
    check(cudaGraphDebugDotPrint(reinterpret_cast<cudaGraph_t>(graph), path,
                                 cudaGraphDebugDotFlagsVerbose));
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
extern "C" int ee_marker(uint64_t stream, int code) {
  try {
    region_marker<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(code);
    check(cudaGetLastError());
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
using Node = cudaGraphNode_t;
static std::set<Node>
reachable(Node seed, const std::unordered_map<Node, std::vector<Node>> &links) {
  std::set<Node> result;
  std::vector<Node> pending{seed};
  while (!pending.empty()) {
    Node n = pending.back();
    pending.pop_back();
    if (!result.insert(n).second)
      continue;
    auto it = links.find(n);
    if (it != links.end())
      for (auto k : it->second)
        pending.push_back(k);
  }
  return result;
}
struct OwnedGraph {
  cudaGraph_t graph = nullptr;
  ~OwnedGraph() {
    if (graph)
      cudaGraphDestroy(graph);
  }
};

extern "C" int ee_convert(uint64_t graph_ptr, uint64_t action_ptr,
                          const int *widths, int groups, int *moved) {
  try {
    auto graph = reinterpret_cast<cudaGraph_t>(graph_ptr);
    const int *action = reinterpret_cast<const int *>(action_ptr);
    size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<Node> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count));
    size_t edges = 0;
    check(cudaGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &edges));
    std::vector<Node> from(edges), to(edges);
    std::vector<cudaGraphEdgeData> edge_data(edges);
    check(cudaGraphGetEdges_v2(graph, from.data(), to.data(), edge_data.data(),
                               &edges));
    std::unordered_map<Node, std::vector<Node>> children, parents;
    for (size_t i = 0; i < edges; ++i) {
      children[from[i]].push_back(to[i]);
      parents[to[i]].push_back(from[i]);
    }
    std::unordered_map<int, Node> marks;
    for (Node n : nodes) {
      cudaGraphNodeType type;
      check(cudaGraphNodeGetType(n, &type));
      if (type != cudaGraphNodeTypeKernel)
        continue;
      CUDA_KERNEL_NODE_PARAMS p{};
      driver_check(
          cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(n), &p));
      const char *name = nullptr;
      driver_check(cuFuncGetName(&name, p.func));
      if (std::string(name).find("region_marker") != std::string::npos) {
        int id = *static_cast<int *>(p.kernelParams[0]);
        if (!marks.emplace(id, n).second)
          throw std::runtime_error("duplicate layer marker");
      }
    }
    if (marks.size() != size_t(groups * 2))
      throw std::runtime_error("wrong conditional marker count: " +
                               std::to_string(marks.size()));
    std::vector<std::set<Node>> regions;
    std::vector<std::unique_ptr<OwnedGraph>> copies;
    for (int group = 0; group < groups; ++group) {
      Node begin = marks.at(group * 2), end = marks.at(group * 2 + 1);
      auto desc = reachable(begin, children), anc = reachable(end, parents);
      std::set<Node> region;
      std::set_intersection(desc.begin(), desc.end(), anc.begin(), anc.end(),
                            std::inserter(region, region.end()));
      region.erase(begin);
      region.erase(end);
      if (region.empty())
        throw std::runtime_error("empty conditional layer region");
      for (Node n : region) {
        for (Node p : parents[n])
          if (!region.count(p) && p != begin)
            throw std::runtime_error(
                "conditional region has external input edge");
        for (Node c : children[n])
          if (!region.count(c) && c != end)
            throw std::runtime_error(
                "conditional region has external output edge");
      }
      // Clone the captured graph before adding any conditional nodes. CUDA
      // clones retain the complete kernel launch representation and all edge
      // metadata. The GetParams + CopyAttributes reconstruction failed for
      // captured BF16 cuBLAS kernels; keep their captured nodes intact.
      auto copy = std::make_unique<OwnedGraph>();
      check(cudaGraphClone(&copy->graph, graph));
      for (Node n : nodes) {
        if (!region.count(n)) {
          Node cloned;
          check(cudaGraphNodeFindInClone(&cloned, n, copy->graph));
          check(cudaGraphDestroyNode(cloned));
        }
      }
      regions.push_back(std::move(region));
      copies.push_back(std::move(copy));
    }
    for (int group = 0; group < groups; ++group) {
      Node begin = marks.at(group * 2), end = marks.at(group * 2 + 1);
      const auto &region = regions[group];
      cudaGraphConditionalHandle handle;
      check(cudaGraphConditionalHandleCreate(&handle, graph, 0,
                                             cudaGraphCondAssignDefault));
      cudaKernelNodeParams kp{};
      kp.func = reinterpret_cast<void *>(set_condition);
      kp.gridDim = dim3(1);
      kp.blockDim = dim3(1);
      int width = widths[group];
      void *args[] = {&handle, &action, &width};
      kp.kernelParams = args;
      Node setter;
      check(cudaGraphAddKernelNode(&setter, graph, &begin, 1, &kp));
      cudaGraphNodeParams cp{};
      cp.type = cudaGraphNodeTypeConditional;
      cp.conditional.handle = handle;
      cp.conditional.type = cudaGraphCondTypeIf;
      cp.conditional.size = 1;
      Node conditional;
      check(cudaGraphAddNode(&conditional, graph, &setter, 1, &cp));
      cudaGraph_t body = cp.conditional.phGraph_out[0];
      Node child;
      check(cudaGraphAddChildGraphNode(&child, body, nullptr, 0,
                                       copies[group]->graph));
      for (Node n : region)
        check(cudaGraphDestroyNode(n));
      // The closing marker rejoins the parent only after the selected body.
      check(cudaGraphAddDependencies(graph, &conditional, &end, 1));
      moved[group] = int(region.size());
    }
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
