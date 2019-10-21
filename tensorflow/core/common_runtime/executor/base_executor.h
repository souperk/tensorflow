//
// Created by kostas on 20/10/19.
//

#ifndef TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_BASE_EXECUTOR_H_
#define TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_BASE_EXECUTOR_H_

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor/graph_view.h"
#include "tensorflow/core/common_runtime/executor/base_executor_state.h"


namespace tensorflow {

class BaseExecutor : public Executor {
 protected:
  // forward declaration
  struct ControlFlowInfo;

 public:
  BaseExecutor(const LocalExecutorParams& params,
               std::unique_ptr<const Graph> graph);
  virtual ~BaseExecutor() = 0;
  virtual Status Initialize();


 protected:
//  virtual void InitializeRootNodes() = 0;
  virtual void InitializePending(const Graph* graph, const ControlFlowInfo& cf_info);

  static Status BuildControlFlowInfo(const Graph* graph,
                                     ControlFlowInfo* cf_info);

  FrameInfo* EnsureFrameInfo(const string& fname);

 protected:
  friend class BaseExecutorState;
  friend class FrameState;

  struct ControlFlowInfo {
    gtl::FlatSet<string> unique_frame_names;
    std::vector<string> frame_names;
  };


  std::unique_ptr<const Graph> graph_;
  GraphView gview_;


  // Root nodes that should form the initial ready queue
  //    1. Eager Executor should initialize this with
  //        no in edges
  //    2. OnDemand Executor should initialize this with
  //        no out edges
  std::vector<const Node*> root_nodes_;

  // Mapping from frame name to static information about the frame.
  // TODO(yuanbyu): We could cache it along with the graph so to avoid
  // the overhead of constructing it for each executor instance.
  gtl::FlatMap<string, FrameInfo*> frame_info_;

  // Owned
  LocalExecutorParams params_;

  // A cached value of params_
  bool device_record_tensor_accesses_ = false;

};

void GetMaxPendingCounts(const Node* n, size_t* max_pending,
                         size_t* max_dead_count);

}  // namespace tensorflow

#endif  // TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_BASE_EXECUTOR_H_
