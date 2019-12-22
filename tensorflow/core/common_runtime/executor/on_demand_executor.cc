/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/executor.h"

#include <atomic>
#include <deque>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/executor/graph_view.h"
#include "tensorflow/core/common_runtime/executor/node_state.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/traceme_recorder.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/common_runtime/executor/warehouse.h"
#include "tensorflow/core/common_runtime/executor/memory_utils.h"

#define TRACE() do { LOG(INFO) << "TRACEME : " << __func__ << "@" << __LINE__; } while(false)
#define LOGVAR(VAR) do { LOG(INFO) << #VAR << " : " << (VAR) << " " << __func__ << "@" << __LINE__; } while(false)

#define NODE_SLOT(node, slot) "[" << node->id() << "::" << slot << "]"
#define NODE_NAME(node) "[" << node->name() << "::" << node->id() << "]"

namespace tensorflow {
namespace {

using NodeState= executor::NodeState;
using NodeItem = executor::NodeItem;
using EdgeInfo = executor::EdgeInfo;
using GraphView = executor::GraphView;
using Warehouse = executor::Warehouse;
using WarehouseStrategy = executor::WarehouseStrategy;
using Entry = executor::Entry;
using EntryVector = executor::EntryVector;

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext *, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

Status PrepareInputs(
    const NodeItem &item,
    EntryVector &entries,
    TensorValueVec *inputs,
    DeviceContextVec *input_device_contexts,
    AllocatorAttributeVec *input_alloc_attrs);

Status ProcessOutputs(
    const NodeItem &item,
    OpKernelContext *ctx,
    EntryVector &outputs_vector,
    DeviceContextMap &device_context_map);


// 1-D, 0 element tensor.
static const Tensor *const kEmptyTensor = new Tensor;

bool IsInitializationOp(const Node *node) {
  return node->op_def().allows_uninitialized_input();
}

// Time the execution of kernels (in CPU cycles).  Used to dynamically identify
// inexpensive kernels which can be dispatched inline.
struct KernelTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }
};

/**
 *
 */
struct IterationState {
  explicit IterationState()
      : outstanding_ops_(0) {}

  /* property accessors */
 public:
  size_t outstanding_ops() const { return outstanding_ops_; }

  /* events */
 public:
  void OnOperationStarted() {
    mutex_lock lock(mu_);
    outstanding_ops_++;
  }

  void OnOperationEnded() {
    mutex_lock lock(mu_);
    outstanding_ops_--;
  }

  const NodeState &node_state(int64 node_id) { return node_state_map_[node_id]; };

 private:

  mutex mu_;
  std::map<int64, NodeState> node_state_map_ GUARDED_BY(mu_);

  // The number of outstanding ops for each iteration.
  size_t outstanding_ops_;

};

class OnDemandExecutor;

class ExecutionContext {

 public:
  ExecutionContext(LocalExecutorParams &params,
                   const Graph *graph,
                   const GraphView *graph_view,
                   std::vector<const Node *> &root_nodes)
      : params_(params), graph_(graph), graph_view_(graph_view), root_nodes_(root_nodes) {

  }

 public:

  Status CreateKernel(const NodeDef &node_def, OpKernel **kernel_ptr) {
    return params_.create_kernel(node_def, kernel_ptr);
  }

  void DeleteKernel(OpKernel *kernel) {
    params_.delete_kernel(kernel);
  }

 public:
  LocalExecutorParams &params() { return params_; }
  Device *device() { return params_.device; }
  const Graph *graph() const { return graph_; }
  const GraphView *graph_view() const { return graph_view_; }
  const NodeItem *node(size_t node_id) const { return graph_view_->node(node_id); }
  bool device_record_tensor_accesses() const { return device_record_tensor_accesses_; }
  const std::vector<const Node *> &root_nodes() { return root_nodes_; }

 private:
  friend OnDemandExecutor;

  // Owned.
  LocalExecutorParams &params_;
  const Graph *graph_;
  const GraphView *graph_view_;

  // Root nodes (with no in edges) that should form the initial ready queue
  const std::vector<const Node *> &root_nodes_;

  // A cached value of params_
  bool device_record_tensor_accesses_ = false;
};

class OnDemandExecutor : public Executor {
 public:
  OnDemandExecutor(const LocalExecutorParams &p, std::unique_ptr<const Graph> g)
      : params_(p), graph_(std::move(g)), graph_view_() {
    CHECK(p.create_kernel != nullptr);
    CHECK(p.delete_kernel != nullptr);

  }

  ~OnDemandExecutor() override {
    for (int i = 0; i < graph_->num_node_ids(); i++) {
      NodeItem *item = graph_view_.node(i);
      if (item != nullptr) {
        params_.delete_kernel(item->kernel);
      }
    }
  }

  Status Initialize();
  void RunAsync(const Args &args, DoneCallback done) override;

 private:

  LocalExecutorParams params_;
  std::unique_ptr<const Graph> graph_;
  GraphView graph_view_;

  // Root nodes (with no in edges) that should form the initial ready queue
  std::vector<const Node *> root_nodes_;

  // A cached value of params_
  bool device_record_tensor_accesses_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(OnDemandExecutor);
};

Status OnDemandExecutor::Initialize() {
  graph_view_.Initialize(graph_.get());

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  for (const Node *n : graph_->nodes()) {
    const int id = n->id();

//    LOG(INFO) << "initializing node [" << n->name() << "::" << n->id() << "] with"
//              << " " << n->num_inputs() << " inputs"
//              << ", " << n->in_edges().size() << " input edges"
//              << ", " << n->num_inputs() << " outputs"
//              << ", " << n->out_edges().size() << " output edges";

//    for (const Edge *edge: n->out_edges()) {
//      LOG(INFO) << "output edge to [" << edge->dst()->name() << "::" << edge->dst()->id() << "] : "
//                << (edge->IsControlEdge() ? "is control edge" : "is not control edge");
//    }

    // TODO(souperk) decide if out_edges().empty() is preferable to IsSink() ||
    //  IsSend()
    if (n->out_edges().empty()) {
//      LOG(INFO) << "adding node " << n->name() << " to root_nodes_";
      root_nodes_.push_back(n);
    }

    NodeItem *item = graph_view_.node(id);
    item->node = n;

    Status s = params_.create_kernel(n->def(), &item->kernel);
    if (!s.ok()) {
      item->kernel = nullptr;
      s = AttachDef(s, *n);
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      return s;
    }

    CHECK(item->kernel);
    item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
    item->is_merge = IsMerge(n);
    item->is_enter = IsEnter(n);

    if (item->is_enter) {
      bool is_constant_enter;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), "is_constant", &is_constant_enter));
      item->is_constant_enter = is_constant_enter;
    } else {
      item->is_constant_enter = false;
    }

    item->is_exit = IsExit(n);
    item->is_control_trigger = IsControlTrigger(n);
    item->is_sink = IsSink(n);
    item->is_enter_exit_or_next_iter =
        (IsEnter(n) || IsExit(n) || IsNextIteration(n));
  }

  return graph_view_.SetAllocAttrs(graph_.get(), params_.device);
}

struct TaggedNode {
  const Node *node = nullptr;
  bool is_dead = false;

  TaggedNode(const Node *node_ptr, bool dead) {
    node = node_ptr;
    is_dead = dead;
  }

  int64 id() const { return node->id(); }
};

typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;

class ExecutionStep {

 public:
  ExecutionStep(TaggedNode tagged_node, const NodeItem &node_item)
      : tagged_node_(tagged_node),
        inputs_(node_item.num_inputs),
        outputs_(node_item.num_outputs) {};

 public:
  bool HasParent() const { return parent_ != nullptr; }
  ExecutionStep *parent() { return parent_; }

  const Node *node() const { return tagged_node_.node; }

  EntryVector &inputs() { return inputs_; }
  EntryVector &outputs() { return outputs_; }

 private:
  TaggedNode tagged_node_;
  ExecutionStep *parent_ = nullptr;

  int edge_index = 0;
  EntryVector inputs_;
  EntryVector outputs_;
};

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState {
 public:
  ExecutorState(const Executor::Args &args, ExecutionContext *execution_context);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:
  // Contains a value for [node->id()] for the device context assigned by the
  // device at the beginning of a step.
  DeviceContextMap device_context_map_;

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

  // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
  const bool log_memory_;

  int64 step_id_;

  // Not owned.
  Rendezvous *rendezvous_;
  Executor::RendezvousFactory *create_rendezvous_ = nullptr;
  CollectiveExecutor *collective_executor_ = nullptr;
  SessionState *session_state_;
  string session_handle_;
  const SessionMetadata *session_metadata_ = nullptr;
  TensorStore *tensor_store_;

  // Step-local container.
  ScopedStepContainer *step_container_;
  StepStatsCollectorInterface *const stats_collector_;
  const tracing::EventCollector *const event_collector_;
  Context context_;

  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper *slice_reader_cache_;
  CallFrameInterface *call_frame_;
  CancellationManager *cancellation_manager_;
  // If not null, use this device to schedule intra-op operation
  std::unique_ptr<DeviceBase> user_device_;
  Executor::Args::Runner runner_;
  bool sync_on_finish_;

  // Owned.
  executor::MemoryWatch memory_watch_{};
  Warehouse *warehouse_;
  IterationState *iteration_state_;
  ExecutionContext *execution_context_;

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  // Available via OpKernelContext to every OpKernel invocation.
  mutex num_deferred_ops_mu_;
  int64 num_deferred_ops_ GUARDED_BY(num_deferred_ops_mu_) = 0;
  bool finish_when_deferred_ops_done_ GUARDED_BY(num_deferred_ops_mu_) = false;

  mutex mu_;
  Status status_ GUARDED_BY(mu_);

  // utillity shorthand
  void Demand(TaggedNode demanded_node) {
    LOGVAR(demanded_node.node);
    const NodeItem &node_item = *execution_context_->node(demanded_node.id());
    EntryVector outputs(node_item.num_outputs);

    Demand(demanded_node, outputs);
  }

  // Demand
  void Demand(TaggedNode demanded_node, EntryVector &outputs_vector);

  // Process a ready node in current thread.
  void Process(TaggedNode tagged_node, EntryVector &inputs_vector, EntryVector &outputs_vector);

  const Tensor *GetTensorValueForDump(const Entry &input);

  void ScheduleFinish();

  // Clean up when this executor is done.
  void Finish();
};

ExecutorState::ExecutorState(const Executor::Args &args, ExecutionContext *execution_context)
    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      create_rendezvous_(&execution_context->params().rendezvous_factory),
      collective_executor_(args.collective_executor),
      session_state_(args.session_state),
      session_handle_(args.session_handle),
      session_metadata_(execution_context->params().session_metadata),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      stats_collector_(args.stats_collector),
      event_collector_(
          tracing::GetEventCollector(tracing::EventCategory::kCompute)),
      context_(ContextKind::kThread),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      execution_context_(execution_context),
      num_outstanding_ops_(0) {
  if (args.user_intra_op_threadpool != nullptr) {
    Device *device = execution_context->params().device;
    user_device_ = RenamedDevice::NewRenamedDevice(
        device->name(), device, false, false, args.user_intra_op_threadpool);
  }

  warehouse_ = new Warehouse(executor::WarehouseStrategy::MemoryPoor, *execution_context_->graph_view());
  iteration_state_ = new IterationState();
}

ExecutorState::~ExecutorState() {
  for (auto it : device_context_map_) {
    it->Unref();
  }
  delete slice_reader_cache_;

  delete execution_context_;
  delete warehouse_;
  delete iteration_state_;
}

void ExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph *graph = execution_context_->graph();

  // Ask the device to fill in the device context map.
  Device *device = execution_context_->device();
  const Status fill_status =
      device->FillContextMap(graph, &device_context_map_);

  if (!fill_status.ok()) {
    delete this;
    done(fill_status);
    return;
  }

  TaggedNodeSeq demand_queue;
  auto root_nodes = execution_context_->root_nodes();

  TRACE();

  if (root_nodes.empty()) {
    // not much to do here ...
    delete this;
    done(Status::OK());
  }

  done_cb_ = std::move(done);

  // Initialize demand queue
  for (const Node *node : root_nodes) {
    LOGVAR(node);
    Demand(TaggedNode{node, false});
  }

  TRACE();

  ScheduleFinish();
}

void ExecutorState::Demand(TaggedNode demanded_node, EntryVector &outputs_vector) {

  LOG(INFO) << "node " << NODE_NAME(demanded_node.node) << " - demanded";

  if (warehouse_->Request(demanded_node.id(), outputs_vector)) {
    // result is available
    return;
  }

  const GraphView &graph_view = *execution_context_->graph_view();
  const NodeItem &dst_item = *graph_view.node(demanded_node.id());
  auto *execution_step = new ExecutionStep(demanded_node, dst_item);
  EntryVector &inputs_vector = execution_step->inputs();


  const EdgeInfo *edges = dst_item.input_edge_list();
  for (int in_index = 0; in_index < dst_item.num_input_edges; in_index++) {
    const EdgeInfo &input_edge = edges[in_index];
    const NodeItem &src_item = *graph_view.node(input_edge.dst_id);

    LOG(INFO) << "processing edge ["
              << NODE_NAME(src_item.node) << " -> " << NODE_NAME(dst_item.node)
              << "]";

    const bool is_control_edge = (input_edge.input_slot == Graph::kControlSlot);

    EntryVector results_vector(src_item.num_outputs);
    Demand(TaggedNode(src_item.node, false), results_vector);

    if (!is_control_edge) {
      LOG(INFO) << "copying " << NODE_SLOT(src_item.node, input_edge.input_slot) << " to "
                << NODE_SLOT(dst_item.node, input_edge.output_slot);

      inputs_vector[input_edge.output_slot] = results_vector[input_edge.input_slot];
    } else {
      LOG(INFO) << NODE_SLOT(dst_item.node, input_edge.output_slot) << " - is control edge";
    }
  }

  Process(demanded_node, inputs_vector, outputs_vector);

  warehouse_->Register(dst_item.node->id(), outputs_vector);

  delete execution_step;
}

void ExecutorState::Process(TaggedNode tagged_node, EntryVector &inputs_vector, EntryVector &outputs_vector) {
  memory_watch_.Update();
  static size_t processed_count = 0;
  processed_count++;
  LOGVAR(processed_count);

  WithContext wc(context_);
  const GraphView &graph_view = *execution_context_->graph_view();

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  DeviceContextVec input_device_contexts;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  params.step_id = step_id_;

  // Override device's threadpool if user provides an intra_op_threadpool
  Device *device = execution_context_->device();

  if (user_device_) {
    params.device = user_device_.get();
  } else {
    params.device = device;
  }

  params.log_memory = log_memory_;
  params.record_tensor_accesses = execution_context_->device_record_tensor_accesses();
  params.rendezvous = rendezvous_;
  params.create_rendezvous = create_rendezvous_;
  params.collective_executor = collective_executor_;
  params.session_state = session_state_;
  params.session_handle = session_handle_;
  params.session_metadata = session_metadata_;
  params.tensor_store = tensor_store_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library = execution_context_->params().function_library;
  params.resource_manager = device->resource_manager();
  params.step_container = step_container_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_device_contexts = &input_device_contexts;
  params.input_alloc_attrs = &input_alloc_attrs;
  params.runner = &runner_;
  params.stats_collector = stats_collector_;

  // TODO(souperk) inspect if this black magic is needed
  params.inc_num_deferred_ops_function = [this]() {
    mutex_lock lock(num_deferred_ops_mu_);
    num_deferred_ops_++;
  };
  params.dec_num_deferred_ops_function = [this]() {
    bool finish_when_deferred_ops_done = false;
    {
      mutex_lock lock(num_deferred_ops_mu_);
      num_deferred_ops_--;
      if (num_deferred_ops_ == 0) {
        finish_when_deferred_ops_done = finish_when_deferred_ops_done_;
      }
    }
    // Invoke Finish if the graph processing has completed. Finish is always
    // called exactly once per ExecutorState, either here if there are any
    // deferred ops, or in ScheduleFinish if there aren't any deferred ops.
    if (finish_when_deferred_ops_done) Finish();
  };

  Status s;

  bool completed = false;
  const Node *node = tagged_node.node;
  const int id = node->id();
  const NodeItem &item = *graph_view.node(id);

  LOG(INFO) << "processing node [" << node->name() << "::" << node->id() << "]";

  // Set the device_context for this node id, if it exists.
  if (id < device_context_map_.size()) {
    params.op_device_context = device_context_map_[id];
  }

  params.track_allocations = false;

  outputs_vector.clear();

  TensorReferenceVector accessed_tensors;
  DeviceContext *device_context = nullptr;

  // Prepares inputs.
  s = PrepareInputs(item, inputs_vector, &inputs, &input_device_contexts, &input_alloc_attrs);
  if (!s.ok()) {
    TRACE();

    // Clear inputs.
    int num_inputs = item.num_inputs;
    for (int i = 0; i < num_inputs; ++i) {
      inputs_vector.at(i).ClearVal();
    }

    return;
  }

  // Set up compute params.
  OpKernel *op_kernel = item.kernel;
  params.op_kernel = op_kernel;
  params.frame_iter = FrameAndIter(0, 0);
  params.is_input_dead = false;
  params.output_attr_array = item.output_attrs();
  params.forward_from_array = item.forward_from();

  if (item.kernel_is_async) {
    CHECK(false); // async kernels not supported right now
  } else {
    // Synchronous computes.
    OpKernelContext ctx(&params, item.num_outputs);

    LOG(INFO) << "node [" << item.node->name() << "::" << item.node->id() << "] : synchronous compute started";

    // In the common case, avoid creating any tracing objects.
    if (op_kernel->IsExpensive()) {
      KernelTimer timer;
      device->Compute(op_kernel, &ctx);
      op_kernel->UpdateCostEstimate(timer.ElapsedCycles());
    } else {
      device->Compute(op_kernel, &ctx);
    }

    LOG(INFO) << "node [" << item.node->name() << "::" << item.node->id() << "] : synchronous compute finished";

    s = ProcessOutputs(item, &ctx, outputs_vector, device_context_map_);

    if (s.ok() && execution_context_->device_record_tensor_accesses()) {
      // Get the list of all tensors accessed during the execution
      ctx.retrieve_accessed_tensors(&accessed_tensors);
      device_context = ctx.op_device_context();
    }
  }

  // Clears inputs.
  const int num_inputs = item.num_inputs;
  for (int i = 0; i < num_inputs; ++i) {
    inputs_vector.at(i).ClearVal();
  }
}

Status PrepareInputs(
    const NodeItem &item,
    EntryVector &entries,
    TensorValueVec *inputs,
    DeviceContextVec *input_device_contexts,
    AllocatorAttributeVec *input_alloc_attrs) {

  CHECK(entries.size() == item.num_inputs);
  inputs->clear();
  inputs->resize(item.num_inputs);
  input_device_contexts->clear();
  input_device_contexts->resize(item.num_inputs);
  input_alloc_attrs->clear();
  input_alloc_attrs->resize(item.num_inputs);

  for (int i = 0; i < item.num_inputs; ++i) {
    LOG(INFO) << "preparing " << NODE_SLOT(item.node, i);
    const bool expect_ref = IsRefType(item.input_type(i));

    Entry &entry = entries.at(i);
    (*input_device_contexts)[i] = entry.device_context;
    (*input_alloc_attrs)[i] = entry.alloc_attr;

    // i-th input.
    TensorValue *inp = &(*inputs)[i];

    DCHECK(entry.has_value);

    if (entry.ref == nullptr) {
      if (expect_ref) {
        return AttachDef(
            errors::InvalidArgument(i, "-th input expects a ref type"),
            item.kernel->def());
      }

      inp->tensor = entry.val.get();
    } else {
      {
        // TODO(souperk) this is suspected to cause deadlocks
        tf_shared_lock ml(*entry.ref_mu);

        if (!entry.ref->IsInitialized() && !IsInitializationOp(item.node)) {
          return AttachDef(errors::FailedPrecondition(
              "Attempting to use uninitialized value ",
              item.kernel->requested_input(i)),
                           item.kernel->def());
        }
      }

      if (expect_ref) {
        inp->mutex_if_ref = entry.ref_mu;
        inp->tensor = entry.ref;
      } else {
        // Automatically deref the tensor ref when the op expects a
        // tensor but is given a ref to a tensor.  Need to deref it
        // under the mutex.
        {
          tf_shared_lock l(*(entry.ref_mu));
          DCHECK(!entry.val_field_is_set);
          entry.val.Init(*entry.ref);
          entry.val_field_is_set = true;
        }

        entry.ref = nullptr;
        entry.ref_mu = nullptr;

        inp->tensor = entry.val.get();

        // The dtype of entry->ref could have been changed by another operation
        // that ran after the operation that "produced" it executed, so
        // re-validate that the type of the dereferenced tensor matches the
        // expected input type.
        if (item.input_type(i) != inp->tensor->dtype()) {
          TRACE();
          return AttachDef(
              errors::InvalidArgument(
                  i, "-th input expects type ",
                  DataTypeString(item.input_type(i)),
                  " but automatically dereferenced input tensor has type ",
                  DataTypeString(inp->tensor->dtype())),
              item.kernel->def());
        }
      }
    }

    LOG(INFO) << "dtype = " << DataTypeString(inp->tensor->dtype());
  }  // end for

  return Status::OK();
}

Status ProcessOutputs(
    const NodeItem &item,
    OpKernelContext *ctx,
    EntryVector &outputs_vector,
    DeviceContextMap &device_context_map) {
  const Node *node = item.node;
  DCHECK_EQ(0, outputs_vector.size());
  outputs_vector.resize(item.num_outputs);

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());

    if (s.code() == error::RESOURCE_EXHAUSTED) {
      s = Status(
          s.code(),
          strings::StrCat(
              s.error_message(),
              "\nHint: If you want to see a list of allocated tensors when "
              "OOM happens, add report_tensor_allocations_upon_oom "
              "to RunOptions for current allocation info.\n"));
    }

    return s;
  }

  // Get the device_context for this node id, if it exists.
  DeviceContext *device_context = nullptr;
  if (node->id() < device_context_map.size()) {
    device_context = device_context_map[node->id()];
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    const TensorValue val = ctx->release_output(i);

    if (val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  FormatNodeForError(*node)));
      }
    } else {
      Entry &output_entry = outputs_vector.at(i);

      // Set the device context of the output entry.
      output_entry.device_context = device_context;

      // Set the allocator attributes of the output entry.
      output_entry.alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types. We need to inspect this safely as
      // we are in the tensor buffer.
      DataType dtype = val.dtype_safe();
      if (dtype == item.output_type(i)) {

        if (val.is_ref()) {
          LOG(INFO) << "[" << node->id() << "::" << i << "] - is ref";
          output_entry.has_value = true;
          output_entry.ref = val.tensor;
          output_entry.ref_mu = val.mutex_if_ref;
        } else {
          // NOTE that std::move is used here, so val.tensor goes to
          // uninitialized state (val.tensor->IsInitialized return false).
          DCHECK(!output_entry.val_field_is_set);
          output_entry.has_value = true;
          output_entry.val_field_is_set = true;
          output_entry.val.Init(std::move(*val.tensor));
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ", FormatNodeForError(*node)));
      }
    }

    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }
  }

  return s;
}

const Tensor *ExecutorState::GetTensorValueForDump(const Entry &input) {
  if (!input.has_value) {
    return kEmptyTensor;
  } else if (input.ref == nullptr) {
    return input.val.get();
  } else {
    return input.ref;
  }
}

void ExecutorState::ScheduleFinish() {
  // Checks condition to decide if needs to invoke Finish(). If there are
  // in-flight deffered ops, wait for `num_deferred_ops_` reaches 0 to invoke
  // Finish(). Otherwise, invoke Finish() directly.
  // Note that it is critical that the ScheduleFinish / Finish codepath does not
  // block, otherwise we might deadlock.  See b/124523000 for details.
  {
    mutex_lock lock(num_deferred_ops_mu_);
    if (num_deferred_ops_ > 0) {
      finish_when_deferred_ops_done_ = true;
      return;
    }
  }

  // Finish is always called exactly once per ExecutorState, either here if
  // there aren't any deferred ops, or in the dec_num_deferred_ops_function if
  // there are deferred ops.
  Finish();
}

void ExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  auto runner = std::move(runner_);
  mu_.unlock();

  CHECK(done_cb != nullptr);
  Device *device = execution_context_->device();

  // There are several potential race conditions below. To name a few:
  // 1. Even if the device's status is OK at the precise moment when
  // num_deferred_ops_ reaches 0, it could go bad before device->RefreshStatus()
  // is called below, caused by work enqueued onto the same device by other
  // concurrent ExecutorState objects.
  // 2. Some implementations of Device::RefreshStatus, such as
  // XlaDevice::RefreshStatus, may be inherently racy because it releases the
  // device mutex after a stream pointer is acquired and before the stream is
  // queried for status.
  // 3. It's the same for some implementations of Device::Sync, such as
  // XlaDevice::Sync.
  //
  // However, these race conditions are acceptable because a stream (and
  // therefore an XlaDevice) can only go from OK to not-OK, never the opposite,
  // which means we will at worst report errors when there isn't any, never the
  // opposite.

  // An early exit for devices don't allow sync on completion. Ops that run on
  // these devices should have used num_deferred_ops correctly to ensure the
  // device has finished all relevant work at this point.
  if (!device->AllowsSyncOnCompletion()) {
    status.Update(device->RefreshStatus());
    if (!status.ok()) {
      // In device async execution mode, it's possible for device execution to
      // lag behind ExecutorState scheduling so much that this is the first
      // place a device execution error surfaces.
      // If so, all ExecutorState::NodeDone calls have already happened with OK
      // status. This is the last defense where StartCancel must be called to
      // abort all computation still running on any device.
      // TODO(b/124523000): Always call Finish in a separate thread, so even if
      // StartCancel blocks the current thread's execution, we won't encounter
      // deadlocks caused by inter-op thread exhaustion.
      if (rendezvous_) {
        rendezvous_->StartAbort(status);
      }
      if (collective_executor_) {
        collective_executor_->StartAbort(status);
      }
      if (cancellation_manager_) {
        cancellation_manager_->StartCancel();
      }
    }
    delete this;
    runner([=]() { done_cb(status); });
    return;
  }


  if (sync_on_finish_ && status.ok()) {
    // Block until the device has finished all queued operations. For
    // devices like GPUs that continue to execute Ops after their Compute
    // methods have completed, this ensures that control is not returned to
    // the user until the step (and its side-effects) has actually completed.
    device->Sync([=](Status new_status) mutable {
      LOG(INFO) << "Memory Usage :: " << memory_watch_.min_memory() << " - " << memory_watch_.max_memory();
      status.Update(new_status);
      delete this;
      runner([=]() { done_cb(status); });
    });
  } else {
    LOG(INFO) << "Memory Usage :: " << memory_watch_.min_memory() << " - " << memory_watch_.max_memory();
    delete this;
    runner([=]() { done_cb(status); });
  }
}

void OnDemandExecutor::RunAsync(const Args &args, DoneCallback done) {
  LOG(INFO) << "ON_DEMAND executor started";
  ExecutionContext *execution_context = new ExecutionContext(params_, graph_.get(), &graph_view_, root_nodes_);
  TRACE();
  (new ExecutorState(args, execution_context))->RunAsync(std::move(done));
}

}  // namespace

Status NewOnDemandExecutor(const LocalExecutorParams &params,
                           std::unique_ptr<const Graph> graph,
                           Executor **executor) {
  auto *impl = new OnDemandExecutor(params, std::move(graph));
  const Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

namespace {

class OnDemandExecutorRegistrar {
 public:
  OnDemandExecutorRegistrar() {
    auto *factory = new Factory;
    ExecutorFactory::Register("ON_DEMAND", factory);
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams &params,
                       std::unique_ptr<const Graph> graph,
                       std::unique_ptr<Executor> *out_executor) override {
      Executor *ret = nullptr;
      TF_RETURN_IF_ERROR(NewOnDemandExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static OnDemandExecutorRegistrar registrar;

}  // namespace

}  // namespace tensorflow
