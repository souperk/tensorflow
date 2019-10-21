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
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/executor/base_executor.h"
#include "tensorflow/core/common_runtime/executor/base_executor_state.h"
#include "tensorflow/core/common_runtime/executor/graph_view.h"
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

namespace tensorflow {
namespace {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

// Time the execution of kernels (in CPU cycles).  Used to dynamically identify
// inexpensive kernels which can be dispatched inline.
struct KernelTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }
};

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

class ExecutorImpl : public BaseExecutor {
 public:
  ExecutorImpl(const LocalExecutorParams& p, std::unique_ptr<const Graph> g)
      : BaseExecutor(p, std::move(g)) {
    CHECK(p.create_kernel != nullptr);
    CHECK(p.delete_kernel != nullptr);
  }

  ~ExecutorImpl() override {
    for (int i = 0; i < graph_->num_node_ids(); i++) {
      NodeItem* item = gview_.node(i);
      if (item != nullptr) {
        params_.delete_kernel(item->kernel);
      }
    }
    for (auto fiter : frame_info_) {
      delete fiter.second;
    }
  }

  Status Initialize() override;

  // TODO(souperk) why is this not implemented? does not seem to be used...

  // Process all Nodes in the current graph, attempting to infer the
  // memory allocation attributes to be used wherever they may allocate
  // a tensor buffer.
  Status SetAllocAttrs();

  void RunAsync(const Args& args, DoneCallback done) override;

 private:
  friend class ExecutorState;

  // Owned.
  LocalExecutorParams params_;
  std::unique_ptr<const Graph> graph_;
  GraphView gview_;

  // Root nodes (with no in edges) that should form the initial ready queue
  std::vector<const Node*> root_nodes_;

  // Mapping from frame name to static information about the frame.
  // TODO(yuanbyu): We could cache it along with the graph so to avoid
  // the overhead of constructing it for each executor instance.
  gtl::FlatMap<string, FrameInfo*> frame_info_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

Status ExecutorImpl::Initialize() {
  Status s = BaseExecutor::Initialize();

  if (!s.ok()) {
    return s;
  }

  for (const Node* n : graph_->nodes()) {
    // See if this node is a root node, and if so, add to root_nodes_.
    if (n->in_edges().empty()) {
      root_nodes_.push_back(n);
    }
  }

  return Status::OK();
}

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState : public BaseExecutorState {
 public:
  ExecutorState(const Executor::Args& args, ExecutorImpl* impl);
  ~ExecutorState();

 protected:
  void PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
                        EntryVector* outputs, TaggedNodeSeq* ready) override;

  // Process a ready node in current thread.
  void Process(TaggedNode node, int64 scheduled_nsec) override;

 private:
  struct AsyncState;

  // Not owned.
  Executor::RendezvousFactory* create_rendezvous_ = nullptr;
  SessionState* session_state_;
  string session_handle_;
  const SessionMetadata* session_metadata_ = nullptr;
  TensorStore* tensor_store_;
  // Step-local container.
  ScopedStepContainer* step_container_;
  const tracing::EventCollector* const event_collector_;
  Context context_;

  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  CallFrameInterface* call_frame_;
};

ExecutorState::ExecutorState(const Executor::Args& args, ExecutorImpl* impl)
    : BaseExecutorState(args, impl),
      create_rendezvous_(&impl->params_.rendezvous_factory),
      session_state_(args.session_state),
      session_handle_(args.session_handle),
      session_metadata_(impl->params_.session_metadata),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      event_collector_(
          tracing::GetEventCollector(tracing::EventCategory::kCompute)),
      context_(ContextKind::kThread),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame) {}

ExecutorState::~ExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }
  for (auto it : device_context_map_) {
    it->Unref();
  }
  delete slice_reader_cache_;
}

// State kept alive for executing an asynchronous node in another
// thread.  NOTE: We need to make a copy of p.input,
// p.input_device_contexts, and p.input_alloc_attrs for asynchronous
// kernels because OpKernelContext methods like input_type(i) needs
// the param points to valid input type vector. It's not an issue for
// sync kernels because these vectors are kept on the stack.
struct ExecutorState::AsyncState {
  AsyncState(const OpKernelContext::Params& p, const TaggedNode& _tagged_node,
             const NodeItem* _item, Entry* _first_input,
             NodeExecStatsInterface* _stats)
      : saved_inputs(*p.inputs),
        saved_input_device_contexts(*p.input_device_contexts),
        saved_input_alloc_attrs(*p.input_alloc_attrs),
        params(p),
        tagged_node(_tagged_node),
        item(_item),
        first_input(_first_input),
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs),
        stats(_stats) {
    params.inputs = &saved_inputs;
    params.input_device_contexts = &saved_input_device_contexts;
    params.input_alloc_attrs = &saved_input_alloc_attrs;
  }

  TensorValueVec saved_inputs;
  DeviceContextVec saved_input_device_contexts;
  AllocatorAttributeVec saved_input_alloc_attrs;
  OpKernelContext::Params params;
  TaggedNode tagged_node;
  const NodeItem* item;
  Entry* first_input;
  OpKernelContext ctx;
  NodeExecStatsInterface* stats;

 private:
  OpKernelContext::Params* ParamsButClearingEigenGPUDevice(
      OpKernelContext::Params* p) {
    // Ensure OpKernelContext constructor will make a new eigen GPU device if
    // necessary.
    p->eigen_gpu_device = nullptr;  // Force allocation
    return p;
  }
};

// Returns true if `item` might be traced by the given trace and event
// collectors. Returns false only if `item` definitely will not be traced.
bool MightTrace(const NodeItem& item,
                const tracing::EventCollector* event_collector) {
  // Tracing will only be enabled if either `event_collector` is non null,
  // or `trace_collector` is non-null and enabled for this particular kernel.
  // Although `profiler::TraceMe`, `tracing::ScopedAnnotation`, and
  // `tracing::ScopedRegion` check subsets of these properties internally in
  // their constructors, the cost of passing the necessary arguments to them can
  // be significant, so we avoid constructing them in the common case (when we
  // know they will not be used).
  if (event_collector != nullptr) {
    return true;
  }

  if (tracing::ScopedAnnotation::IsEnabled()) return true;

  return profiler::TraceMeRecorder::Active(
      profiler::GetTFTraceMeLevel(item.kernel->IsExpensive()));
}

void ExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
                                     const NodeItem* item, EntryVector* outputs,
                                     TaggedNodeSeq* ready) {
  auto activity_handle = absl::make_unique<profiler::TraceMe>(
      [&]() {
        return strings::StrCat("ExecutorPropagateOutputs:",
                               item->kernel->name(), "#id=", step_id_, "#");
      },
      profiler::GetTFTraceMeLevel(/*is_expensive=*/false));

  const Node* node = tagged_node.node;
  FrameState* input_frame = tagged_node.input_frame;
  const int64 input_iter = tagged_node.input_iter;
  const bool is_dead = tagged_node.is_dead;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  ready->clear();
  bool is_frame_done = false;
  FrameState* output_frame = input_frame;
  int64 output_iter = input_iter;

  if (!item->is_enter_exit_or_next_iter) {
    // Fast path for nodes types that don't need special handling
    DCHECK_EQ(input_frame, output_frame);

    // Normal path for most nodes
    mutex_lock l(input_frame->mu);
    output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);

    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &((ExecutorImpl*)executor_)->gview_, input_iter, ready);
  } else if (item->is_enter) {
    FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
    output_iter = 0;
    {
      const NodeItem* item = ((ExecutorImpl*)executor_)->gview_.node(node->id());
      mutex_lock l(output_frame->mu);

      if (item->is_constant_enter) {
        // Propagate to all active iterations if this is a loop invariant.
        output_frame->AddLoopInv(item, (*outputs)[0], ready);
      } else {
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }

      output_frame->num_pending_inputs--;
    }
    is_frame_done = input_frame->DecrementOutstandingOps(&((ExecutorImpl*)executor_)->gview_,
                                                         input_iter, ready);
  } else if (item->is_exit) {
    if (is_dead) {
      mutex_lock l(input_frame->mu);
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      is_frame_done = input_frame->DecrementOutstandingOpsLocked(
          &((ExecutorImpl*)executor_)->gview_, input_iter, ready);
    } else {
      output_frame = input_frame->parent_frame;
      output_iter = input_frame->parent_iter;
      {
        mutex_lock l(output_frame->mu);
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }
      is_frame_done = input_frame->DecrementOutstandingOps(&((ExecutorImpl*)executor_)->gview_,
                                                           input_iter, ready);
    }
  } else {
    // this is next iteration
    DCHECK(IsNextIteration(node));
    mutex_lock l(input_frame->mu);
    if (is_dead) {
      // Stop the deadness propagation.
      output_frame = nullptr;
    } else {
      if (input_iter == input_frame->iteration_count &&
          input_frame->num_outstanding_iterations ==
              input_frame->max_parallel_iterations) {
        // Reached the maximum for parallel iterations.
        input_frame->next_iter_roots.push_back({node, (*outputs)[0]});
        output_frame = nullptr;
      } else {
        // If this is a new iteration, start it.
        if (input_iter == input_frame->iteration_count) {
          input_frame->IncrementIteration(&((ExecutorImpl*)executor_)->gview_, ready);
        }
        output_iter = input_iter + 1;
      }
    }

    if (output_frame != nullptr) {
      // This is the case when node is not Enter, Exit, or NextIteration.
      DCHECK(input_frame == output_frame);
      output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
    }

    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &((ExecutorImpl*)executor_)->gview_, input_iter, ready);
  }

  // At this point, this node is completely done. We also know if the
  // completion of this node makes its frame completed.
  if (is_frame_done) {
    FrameState* parent_frame = input_frame->parent_frame;
    const int64 parent_iter = input_frame->parent_iter;
    DeleteFrame(input_frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_nsec) {
  WithContext wc(context_);
  const GraphView& gview = ((ExecutorImpl*)executor_)->gview_;
  TaggedNodeSeq ready;
  TaggedNodeReadyQueue inline_ready;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  DeviceContextVec input_device_contexts;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  params.step_id = step_id_;
  // Override device's threadpool if user provides an intra_op_threadpool
  Device* device = ((ExecutorImpl*)executor_)->params_.device;
  if (user_device_) {
    params.device = user_device_.get();
  } else {
    params.device = device;
  }

  params.log_memory = log_memory_;
  params.record_tensor_accesses =
      ((ExecutorImpl*)executor_)->device_record_tensor_accesses_;
  params.rendezvous = rendezvous_;
  params.create_rendezvous = create_rendezvous_;
  params.collective_executor = collective_executor_;
  params.session_state = session_state_;
  params.session_handle = session_handle_;
  params.session_metadata = session_metadata_;
  params.tensor_store = tensor_store_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library =
      ((ExecutorImpl*)executor_)->params_.function_library;
  params.resource_manager = device->resource_manager();
  params.step_container = step_container_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_device_contexts = &input_device_contexts;
  params.input_alloc_attrs = &input_alloc_attrs;
  params.runner = &runner_;
  params.stats_collector = stats_collector_;
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
  NodeExecStatsInterface* stats = nullptr;

  EntryVector outputs;
  bool completed = false;
  inline_ready.push_back(tagged_node);
  while (!inline_ready.empty()) {
    tagged_node = inline_ready.front();
    inline_ready.pop_front();
    const Node* node = tagged_node.node;
    FrameState* input_frame = tagged_node.input_frame;
    const int64 input_iter = tagged_node.input_iter;
    const int id = node->id();
    const NodeItem& item = *gview.node(id);

    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      mutex_lock l(input_frame->mu);
      input_frame->GetIteration(input_iter)->mark_started(item.pending_id);
    }

    // Set the device_context for this node id, if it exists.
    if (id < device_context_map_.size()) {
      params.op_device_context = device_context_map_[id];
    }

    params.track_allocations = false;
    stats = nullptr;

    if (stats_collector_ && !tagged_node.is_dead) {
      stats = stats_collector_->CreateNodeExecStats(node);
      // Track allocations if and only if we are collecting statistics, and
      // `stats` object is expecting allocations to be tracked.
      params.track_allocations = stats ? stats->TrackAllocations() : false;
      nodestats::SetScheduled(stats, scheduled_nsec);
      nodestats::SetAllStart(stats);
    }

    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
              << SummarizeNode(*node) << (tagged_node.is_dead ? " is dead" : "")
              << " device: " << device->name();
    }

    Entry* input_tensors = GetInputTensors(input_frame, input_iter);
    Entry* first_input = input_tensors + item.input_start;
    outputs.clear();

    TensorReferenceVector accessed_tensors;
    DeviceContext* device_context = nullptr;

    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;
    if (tagged_node.is_dead && !IsTransferNode(node)) {
      outputs.resize(item.num_outputs);
    } else {
      // Prepares inputs.
      bool is_input_dead = false;
      s = PrepareInputs(item, first_input, &inputs, &input_device_contexts,
                        &input_alloc_attrs, &is_input_dead);
      if (!s.ok()) {
        // Clear inputs.
        int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->ClearVal();
        }
        MaybeMarkCompleted(input_frame, input_iter, id);
        // Continue to process the nodes in 'inline_ready'.
        completed = NodeDone(s, item.node, ready, stats, &inline_ready);
        continue;
      }

      // Set up compute params.
      OpKernel* op_kernel = item.kernel;
      params.op_kernel = op_kernel;
      params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
      params.is_input_dead = is_input_dead;
      params.output_attr_array = item.output_attrs();
      params.forward_from_array = item.forward_from();

      if (item.kernel_is_async) {
        // Asynchronous computes.
        AsyncOpKernel* async = item.kernel->AsAsync();
        DCHECK(async != nullptr);
        launched_asynchronously = true;
        AsyncState* state =
            new AsyncState(params, tagged_node, &item, first_input, stats);

        auto done = [this, state]() {
          Device* device = ((ExecutorImpl*)executor_)->params_.device;
          NodeExecStatsInterface* stats = state->stats;  // Shorthand
          Entry* first_input = state->first_input;       // Shorthand

          nodestats::SetOpEnd(stats);
          EntryVector outputs;
          Status s = ProcessOutputs(*state->item, &state->ctx, &outputs, stats);
          nodestats::SetMemory(stats, &state->ctx);
          if (vlog_) {
            VLOG(2) << "Async kernel done: " << state->item->node->id()
                    << " step " << step_id_ << " "
                    << SummarizeNode(*state->item->node)
                    << (state->tagged_node.is_dead ? " is dead" : "")
                    << " device: " << device->name();
          }

          // Clears inputs.
          const int num_inputs = state->item->num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->ClearVal();
          }
          FrameState* input_frame = state->tagged_node.input_frame;
          const int64 input_iter = state->tagged_node.input_iter;
          const int id = state->tagged_node.node->id();
          MaybeMarkCompleted(input_frame, input_iter, id);
          TaggedNodeSeq ready;

          if (s.ok()) {
            PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
          }

          outputs.clear();
          if (s.ok() &&
              ((ExecutorImpl*)executor_)->device_record_tensor_accesses_) {
            // Get the list of all tensors accessed during the execution
            TensorReferenceVector accessed;
            state->ctx.retrieve_accessed_tensors(&accessed);
            nodestats::SetReferencedTensors(stats, accessed);
            // callee takes ownership of the vector
            device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(),
                                                 accessed);
          }
          const bool completed =
              NodeDone(s, state->item->node, ready, stats, nullptr);
          delete state;
          if (completed) ScheduleFinish();
        };

        nodestats::SetOpStart(stats);
        {
          profiler::TraceMe activity(
              [&] {
                return strings::StrCat(
                    op_kernel->name(), ":", op_kernel->type_string(),
                    "#id=", step_container_ ? step_container_->step_id() : 0,
                    ",device=", device->name(), ",async=true#");
              },
              profiler::GetTFTraceMeLevel(op_kernel->IsExpensive()));
          device->ComputeAsync(async, &state->ctx, done);
        }
      } else {
        // Synchronous computes.
        OpKernelContext ctx(&params, item.num_outputs);
        nodestats::SetOpStart(stats);

        if (TF_PREDICT_FALSE(MightTrace(item, event_collector_))) {
          const string& op_name = op_kernel->name();
          const string kernel_label = strings::StrCat(
              op_name, ":", op_kernel->type_string(),
              "#id=", step_container_ ? step_container_->step_id() : 0,
              ",device=", device->name(), ",async=false#");
          tracing::ScopedRegion region(tracing::EventCategory::kCompute,
                                       op_name);
          // 'TraceMe' will trace the OpKernel scheduling time.
          profiler::TraceMe activity(
              absl::string_view(kernel_label),
              profiler::GetTFTraceMeLevel(op_kernel->IsExpensive()));
          // 'ScopedAnnotation' will trace the OpKernel execution time.
          tracing::ScopedAnnotation annotation(kernel_label);
          device->Compute(op_kernel, &ctx);
        } else {
          // In the common case, avoid creating any tracing objects.
          if (op_kernel->IsExpensive()) {
            KernelTimer timer;
            device->Compute(op_kernel, &ctx);
            op_kernel->UpdateCostEstimate(timer.ElapsedCycles());
          } else {
            device->Compute(op_kernel, &ctx);
          }
        }

        nodestats::SetOpEnd(stats);
        s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (s.ok() &&
            ((ExecutorImpl*)executor_)->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }
        nodestats::SetMemory(stats, &ctx);
      }
    }

    if (!launched_asynchronously) {
      if (vlog_) {
        VLOG(2) << "Synchronous kernel done: " << id << " step "
                << params.step_id << " " << SummarizeNode(*node)
                << (tagged_node.is_dead ? " is dead: " : "")
                << " device: " << device->name();
      }

      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      MaybeMarkCompleted(input_frame, input_iter, id);

      // Propagates outputs.
      if (s.ok()) {
        PropagateOutputs(tagged_node, &item, &outputs, &ready);
      }

      outputs.clear();

      if (!accessed_tensors.empty()) {
        nodestats::SetReferencedTensors(stats, accessed_tensors);
        // device_context is set above in synchronous computes
        device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
      }

      if (stats) {
        scheduled_nsec = nodestats::NowInNsec();
      }
      // Postprocess.
      completed = NodeDone(s, item.node, ready, stats, &inline_ready);
    }
  }  // while !inline_ready.empty()

  // This thread of computation is done if completed = true.
  if (completed) ScheduleFinish();
}

void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  (new ExecutorState(args, this))->RunAsync(std::move(done));
}

}  // namespace

Status NewLocalExecutor(const LocalExecutorParams& params,
                        std::unique_ptr<const Graph> graph,
                        Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params, std::move(graph));
  const Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel) {
  const auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib, ndef,
                        graph_def_version, kernel);
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

namespace {

class DefaultExecutorRegistrar {
 public:
  DefaultExecutorRegistrar() {
    Factory* factory = new Factory;
    ExecutorFactory::Register("", factory);
    ExecutorFactory::Register("DEFAULT", factory);
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params,
                       std::unique_ptr<const Graph> graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret = nullptr;
      TF_RETURN_IF_ERROR(NewLocalExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static DefaultExecutorRegistrar registrar;

}  // namespace

}  // namespace tensorflow
