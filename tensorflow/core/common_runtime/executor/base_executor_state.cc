//
// Created by kostas on 20/10/19.
//

#include "tensorflow/core/common_runtime/executor/base_executor_state.h"

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor/base_executor.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/platform/default/tracing_impl.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

// Helper routines for collecting step stats.
namespace nodestats {

void SetScheduled(NodeExecStatsInterface* stats, int64 micros) {
  if (!stats) return;
  stats->SetScheduled(micros * EnvTime::kMicrosToNanos);
}

void SetAllStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorStarted();
}

void SetOpStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeStarted();
}

void SetOpEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeEnded();
}

void SetAllEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorEnded();
}

void SetOutput(NodeExecStatsInterface* stats, int slot, const Tensor* v) {
  if (!stats) return;
  stats->SetOutput(slot, v);
}

void SetMemory(NodeExecStatsInterface* stats, OpKernelContext* ctx) {
  if (!stats) return;
  stats->SetMemory(ctx);
}

void SetReferencedTensors(NodeExecStatsInterface* stats,
                          const TensorReferenceVector& tensors) {
  if (!stats) return;
  stats->SetReferencedTensors(tensors);
}

}  // namespace nodestats

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

bool IsInitializationOp(const Node* node) {
  return node->op_def().allows_uninitialized_input();
}

void FrameState::InitializeFrameInfo(const string& enter_name) {
  auto it_frame_info = executor->frame_info_.find(enter_name);
  DCHECK(it_frame_info != executor->frame_info_.end());
  FrameInfo* finfo = it_frame_info->second;
  pending_counts = finfo->pending_counts;
  total_input_tensors = finfo->total_inputs;
  num_pending_inputs = finfo->input_count;
  nodes = finfo->nodes;
}

void FrameState::ActivateNodes(const NodeItem* item, const bool is_dead,
                               int64 iter, EntryVector* outputs,
                               TaggedNodeSeq* ready) {
  const GraphView& gview = executor->gview_;
  IterationState* iter_state = GetIteration(iter);
  const size_t num_output_edges = item->num_output_edges;
  const EdgeInfo* edges = item->output_edge_list();
  Entry* input_tensors = iter_state->input_tensors;

  for (size_t out_index = 0; out_index < num_output_edges; out_index++) {
    const EdgeInfo& e = edges[out_index];
    const int dst_id = e.dst_id;
    const NodeItem* dst_item = gview.node(dst_id);
    const PendingCounts::Handle dst_pending_id = dst_item->pending_id;
    const int src_slot = e.output_slot;

    // TODO(yuanbyu): We don't need this if we require the subgraph
    // given to an executor not to contain a sink node.
    if (dst_item->is_sink) continue;

    bool dst_dead = false;
    bool dst_ready = false;

    // True iff this input for dst is needed. We only set this input for
    // dst if this flag is true. This is needed to make the thread safety
    // analysis happy.
    const bool is_control_edge = (src_slot == Graph::kControlSlot);
    bool dst_need_input = !is_control_edge;
    if (dst_item->is_merge) {
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are
      // dead. For Merge, pending's LSB is set iff a live data input has
      // arrived.
      if (is_control_edge) {
        iter_state->decrement_pending(dst_pending_id, 2);
        int count = iter_state->pending(dst_pending_id);
        int dead_cnt = iter_state->dead_count(dst_pending_id);
        dst_dead = (dead_cnt == dst_item->num_inputs);
        dst_ready = (count == 0) || ((count == 1) && dst_dead);
      } else {
        if ((*outputs)[src_slot].has_value) {
          // This is a live data input.
          int count = iter_state->pending(dst_pending_id);
          iter_state->mark_live(dst_pending_id);
          // Only the first live edge sets the input and (potentially)
          // triggers execution. The low bit of count is set if and
          // only if no live input has been used yet (mark_live clears
          // it). The node should be started if and only if this is
          // the first live input and there are no pending control
          // edges, i.e. count == 1.
          dst_ready = (count == 1);
          dst_need_input = ((count & 0x1) == 1);
        } else {
          // This is a dead data input. Note that dst_node is dead if node is
          // a dead enter. We need this to handle properly a while loop on
          // the untaken branch of a conditional.
          // TODO(yuanbyu): This is a bit hacky, but a good solution for
          // now.
          iter_state->increment_dead_count(dst_pending_id);
          const int dead_cnt = iter_state->dead_count(dst_pending_id);
          dst_dead = (dead_cnt == dst_item->num_inputs) || item->is_enter;
          dst_ready = (iter_state->pending(dst_pending_id) == 1) && dst_dead;
          dst_need_input = false;
        }
      }
    } else {
      const bool increment_dead =
          (is_dead || (!is_control_edge && !(*outputs)[src_slot].has_value));
      int pending, dead;
      iter_state->adjust_for_activation(dst_pending_id, increment_dead,
                                        &pending, &dead);
      dst_dead = (dead > 0);
      dst_ready = (pending == 0);
    }

    if (dst_need_input) {
      const int dst_slot = e.input_slot;
      const int dst_loc = dst_item->input_start + dst_slot;
      if (e.is_last) {
        input_tensors[dst_loc] = std::move((*outputs)[src_slot]);
      } else {
        input_tensors[dst_loc] = (*outputs)[src_slot];
      }
    }

    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      if (dst_item->is_control_trigger) dst_dead = false;
      ready->emplace_back(dst_item->node, this, iter, dst_dead);
      iter_state->outstanding_ops++;
    }
  }
}

void FrameState::ActivateNexts(const GraphView* gview, int64 iter,
                               TaggedNodeSeq* ready) {
  // Propagate the deferred NextIteration nodes to the new iteration.
  for (auto& node_entry : next_iter_roots) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    const NodeItem* item = gview->node(node->id());
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
  next_iter_roots.clear();
}

void FrameState::ActivateLoopInvs(const GraphView* gview, int64 iter,
                                  TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : inv_values) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    const NodeItem* item = gview->node(node->id());
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
}

void FrameState::AddLoopInv(const NodeItem* item, const Entry& entry,
                            TaggedNodeSeq* ready) {
  // Store this value.
  inv_values.push_back({item->node, entry});

  // Make this value available to all iterations.
  const bool is_dead = !entry.has_value;
  for (int i = 0; i <= iteration_count; ++i) {
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, i, &outputs, ready);
  }
}

bool FrameState::IsIterationDone(int64 iter) {
  IterationState* iter_state = GetIteration(iter);
  if (iter_state->outstanding_ops == 0 &&
      iter_state->outstanding_frame_count == 0) {
    if (iter == 0) {
      // The enclosing frame has no pending input.
      return num_pending_inputs == 0;
    } else {
      // The preceding iteration is deleted (and therefore done).
      return (GetIteration(iter - 1) == nullptr);
    }
  }
  return false;
}

void FrameState::IncrementIteration(const GraphView* gview,
                                    TaggedNodeSeq* ready) {
  iteration_count++;
  const int64 next_iter = iteration_count;

  // Initialize the next iteration.
  IterationState* iter_state =
      new IterationState(pending_counts, total_input_tensors);
  SetIteration(next_iter, iter_state);
  num_outstanding_iterations++;
  dead_exits.clear();

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(gview, next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(gview, next_iter, ready);
}

bool FrameState::CleanupIterations(const GraphView* gview, int64 iter,
                                   TaggedNodeSeq* ready) {
  int64 curr_iter = iter;
  while (curr_iter <= iteration_count && IsIterationDone(curr_iter)) {
    // Delete the iteration curr_iter.
    delete GetIteration(curr_iter);
    SetIteration(curr_iter, nullptr);
    --num_outstanding_iterations;
    ++curr_iter;

    // When one iteration is completed, we check for deferred iteration,
    // and start it if there is one.
    if (!next_iter_roots.empty()) {
      IncrementIteration(gview, ready);
    }
  }
  return IsFrameDone();
}

BaseExecutorState::BaseExecutorState(const Executor::Args& args,
                                     BaseExecutor* executor)
    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      collective_executor_(args.collective_executor),
      stats_collector_(args.stats_collector),
      cancellation_manager_(args.cancellation_manager),
      executor_(executor),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      num_outstanding_ops_(0) {
  if (args.user_intra_op_threadpool != nullptr) {
    Device* device = executor_->params_.device;
    user_device_ = RenamedDevice::NewRenamedDevice(
        device->name(), device, false, false, args.user_intra_op_threadpool);
  }

  // We start the entire execution in iteration 0 of the root frame
  // so let us create the root frame and the state for iteration 0.
  // We assume root_frame_->frame_name.empty().
  root_frame_ = new FrameState(executor_, 1);
  root_frame_->frame_id = 0;  // must be 0
  root_frame_->InitializeFrameInfo(root_frame_->frame_name);

  // Initialize iteration 0.
  root_frame_->iterations.resize(root_frame_->max_parallel_iterations);
  root_frame_->iterations[0] = new IterationState(
      root_frame_->pending_counts, root_frame_->total_input_tensors);

  outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

void BaseExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = executor_->graph_.get();
  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = executor_->params_.device;
  const Status fill_status =
      device->FillContextMap(graph, &device_context_map_);
  if (!fill_status.ok()) {
    delete this;
    done(fill_status);
    return;
  }

  // Initialize the ready queue.
  for (const Node* n : executor_->root_nodes_) {
    // souperk - DCHECK is not longer relevant
    //    DCHECK_EQ(n->in_edges().size(), 0);
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
  }

  if (ready.empty()) {
    delete this;
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = std::move(done);

    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}

Status BaseExecutorState::PrepareInputs(
    const NodeItem& item, Entry* first_input, TensorValueVec* inputs,
    DeviceContextVec* input_device_contexts,
    AllocatorAttributeVec* input_alloc_attrs, bool* is_input_dead) {
  const Node* node = item.node;

  inputs->clear();
  inputs->resize(item.num_inputs);
  input_device_contexts->clear();
  input_device_contexts->resize(item.num_inputs);
  input_alloc_attrs->clear();
  input_alloc_attrs->resize(item.num_inputs);

  *is_input_dead = false;

  bool is_merge = item.is_merge;
  for (int i = 0; i < item.num_inputs; ++i) {
    const bool expect_ref = IsRefType(item.input_type(i));
    Entry* entry = first_input + i;
    (*input_device_contexts)[i] = entry->device_context;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    // Only merge and transfer nodes can have no-value inputs.
    if (!entry->has_value) {
      if (!is_merge) {
        DCHECK(IsTransferNode(node)) << node->name() << " - input " << i;
        DCHECK(!entry->val_field_is_set) << node->name() << " - input " << i;
        entry->has_value = true;
        entry->val_field_is_set = true;
        entry->val.Init(*kEmptyTensor);
        inp->tensor = entry->val.get();
        *is_input_dead = true;
      }
      continue;
    }

    if (entry->ref == nullptr) {
      if (expect_ref) {
        return AttachDef(
            errors::InvalidArgument(i, "-th input expects a ref type"),
            item.kernel->def());
      }
      inp->tensor = entry->val.get();
    } else {
      {
        tf_shared_lock ml(*entry->ref_mu);
        if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
          return AttachDef(errors::FailedPrecondition(
                               "Attempting to use uninitialized value ",
                               item.kernel->requested_input(i)),
                           item.kernel->def());
        }
      }

      if (expect_ref) {
        inp->mutex_if_ref = entry->ref_mu;
        inp->tensor = entry->ref;
      } else {
        // Automatically deref the tensor ref when the op expects a
        // tensor but is given a ref to a tensor.  Need to deref it
        // under the mutex.
        {
          tf_shared_lock l(*(entry->ref_mu));
          DCHECK(!entry->val_field_is_set);
          entry->val.Init(*entry->ref);
          entry->val_field_is_set = true;
        }
        entry->ref = nullptr;
        entry->ref_mu = nullptr;

        inp->tensor = entry->val.get();
        // The dtype of entry->ref could have been changed by another operation
        // that ran after the operation that "produced" it executed, so
        // re-validate that the type of the dereferenced tensor matches the
        // expected input type.
        if (item.input_type(i) != inp->tensor->dtype()) {
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
  }
  return Status::OK();
}

Status BaseExecutorState::ProcessOutputs(const NodeItem& item,
                                         OpKernelContext* ctx,
                                         EntryVector* outputs,
                                         NodeExecStatsInterface* stats) {
  const Node* node = item.node;
  DCHECK_EQ(0, outputs->size());
  outputs->resize(item.num_outputs);

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
      DumpState();
    }
    if (s.code() == error::RESOURCE_EXHAUSTED) {
      if (stats_collector_) {
        string err = stats_collector_->ReportAllocsOnResourceExhausted(
            s.error_message());
        s = Status(s.code(), strings::StrCat(s.error_message(), err));
      } else {
        s = Status(
            s.code(),
            strings::StrCat(
                s.error_message(),
                "\nHint: If you want to see a list of allocated tensors when "
                "OOM happens, add report_tensor_allocations_upon_oom "
                "to RunOptions for current allocation info.\n"));
      }
    }
    return s;
  }

  // Get the device_context for this node id, if it exists.
  DeviceContext* device_context = nullptr;
  if (node->id() < device_context_map_.size()) {
    device_context = device_context_map_[node->id()];
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
      Entry* out = &((*outputs)[i]);

      // Set the device context of the output entry.
      out->device_context = device_context;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types. We need to inspect this safely as
      // we are in the tensor buffer.
      DataType dtype = val.dtype_safe();
      if (dtype == item.output_type(i)) {
        if (stats && val.tensor->IsInitialized()) {
          nodestats::SetOutput(stats, i, val.tensor);
        }
        if (val.is_ref()) {
          out->has_value = true;
          out->ref = val.tensor;
          out->ref_mu = val.mutex_if_ref;
          if (log_memory_) {
            Tensor to_log;
            {
              // Dereference the tensor under the lock.
              tf_shared_lock l(*out->ref_mu);
              to_log = *out->ref;
            }
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, to_log);
          }
        } else {
          // NOTE that std::move is used here, so val.tensor goes to
          // uninitialized state (val.tensor->IsInitialized return false).
          DCHECK(!out->val_field_is_set);
          out->has_value = true;
          out->val_field_is_set = true;
          out->val.Init(std::move(*val.tensor));
          if (log_memory_) {
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, *out->val);
          }
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

void BaseExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  TaggedNodeReadyQueue* inline_ready) {
  if (ready.empty()) return;

  int64 scheduled_nsec = 0;
  if (stats_collector_) {
    scheduled_nsec = nodestats::NowInNsec();
  }

  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
      runner_([=]() { Process(tagged_node, scheduled_nsec); });
    }
    return;
  }

  const GraphView& gview = (executor_)->gview_;
  const TaggedNode* curr_expensive_node = nullptr;
  for (auto& tagged_node : ready) {
    const NodeItem& item = *gview.node(tagged_node.node->id());
    if (tagged_node.is_dead || !item.kernel->IsExpensive()) {
      // Inline this inexpensive node.
      inline_ready->push_back(tagged_node);
    } else {
      if (curr_expensive_node) {
        // Dispatch to another thread since there is plenty of work to
        // do for this thread.
        runner_(std::bind(&BaseExecutorState::Process, this, *curr_expensive_node,
                          scheduled_nsec));
      }
      curr_expensive_node = &tagged_node;
    }
  }

  if (curr_expensive_node) {
    if (inline_ready->empty()) {
      // Tail recursion optimization
      inline_ready->push_back(*curr_expensive_node);
    } else {
      // There are inline nodes to run already. We dispatch this expensive
      // node to other thread.
      runner_(std::bind(&BaseExecutorState::Process, this, *curr_expensive_node,
                        scheduled_nsec));
    }
  }
}


bool BaseExecutorState::NodeDone(const Status& s, const Node* node,
                                 const TaggedNodeSeq& ready,
                                 NodeExecStatsInterface* stats,
                                 TaggedNodeReadyQueue* inline_ready) {
  nodestats::SetAllEnd(stats);
  if (stats) {
    if (stats_collector_) {
      stats->Done(executor_->params_.device->name());
    } else {
      delete stats;
    }
  }

  bool abort_run = false;
  if (!s.ok()) {
    // Some error happened. This thread of computation is done.
    mutex_lock l(mu_);

    if (status_.ok()) {
      abort_run = true;

      // If execution has been cancelled, mark any new errors as being derived.
      // This ensures any errors triggered by cancellation are marked as
      // derived.
      if (cancellation_manager_ && cancellation_manager_->IsCancelled()) {
        status_ = StatusGroup::MakeDerived(s);
      } else {
        status_ = s;
      }
    }
  }

  if (abort_run) {
    TRACEPRINTF("StartAbort: %s", s.ToString().c_str());
    if (cancellation_manager_) {
      // only log when the abort happens during the actual run time.
      auto device_name = executor_->params_.device->name();
      // Use VLOG instead of LOG(warning) because error status is expected when
      // the executor is run under the grappler optimization phase or when
      // iterating through a tf.data input pipeline.
      VLOG(1) << "[" << device_name << "] Executor start aborting: " << s;
    }

    if (rendezvous_) {
      rendezvous_->StartAbort(s);
    }
    if (collective_executor_) {
      collective_executor_->StartAbort(s);
    }
    if (cancellation_manager_) {
      cancellation_manager_->StartCancel();
    }
  }

  bool completed = false;
  const size_t ready_size = ready.size();
  if (ready_size == 0 || !s.ok()) {
    completed = (num_outstanding_ops_.fetch_sub(1) == 1);
  } else if (ready_size > 1) {
    num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);
  }

  // Schedule the ready nodes in 'ready'.
  if (s.ok()) {
    ScheduleReady(ready, inline_ready);
  }

  return completed;
}

void BaseExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter,
                                               const Node* node,
                                               FrameState** child) {
  // Get the child frame name.
  string enter_name;
  Status s = GetNodeAttr(node->attrs(), "frame_name", &enter_name);
  DCHECK(s.ok()) << s;
  const string child_name = MakeFrameName(frame, iter, enter_name);

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
      return;
    }
  }

  // Need to create a new frame instance.
  // Note that this new frame instance is created without any locks.
  if (vlog_) VLOG(2) << "Create frame: " << child_name;

  int parallel_iters;
  s = GetNodeAttr(node->attrs(), "parallel_iterations", &parallel_iters);
  DCHECK(s.ok()) << s;

  auto* temp = new FrameState(executor_, parallel_iters);
  temp->frame_name = child_name;
  temp->frame_id = Hash64(child_name);
  temp->parent_frame = frame;
  temp->parent_iter = iter;
  temp->InitializeFrameInfo(enter_name);

  // 'iterations' is a fixed-length circular buffer.
  temp->iterations.resize(temp->max_parallel_iterations + 1);
  // Initialize iteration 0.
  temp->iterations[0] =
      new IterationState(temp->pending_counts, temp->total_input_tensors);

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
    } else {
      mutex_lock frame_lock(frame->mu);
      frame->GetIteration(iter)->outstanding_frame_count++;
      outstanding_frames_[child_name] = temp;
      *child = temp;
      temp = nullptr;
    }
  }

  delete temp;  // Not used so delete it.
}

void BaseExecutorState::DeleteFrame(FrameState* frame, TaggedNodeSeq* ready) {
  // First, propagate dead_exits (if any) to the parent frame.
  FrameState* parent_frame = frame->parent_frame;
  const int64 parent_iter = frame->parent_iter;
  if (parent_frame != nullptr) {
    mutex_lock parent_frame_lock(parent_frame->mu);
    // Propagate all the dead exits to the parent frame.
    mutex_lock this_frame_lock(frame->mu);
    for (const Node* node : frame->dead_exits) {
      auto parent_iter_state = parent_frame->GetIteration(parent_iter);
      for (const Edge* e : node->out_edges()) {
        const Node* dst_node = e->dst();

        const auto dst_pending_id =
            executor_->gview_.node(dst_node->id())->pending_id;

        // TODO(yuanbyu): We don't need this if we require the subgraph
        // given to an executor not to contain a sink node.
        if (dst_node->IsSink()) continue;

        bool dst_dead = true;
        bool dst_ready = false;
        // We know this is a dead input to dst.
        if (IsMerge(dst_node)) {
          if (e->IsControlEdge()) {
            parent_iter_state->decrement_pending(dst_pending_id, 2);
            int count = parent_iter_state->pending(dst_pending_id);
            int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
            dst_dead = (dead_cnt == dst_node->num_inputs());
            dst_ready = (count == 0) || ((count == 1) && dst_dead);
          } else {
            parent_iter_state->increment_dead_count(dst_pending_id);
            const int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
            dst_dead = (dead_cnt == dst_node->num_inputs());
            dst_ready =
                (parent_iter_state->pending(dst_pending_id) == 1) && dst_dead;
          }
        } else {
          parent_iter_state->increment_dead_count(dst_pending_id);
          dst_ready =
              (parent_iter_state->decrement_pending(dst_pending_id, 1) == 0);
        }

        if (dst_ready) {
          if (IsControlTrigger(dst_node)) dst_dead = false;
          ready->emplace_back(dst_node, parent_frame, parent_iter, dst_dead);
          parent_iter_state->outstanding_ops++;
        }
      }
    }
  }

  // Delete the frame.
  const string& frame_name = frame->frame_name;
  if (vlog_) VLOG(2) << "Delete frame " << frame_name;
  {
    mutex_lock executor_lock(mu_);
    outstanding_frames_.erase(frame_name);
  }
  delete frame;
}

const Tensor* BaseExecutorState::GetTensorValueForDump(const Entry& input) {
  if (!input.has_value) {
    return kEmptyTensor;
  } else if (input.ref == nullptr) {
    return input.val.get();
  } else {
    return input.ref;
  }
}

void BaseExecutorState::MaybeMarkCompleted(FrameState* frame, int64 iter, int64 node_id) {
  // TODO(misard) Replace with a finer-grain enabling flag once we
  // add better optional debugging support.

  if (vlog_ && VLOG_IS_ON(1)) {
    const NodeItem* item = executor_->gview_.node(node_id);
    mutex_lock l(frame->mu);
    frame->GetIteration(iter)->mark_completed(item->pending_id);
  }
}


void BaseExecutorState::DumpPendingNodeState(
    const int node_id, const Entry* input_vector,
    const bool show_nodes_with_no_ready_inputs) {
  const NodeItem& node_item = *executor_->gview_.node(node_id);
  const Node& node = *node_item.node;
  const int input_base = node_item.input_start;
  if (!show_nodes_with_no_ready_inputs) {
    bool has_ready_input = false;
    for (int i = 0; i < node.num_inputs(); ++i) {
      const Entry& input = input_vector[input_base + i];
      const Tensor* tensor = GetTensorValueForDump(input);
      if (tensor->IsInitialized()) {
        has_ready_input = true;
        break;
      }
    }
    if (!has_ready_input) {
      return;
    }
  }
  LOG(WARNING) << "    Pending Node: " << node.DebugString();
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void BaseExecutorState::DumpActiveNodeState(const int node_id,
                                            const Entry* input_vector) {
  const NodeItem& node_item = *executor_->gview_.node(node_id);
  const Node& node = *node_item.node;
  LOG(WARNING) << "    Active Node: " << node.DebugString();
  const int input_base = node_item.input_start;
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void BaseExecutorState::DumpIterationState(const FrameState* frame,
                                           IterationState* iteration) {
  const std::vector<const Node*>* nodes = frame->nodes;
  // Dump any waiting nodes that are holding on to tensors.
  for (const Node* node : *nodes) {
    const int node_id = node->id();
    PendingCounts::Handle pending_id =
        executor_->gview_.node(node_id)->pending_id;
    if (iteration->node_state(pending_id) == PendingCounts::PENDING_NOTREADY ||
        iteration->node_state(pending_id) == PendingCounts::PENDING_READY) {
      DumpPendingNodeState(node_id, iteration->input_tensors, false);
    }
  }
  // Then the active nodes.
  for (const Node* node : *nodes) {
    const int node_id = node->id();
    PendingCounts::Handle pending_id =
        executor_->gview_.node(node_id)->pending_id;
    if (iteration->node_state(pending_id) == PendingCounts::STARTED) {
      DumpActiveNodeState(node_id, iteration->input_tensors);
    }
  }
  // Show all input tensors in use.
  const int total_input_tensors = frame->total_input_tensors;
  size_t total_bytes = 0;
  for (int i = 0; i < total_input_tensors; ++i) {
    const Entry& input = iteration->input_tensors[i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(),
                          ", bytes: ", tensor->TotalBytes(), ">");
      total_bytes += tensor->TotalBytes();
    }
  }
  LOG(WARNING) << "    Total bytes " << total_bytes;
}

void BaseExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter,
                                                TaggedNodeSeq* ready) {
  bool is_frame_done = false;
  {
    mutex_lock frame_lock(frame->mu);
    frame->GetIteration(iter)->outstanding_frame_count--;
    is_frame_done = frame->CleanupIterations(&executor_->gview_, iter, ready);
  }
  if (is_frame_done) {
    FrameState* parent_frame = frame->parent_frame;
    const int64 parent_iter = frame->parent_iter;
    DeleteFrame(frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

void BaseExecutorState::DumpState() {
  mutex_lock l(mu_);
  if (!dumped_on_error_) {
    LOG(WARNING) << "Dumping state";
    for (auto& frame : outstanding_frames_) {
      LOG(WARNING) << frame.first;
      FrameState* frame_state = frame.second;
      mutex_lock frame_lock(frame_state->mu);
      for (IterationState* iteration : frame_state->iterations) {
        LOG(WARNING) << "  Iteration:";
        DumpIterationState(frame_state, iteration);
      }
    }
    dumped_on_error_ = true;
  }
}

void BaseExecutorState::ScheduleFinish() {
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

void BaseExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  auto runner = std::move(runner_);
  mu_.unlock();
  CHECK(done_cb != nullptr);
  Device* device = executor_->params_.device;

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
      status.Update(new_status);
      delete this;
      runner([=]() { done_cb(status); });
    });
  } else {
    delete this;
    runner([=]() { done_cb(status); });
  }
}

}  // namespace tensorflow