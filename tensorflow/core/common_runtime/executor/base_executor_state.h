//
// Created by kostas on 20/10/19.
//

#ifndef TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_BASE_EXECUTOR_STATE_H_
#define TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_BASE_EXECUTOR_STATE_H_

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor/graph_view.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/gtl/stl_util.h"

namespace tensorflow {

struct TaggedNode;
typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;

struct Entry;
typedef gtl::InlinedVector<Entry, 4> EntryVector;

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInNsec() { return Env::Default()->NowNanos(); }

void SetScheduled(NodeExecStatsInterface* stats, int64 micros);

void SetAllStart(NodeExecStatsInterface* stats);

void SetOpStart(NodeExecStatsInterface* stats);

void SetOpEnd(NodeExecStatsInterface* stats);

void SetAllEnd(NodeExecStatsInterface* stats);

void SetOutput(NodeExecStatsInterface* stats, int slot, const Tensor* v);

void SetMemory(NodeExecStatsInterface* stats, OpKernelContext* ctx);

void SetReferencedTensors(NodeExecStatsInterface* stats,
                          const TensorReferenceVector& tensors);

}  // namespace nodestats

struct BaseExecutor;

// Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
// TODO(yuanbyu): A better way to do "has_value"?
struct Entry {
  Entry() {}
  Entry(const Entry& other)
      : ref(other.ref),
        ref_mu(other.ref_mu),
        has_value(other.has_value),
        val_field_is_set(other.val_field_is_set),
        alloc_attr(other.alloc_attr),
        device_context(other.device_context) {
    if (val_field_is_set) {
      val.Init(*other.val);
    }
  }
  ~Entry() {
    if (val_field_is_set) val.Destroy();
  }

  Entry& operator=(const Entry& other) {
    if (val_field_is_set) {
      val.Destroy();
    }
    ref = other.ref;
    ref_mu = other.ref_mu;
    has_value = other.has_value;
    val_field_is_set = other.val_field_is_set;
    alloc_attr = other.alloc_attr;
    device_context = other.device_context;
    if (val_field_is_set) {
      val.Init(*other.val);
    }
    return *this;
  }

  Entry& operator=(Entry&& other) {
    if (val_field_is_set) {
      val.Destroy();
    }
    ref = other.ref;
    ref_mu = other.ref_mu;
    has_value = other.has_value;
    val_field_is_set = other.val_field_is_set;
    alloc_attr = other.alloc_attr;
    device_context = other.device_context;
    if (val_field_is_set) {
      val.Init(std::move(*other.val));
    }
    return *this;
  }

  // Clears the <val> field.
  void ClearVal() {
    if (val_field_is_set) {
      val.Destroy();
      val_field_is_set = false;
      has_value = false;
    }
  }

  // A tensor value, if val_field_is_set.
  ManualConstructor<Tensor> val;

  Tensor* ref = nullptr;    // A tensor reference.
  mutex* ref_mu = nullptr;  // mutex for *ref if ref is not nullptr.

  // Whether the value exists, either in <val> or <ref>.
  bool has_value = false;

  bool val_field_is_set = false;

  // The attributes of the allocator that creates the tensor.
  AllocatorAttributes alloc_attr;

  // Every entry carries an optional DeviceContext containing
  // Device-specific information about how the Tensor was produced.
  DeviceContext* device_context = nullptr;
};

struct IterationState {
  explicit IterationState(const PendingCounts* pending_counts,
                          int total_input_tensors)
      : input_tensors(new Entry[total_input_tensors]),
        outstanding_ops(0),
        outstanding_frame_count(0),
        counts_(*pending_counts) {  // Initialize with copy of *pending_counts
  }

  // The state of an iteration.

  // One copy per iteration. For iteration k, i-th node's j-th input is in
  // input_tensors[k][impl_->nodes[i].input_start + j]. An entry is either
  // a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
  //
  // NOTE: No need to protect input_tensors[i] by any locks because it
  // is resized once. Each element of tensors_ is written once by the
  // source node of an edge and is cleared by the destination of the same
  // edge. The latter node is never run concurrently with the former node.
  Entry* input_tensors;

  // The number of outstanding ops for each iteration.
  size_t outstanding_ops;

  // The number of outstanding frames for each iteration.
  int outstanding_frame_count;
  int pending(PendingCounts::Handle h) { return counts_.pending(h); }
  int decrement_pending(PendingCounts::Handle h, int v) {
    return counts_.decrement_pending(h, v);
  }
  // Mark a merge node as live
  // REQUIRES: Node corresponding to "h" is a merge node
  void mark_live(PendingCounts::Handle h) { counts_.mark_live(h); }
  // Mark a node to show that processing has started.
  void mark_started(PendingCounts::Handle h) { counts_.mark_started(h); }
  // Mark a node to show that processing has completed.
  void mark_completed(PendingCounts::Handle h) { counts_.mark_completed(h); }
  PendingCounts::NodeState node_state(PendingCounts::Handle h) {
    return counts_.node_state(h);
  }

  int dead_count(PendingCounts::Handle h) { return counts_.dead_count(h); }
  void increment_dead_count(PendingCounts::Handle h) {
    counts_.increment_dead_count(h);
  }
  void adjust_for_activation(PendingCounts::Handle h, bool increment_dead,
                             int* pending_result, int* dead_result) {
    counts_.adjust_for_activation(h, increment_dead, pending_result,
                                  dead_result);
  }

  ~IterationState() { delete[] input_tensors; }

 private:
  PendingCounts counts_;
};

struct FrameInfo {
  FrameInfo()
      : input_count(0),
        total_inputs(0),
        pending_counts(nullptr),
        nodes(nullptr) {}

  // The total number of inputs to a frame.
  int input_count;

  // The total number of input tensors of a frame.
  // == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
  int total_inputs;

  // Used to determine the next place to allocate space in the
  // pending_counts data structure we'll eventually construct
  PendingCounts::Layout pending_counts_layout;

  // Each frame has its own PendingCounts only for the nodes in the frame.
  PendingCounts* pending_counts;  // Owned

  // The nodes in a frame. Used only for debugging.
  std::vector<const Node*>* nodes;  // Owned

  ~FrameInfo() {
    delete pending_counts;
    delete nodes;
  }
};

// A new frame is created for each loop. Execution starts at iteration 0.
// When a value at iteration 0 passes through a NextIteration node,
// iteration 1 is created and starts running. Note that iteration 0 may
// still be running so multiple iterations may run in parallel. The
// frame maintains the state of iterations in several data structures
// such as pending_count and input_tensors. When iteration 0 completes,
// we garbage collect the state of iteration 0.
//
// A frame instance is considered "done" and can be garbage collected
// if all its inputs have entered and all its iterations are "done".
//
// A frame manages the live iterations of an iterative computation.
// Iteration i is considered "done" when there are no outstanding ops,
// frames at iteration i are done, all recvs for this iteration are
// completed, and iteration i-1 is done. For iteration 0, we instead
// wait for there to be no more pending inputs of the frame.
//
// Frames and iterations are garbage collected once they are done.
// The state we need to keep around is highly dependent on the
// parallelism enabled by the scheduler. We may want to have the
// scheduler dynamically control the outstanding number of live
// parallel frames and iterations. To reduce the state space, the
// scheduler might want to schedule ops in inner frames first and
// lower iterations first.
//
// This frame state is mostly initialized lazily on demand so we
// don't introduce unnecessary overhead.
struct FrameState {
  explicit FrameState(const BaseExecutor* executor, int parallel_iters)
      : executor(executor),
        max_parallel_iterations(parallel_iters),
        num_outstanding_iterations(1) {}

  // The executor the frame is in.
  const BaseExecutor* executor = nullptr;

  // The name of this frame, which is the concatenation of its parent
  // frame name, the iteration of the parent frame when this frame was
  // created, and the value of the attr 'frame_name'.
  string frame_name;

  // The unique id for this frame. Generated by fingerprinting
  // frame_name.
  uint64 frame_id;

  // The iteration id of its parent frame when this frame is created.
  // -1 if there is no parent frame. The frame_name/parent_iter pair
  // uniquely identifies this FrameState.
  int64 parent_iter = -1;

  // The FrameState of its parent frame.
  FrameState* parent_frame = nullptr;

  // The maximum allowed number of parallel iterations.
  const int max_parallel_iterations;

  // The number of inputs this frame is still waiting.
  int num_pending_inputs = 0;

  // The highest iteration number we have reached so far in this frame.
  int64 iteration_count GUARDED_BY(mu) = 0;

  // The number of outstanding iterations.
  int num_outstanding_iterations GUARDED_BY(mu) = 1;

  // The active iteration states of this frame.
  gtl::InlinedVector<IterationState*, 12> iterations;

  // The NextIteration nodes to enter a new iteration. If the number of
  // outstanding iterations reaches the limit, we will defer the start of
  // the next iteration until the number of outstanding iterations falls
  // below the limit.
  std::vector<std::pair<const Node*, Entry>> next_iter_roots GUARDED_BY(mu);

  // The values of the loop invariants for this loop. They are added into
  // this list as they "enter" the frame. When a loop invariant enters,
  // we make it available to all active iterations. When the frame starts
  // a new iteration, we make all the current loop invariants available
  // to the new iteration.
  std::vector<std::pair<const Node*, Entry>> inv_values GUARDED_BY(mu);

  // The list of dead exit nodes for the current highest iteration. We
  // will only "execute" the dead exits of the final iteration.
  std::vector<const Node*> dead_exits GUARDED_BY(mu);

  // Static information specific to this frame.
  PendingCounts* pending_counts = nullptr;
  int total_input_tensors = 0;
  std::vector<const Node*>* nodes = nullptr;

  // Lock ordering: ExecutorState.mu_ < mu;
  // during structured traversal: parent_frame->mu < mu.
  mutex mu;

  void InitializeFrameInfo(const string& enter_name);

  inline IterationState* GetIteration(int64 iter) EXCLUSIVE_LOCKS_REQUIRED(mu) {
    size_t index = iter % iterations.size();
    return iterations[index];
  }

  inline void SetIteration(int64 iter, IterationState* state)
      EXCLUSIVE_LOCKS_REQUIRED(mu) {
    size_t index = iter % iterations.size();
    DCHECK(state == nullptr || iterations[index] == nullptr);
    iterations[index] = state;
  }

  // Decrement the outstanding op count and clean up the iterations in the
  // frame. Return true iff the execution of the frame is done.
  inline bool DecrementOutstandingOps(const GraphView* gview, int64 iter,
                                      TaggedNodeSeq* ready) {
    mutex_lock l(mu);
    return DecrementOutstandingOpsLocked(gview, iter, ready);
  }

  // Decrement the outstanding op count and clean up the iterations in the
  // frame. Return true iff the execution of the frame is done.
  inline bool DecrementOutstandingOpsLocked(const GraphView* gview, int64 iter,
                                            TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu) {
    IterationState* istate = GetIteration(iter);
    istate->outstanding_ops--;
    if (istate->outstanding_ops != 0) {
      return false;
    } else {
      return CleanupIterations(gview, iter, ready);
    }
  }

  // Returns true if the computation in the frame is completed.
  inline bool IsFrameDone() EXCLUSIVE_LOCKS_REQUIRED(mu) {
    return (num_pending_inputs == 0 && num_outstanding_iterations == 0);
  }

  // Returns true if the iteration of the frame is completed.
  bool IsIterationDone(int64 iter) EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Increments the iteration id. If this is a new iteration, initialize it.
  void IncrementIteration(const GraphView* gview, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Activate all the deferred NextIteration nodes in a new iteration.
  void ActivateNexts(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Activate all the current loop invariants in a new iteration.
  void ActivateLoopInvs(const GraphView* gview, int64 iter,
                        TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Add a new loop invariant and make it available to all active
  // iterations.
  void AddLoopInv(const NodeItem* item, const Entry& value,
                  TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Activate the successors of a node. Contents of *outputs are left in an
  // indeterminate state after returning from this method.
  void ActivateNodes(const NodeItem* item, const bool is_dead, int64 iter,
                     EntryVector* outputs, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Cleanup iterations of this frame starting from iteration iter.
  bool CleanupIterations(const GraphView* gview, int64 iter,
                         TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

  ~FrameState() {
    for (size_t i = 0; i < iterations.size(); ++i) {
      delete iterations[i];
      iterations[i] = nullptr;
    }
  }
};

// A tagged node: <frame*, iter, node*>.
struct TaggedNode {
  const Node* node = nullptr;
  FrameState* input_frame = nullptr;
  int64 input_iter = -1;
  bool is_dead = false;

  TaggedNode(const Node* t_node, FrameState* in_frame, int64 in_iter,
             bool dead) {
    node = t_node;
    input_frame = in_frame;
    input_iter = in_iter;
    is_dead = dead;
  }
};

// A drop-in replacement for std::deque<TaggedNode>.  We typically don't
// have that many nodes in the ready queue, so we just use a vector and
// don't free up memory from the queue as we consume nodes.
class TaggedNodeReadyQueue {
 public:
  TaggedNodeReadyQueue() : front_index_(0) {}

  void push_back(TaggedNode node) { ready_.push_back(node); }
  TaggedNode front() const {
    DCHECK_LT(front_index_, ready_.size());
    return ready_[front_index_];
  }
  void pop_front() {
    DCHECK_LT(front_index_, ready_.size());
    front_index_++;
    if ((front_index_ == ready_.size()) || (front_index_ > 16384)) {
      if (front_index_ == ready_.size()) {
        ready_.clear();
      } else {
        // Lots of unused entries at beginning of vector: move everything
        // down to start of vector.
        ready_.erase(ready_.begin(), ready_.begin() + front_index_);
      }
      front_index_ = 0;
    }
  }
  bool empty() const { return ready_.empty(); }
  const TaggedNode* begin() const { return ready_.begin() + front_index_; }
  const TaggedNode* end() const { return ready_.end(); }

 private:
  gtl::InlinedVector<TaggedNode, 16> ready_;
  int front_index_;
};

class BaseExecutorState {
 public:
  BaseExecutorState(const Executor::Args& args, BaseExecutor* executor);

  void RunAsync(Executor::DoneCallback done);

 protected:

  virtual void Process(TaggedNode tagged_node, int64 scheduled_nsec) = 0;

 protected:
  // Contains a value for [node->id()] for the device context assigned by the
  // device at the beginning of a step.
  DeviceContextMap device_context_map_;

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

  // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
  const bool log_memory_;

  int64 step_id_;

  // Not Owned
  Rendezvous* rendezvous_;
  CollectiveExecutor* collective_executor_ = nullptr;

  StepStatsCollectorInterface* const stats_collector_;
  CancellationManager* cancellation_manager_;

  BaseExecutor* executor_;
  Executor::Args::Runner runner_;

  bool sync_on_finish_;

  // If not null, use this device to schedule intra-op operation
  std::unique_ptr<DeviceBase> user_device_;

  // Owned

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // The root frame in which the execution of this step is started.
  FrameState* root_frame_;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  // Available via OpKernelContext to every OpKernel invocation.
  mutex num_deferred_ops_mu_;
  int64 num_deferred_ops_ GUARDED_BY(num_deferred_ops_mu_) = 0;
  bool finish_when_deferred_ops_done_ GUARDED_BY(num_deferred_ops_mu_) = false;

  mutex mu_;
  Status status_ GUARDED_BY(mu_);

  // Mapping from frame name to outstanding frames. A new frame is created
  // at some iteration of an active frame. So the unique key for the new
  // child frame is composed of the name of the parent frame, the iteration
  // number at which the parent frame is creating the new frame, and the
  // name of the new frame from nodedef.
  gtl::FlatMap<string, FrameState*> outstanding_frames_ GUARDED_BY(mu_);

  // The unique name of a frame.
  inline string MakeFrameName(FrameState* frame, int64 iter_id,
                              const string& name) {
    return strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
  }

  // Find an existing or create a new child frame in the frame 'frame' at
  // iteration 'iter'.
  void FindOrCreateChildFrame(FrameState* frame, int64 iter, const Node* node,
                              FrameState** child);

  // Delete a frame. Called when the frame is done.
  void DeleteFrame(FrameState* frame, TaggedNodeSeq* ready);

  // Cleanup frames and iterations starting from frame/iter. Called when
  // a child frame is done.
  void CleanupFramesIterations(FrameState* frame, int64 iter,
                               TaggedNodeSeq* ready);

  // Before invoking item->kernel, fills in its "inputs".
  virtual Status PrepareInputs(const NodeItem& item, Entry* first_input,
                       TensorValueVec* inputs,
                       DeviceContextVec* input_device_contexts,
                       AllocatorAttributeVec* input_alloc_attrs,
                       bool* is_input_dead);

  // After item->kernel computation is done, processes its outputs.
  virtual Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        EntryVector* outputs, NodeExecStatsInterface* stats);

  // After processing the outputs, propagates the outputs to their dsts.
  // Contents of *outputs are left in an indeterminate state after
  // returning from this method.
  virtual void PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
                        EntryVector* outputs, TaggedNodeSeq* ready);


  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  virtual void ScheduleReady(const TaggedNodeSeq& ready,
                     TaggedNodeReadyQueue* inline_ready);



  // "node" just finishes. Takes ownership of "stats". Returns true if
  // execution has completed.
  virtual bool NodeDone(const Status& s, const Node* node, const TaggedNodeSeq& ready,
                NodeExecStatsInterface* stats,
                TaggedNodeReadyQueue* inline_ready);

  void MaybeMarkCompleted(FrameState* frame, int64 iter, int64 node_id);

  // Provide debugging output about an outstanding node in the executor.
  void DumpPendingNodeState(const int node_id, const Entry* input_vector,
                            bool show_nodes_with_no_ready_inputs);
  void DumpActiveNodeState(const int node_id, const Entry* input_vector);

  // Provide debugging output about an outstanding iteration in the executor.
  void DumpIterationState(const FrameState* frame, IterationState* iteration);

  // Provide debugging output of the state of the executor.
  void DumpState();

  const Tensor* GetTensorValueForDump(const Entry& input);

  // Clean up when this executor is done.
  void Finish();
  void ScheduleFinish();

  // A standalone routine for this expression so that we can express
  // that we don't want thread safety analysis on this reference (it's
  // safe to do without the lock because the iterations array never
  // resizes and this particular iteration's array element will not
  // be changed out from under us because the iteration is still alive).
  Entry* GetInputTensors(FrameState* input_frame,
                         int64 input_iter) const NO_THREAD_SAFETY_ANALYSIS {
    return input_frame->GetIteration(input_iter)->input_tensors;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_BASE_EXECUTOR_STATE_H_
