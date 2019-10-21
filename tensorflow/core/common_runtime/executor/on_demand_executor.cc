//
// Created by kostas on 20/10/19.
//

#include "tensorflow/core/common_runtime/executor/on_demand_executor.h"

#include "tensorflow/core/common_runtime/executor/base_executor.h"
#include "tensorflow/core/common_runtime/executor/base_executor_state.h"
#include "tensorflow/core/common_runtime/executor/graph_view.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

namespace {

class OnDemandExecutorState : public BaseExecutorState {
 public:
  OnDemandExecutorState(const Executor::Args& args, BaseExecutor* executor)
      : BaseExecutorState(args, executor) {}

 protected:
  void PropagateOutputs(const TaggedNode& tagged_node,
                                               const NodeItem* item,
                                               EntryVector* outputs,
                                               TaggedNodeSeq* ready) override;

  void Process(TaggedNode tagged_node, int64 scheduled_nsec) override;
};

class OnDemandExecutor : public BaseExecutor {
  friend class OnDemandExecutorState;

 public:
  OnDemandExecutor(const LocalExecutorParams& params,
                   std::unique_ptr<const Graph> graph)
      : BaseExecutor(params, std::move(graph)) {}

  Status Initialize() override;

  void RunAsync(const Args& args, DoneCallback done) override;

 private:
};

void OnDemandExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
                                             const NodeItem* item,
                                             EntryVector* outputs,
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
        &((OnDemandExecutor*)executor_)->gview_, input_iter, ready);
  } else if (item->is_exit) {
    FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
    output_iter = 0;
    {
      const NodeItem* item =
          ((OnDemandExecutor*)executor_)->gview_.node(node->id());
      mutex_lock l(output_frame->mu);

      if (item->is_constant_enter) {
        // Propagate to all active iterations if this is a loop invariant.
        output_frame->AddLoopInv(item, (*outputs)[0], ready);
      } else {
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }

      output_frame->num_pending_inputs--;
    }
    is_frame_done = input_frame->DecrementOutstandingOps(
        &((OnDemandExecutor*)executor_)->gview_, input_iter, ready);
  } else if (item->is_enter) {
    if (is_dead) {
      mutex_lock l(input_frame->mu);
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      is_frame_done = input_frame->DecrementOutstandingOpsLocked(
          &((OnDemandExecutor*)executor_)->gview_, input_iter, ready);
    } else {
      output_frame = input_frame->parent_frame;
      output_iter = input_frame->parent_iter;
      {
        mutex_lock l(output_frame->mu);
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }
      is_frame_done = input_frame->DecrementOutstandingOps(
          &((OnDemandExecutor*)executor_)->gview_, input_iter, ready);
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
          input_frame->IncrementIteration(&((OnDemandExecutor*)executor_)->gview_,
                                          ready);
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
        &((OnDemandExecutor*)executor_)->gview_, input_iter, ready);
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

void OnDemandExecutorState::Process(TaggedNode tagged_node,
                                    int64 scheduled_nsec) {}


Status OnDemandExecutor::Initialize() {
  Status s = BaseExecutor::Initialize();

  if (!s.ok()) {
    return s;
  }

  for (const Node* n : graph_->nodes()) {
    // TODO(souperk) should I execute first 1) nodes without output edges or
    //  2) sink nodes ?

    if (n->out_edges().empty()) {
      root_nodes_.push_back(n);
    }
  }

  return Status::OK();
}

void OnDemandExecutor::RunAsync(const Executor::Args& args,
                                Executor::DoneCallback done) {
  // TODO(kalex) implement this
  (new OnDemandExecutorState(args, (BaseExecutor*)this))
      ->RunAsync(std::move(done));
}

}  // namespace

Status NewOnDemandExecutor(const LocalExecutorParams& params,
                           std::unique_ptr<const Graph> graph,
                           Executor** executor) {
  OnDemandExecutor* created_executor =
      new OnDemandExecutor(params, std::move(graph));
  const Status s = created_executor->Initialize();

  if (s.ok()) {
    *executor = created_executor;
  } else {
    delete created_executor;
  }

  return s;
}

namespace {

class OnDemandExecutorRegistrar {
 public:
  OnDemandExecutorRegistrar() {
    Factory* factory = new Factory;
    ExecutorFactory::Register("ON_DEMAND", factory);
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params,
                       std::unique_ptr<const Graph> graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret = nullptr;
      TF_RETURN_IF_ERROR(NewOnDemandExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static OnDemandExecutorRegistrar registrar;

}  // namespace

}  // namespace tensorflow