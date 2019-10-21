//
// Created by kostas on 20/10/19.
//

#include "tensorflow/core/common_runtime/executor/base_executor.h"

namespace tensorflow {

BaseExecutor::BaseExecutor(const LocalExecutorParams& p,
                           std::unique_ptr<const Graph> g)
    : params_(p), graph_(std::move(g)), gview_() {}

Status BaseExecutor::Initialize() {
  gview_.Initialize(graph_.get());

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();

  // Build the information about frames in this subgraph.
  ControlFlowInfo cf_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph_.get(), &cf_info));

  for (auto& it : cf_info.unique_frame_names) {
    EnsureFrameInfo(it)->nodes = new std::vector<const Node*>;
  }

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();
    const string& frame_name = cf_info.frame_names[id];
    FrameInfo* frame_info = EnsureFrameInfo(frame_name);

    NodeItem* item = gview_.node(id);
    item->node = n;

    item->input_start = frame_info->total_inputs;
    frame_info->total_inputs += n->num_inputs();

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

    // Compute the maximum values we'll store for this node in the
    // pending counts data structure, and allocate a handle in
    // that frame's pending counts data structure that has enough
    // space to store these maximal count values.
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    item->pending_id =
        frame_info->pending_counts_layout.CreateHandle(max_pending, max_dead);

    // Initialize static information about the frames in the graph.
    frame_info->nodes->push_back(n);
    if (IsEnter(n)) {
      string enter_name;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "frame_name", &enter_name));
      EnsureFrameInfo(enter_name)->input_count++;
    }
  }

  // Initialize PendingCounts only after item->pending_id is initialized for
  // all nodes.
  InitializePending(graph_.get(), cf_info);

  return gview_.SetAllocAttrs(graph_.get(), params_.device);
}

void BaseExecutor::InitializePending(const Graph* graph,
                                     const ControlFlowInfo& cf_info) {
  for (auto& it : cf_info.unique_frame_names) {
    FrameInfo* finfo = EnsureFrameInfo(it);
    PendingCounts* counts = new PendingCounts(finfo->pending_counts_layout);
    DCHECK_EQ(finfo->pending_counts, nullptr);
    finfo->pending_counts = counts;
  }

  for (const Node* n : graph->nodes()) {
    const int id = n->id();
    const string& name = cf_info.frame_names[id];
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    const NodeItem* item = gview_.node(id);
    PendingCounts* counts = EnsureFrameInfo(name)->pending_counts;
    counts->set_initial_count(item->pending_id, max_pending);
  }
}

FrameInfo* BaseExecutor::EnsureFrameInfo(const string& fname) {
  auto slot = &frame_info_[fname];
  if (*slot == nullptr) {
    *slot = new FrameInfo;
  }
  return *slot;
}

Status BaseExecutor::BuildControlFlowInfo(const Graph* g,
                                          ControlFlowInfo* cf_info) {
  const int num_nodes = g->num_node_ids();
  cf_info->frame_names.resize(num_nodes);
  std::vector<Node*> parent_nodes;
  parent_nodes.resize(num_nodes);
  std::vector<bool> visited;
  visited.resize(num_nodes);

  string frame_name;
  std::deque<Node*> ready;

  // Initialize with the root nodes.
  for (Node* n : g->nodes()) {
    if (n->in_edges().empty()) {
      visited[n->id()] = true;
      cf_info->unique_frame_names.insert(frame_name);
      ready.push_back(n);
    }
  }

  while (!ready.empty()) {
    Node* curr_node = ready.front();
    int curr_id = curr_node->id();
    ready.pop_front();

    Node* parent = nullptr;
    if (IsEnter(curr_node)) {
      // Enter a child frame.
      TF_RETURN_IF_ERROR(
          GetNodeAttr(curr_node->attrs(), "frame_name", &frame_name));
      parent = curr_node;
    } else if (IsExit(curr_node)) {
      // Exit to the parent frame.
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[parent->id()];
      parent = parent_nodes[parent->id()];
    } else {
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[curr_id];
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      Node* out = out_edge->dst();
      const int out_id = out->id();

      // Add to ready queue if not visited.
      bool is_visited = visited[out_id];
      if (!is_visited) {
        ready.push_back(out);
        visited[out_id] = true;

        // Process the node 'out'.
        cf_info->frame_names[out_id] = frame_name;
        parent_nodes[out_id] = parent;
        cf_info->unique_frame_names.insert(frame_name);
      }
    }
  }

  return Status::OK();
}

void GetMaxPendingCounts(const Node* n, size_t* max_pending,
                         size_t* max_dead_count) {
  const size_t num_in_edges = n->in_edges().size();
  size_t initial_count;
  if (IsMerge(n)) {
    // merge waits all control inputs so we initialize the pending
    // count to be the number of control edges.
    int32 num_control_edges = 0;
    for (const Edge* edge : n->in_edges()) {
      if (edge->IsControlEdge()) {
        num_control_edges++;
      }
    }
    // Use bit 0 to indicate if we are waiting for a ready live data input.
    initial_count = 1 + (num_control_edges << 1);
  } else {
    initial_count = num_in_edges;
  }

  *max_pending = initial_count;
  *max_dead_count = num_in_edges;
}

}  // namespace tensorflow