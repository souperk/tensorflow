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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_NODE_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_NODE_STATE_H_

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace executor {

enum NodeExecutionState {
  None, Ready, Available, Dead
};

class NodeState {

 public:
  NodeState() = default;
  ~NodeState() = default;

  bool IsDead() const { return state_ == Dead; }
  bool IsDemanded() const { return demands_ > 0; }

  void SetReady() {
    state_ = Ready;
  }

  void OnComputed(bool is_dead) {
    DCHECK(state_ == Ready);

    if (is_dead) {
      state_ = Dead;
    } else {
      state_ = Available;
    }
  }

  void Kill() {
    pending_ = 0;
    state_ = Dead;
  }

  void Discard() {
    if (state_ == Dead) {
      // dead nodes can't become available,
      // do not change it's state
      return;
    }

    state_ = None;
  }

  void AddDependency() {
    pending_++;
  }

  void OnDependencyFinished(bool is_dead) {
    DCHECK(pending_ > 0);

    if (is_dead) {
      Kill();
    } else {
      pending_--;

      if (pending_ == 0) {
        state_ = Ready;
      }
    }
  }

  void AddDemand() {
    demands_++;
  }

  void RemoveDemand() {
    DCHECK(demands_ > 0);
    demands_--;
  }

  NodeExecutionState execution_state() const { return state_; }
  int demands() const { return demands_; }
  int pending() const { return pending_; }

 private:
  NodeExecutionState state_ = NodeExecutionState::None;
  int demands_ = 0;
  int pending_ = 0;
};

} // namespace executor
} // namespace tensorflow

#endif //TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_NODE_STATE_H_
