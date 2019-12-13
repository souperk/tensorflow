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

#include "warehouse.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace executor {

bool AnyReferenceTypes(const EntryVector &vector);

Warehouse::Warehouse(WarehouseStrategy strategy) : strategy_(strategy) {

}

bool Warehouse::Request(int64 node_id, EntryVector &output_vector) {
  mutex_lock lock(mu_);

  auto it = live_entry_map_.find(node_id);

  if (it == live_entry_map_.end()) {
    LOG(INFO) << "item " << node_id << " - not available";
    return false;
  }
  LOG(INFO) << "item " << node_id << " - available";

  WarehouseEntry &warehouse_entry = it->second;
  output_vector.resize(warehouse_entry.size());

  for (size_t i = 0; i < warehouse_entry.size(); i++) {
    output_vector[i] = warehouse_entry.entry(i);
  }

  return true;
}

void Warehouse::Register(int64 node_id, const EntryVector &output_vector) {
  mutex_lock lock(mu_);

  if (node_id == 1028) {
    LOG(INFO) << "1028 AnyReferenceTypes() == " << AnyReferenceTypes(output_vector);
  }

  // a race condition exists where a node may be requested
  // and evaluated by two nodes when
  if (strategy_ == MemoryRich || AnyReferenceTypes(output_vector)) {
    WarehouseEntry &warehouse_entry = live_entry_map_[node_id];
    CHECK(!warehouse_entry.IsInitialized());
    warehouse_entry.Initialize(output_vector);
  }
}

void Warehouse::WarehouseEntry::Initialize(const EntryVector &output_vector) {
  size_ = output_vector.size();
  outputs_ = new Entry[size_];

  // copy entries
  for (size_t i = 0; i < size_; i++) {
    outputs_[i] = output_vector[i];
  }
}

bool AnyReferenceTypes(const EntryVector &vector) {
  for (auto &entry : vector) {
    if (entry.ref != nullptr) {
      return true;
    }
  }
  return false;
}

} //namespace executor
} //namespace tensorflow
