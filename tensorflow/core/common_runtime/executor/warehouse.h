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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_WAREHOUSE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_WAREHOUSE_H_

#include <unordered_map>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"

namespace tensorflow {
namespace executor {

enum WarehouseStrategy { MemoryPoor, MemoryRich, Magic };

// TODO(yuanbyu): A better way to do "has_value"?
/**
 * Entry holds an tensor, it guaranteed to hold the tensor (value or reference)
 * until it is disposed. If no other references exist on the tensor it will be
 * disposed.
 */
struct Entry {
  Entry() {}
  Entry(const Entry &other)
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

  Entry &operator=(const Entry &other) {
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

  Entry &operator=(Entry &&other) {
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

  Tensor *ref = nullptr;    // A tensor reference.
  mutex *ref_mu = nullptr;  // mutex for *ref if ref is not nullptr.

  // Whether the value exists, either in <val> or <ref>.
  bool has_value = false;

  bool val_field_is_set = false;

  // The attributes of the allocator that creates the tensor.
  AllocatorAttributes alloc_attr;

  // Every entry carries an optional DeviceContext containing
  // Device-specific information about how the Tensor was produced.
  DeviceContext *device_context = nullptr;
};

typedef gtl::InlinedVector<Entry, 4> EntryVector;

class Warehouse {

 public:
  explicit Warehouse(WarehouseStrategy strategy);

  bool Request(int64 node_id, EntryVector& outputs_vector);
  void Register(int64 node_id, const EntryVector &outputs_vector);

 private:
  WarehouseStrategy strategy_;

  mutex mu_;
  class WarehouseEntry;
  std::map<int64, WarehouseEntry> live_entry_map_ GUARDED_BY(mu_);

  // TODO(souperk) maybe rename this WarehouseItem
  class WarehouseEntry {

   public:
    WarehouseEntry() : outputs_(nullptr), size_(0) {

    }

    ~WarehouseEntry() {
      delete[] outputs_;
    }

   public:

    bool IsInitialized() const { return size_ > 0; }
    const Entry *outputs() const { return outputs_; }
    const Entry& entry(size_t index) const { return outputs_[index]; }
    size_t size() const { return size_; }

    void Initialize(const EntryVector &outputs_vector);

   private:
    Entry *outputs_;
    size_t size_;
  };

};

} // namespace executor
} // namespace tensorflow
#endif //TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_WAREHOUSE_H_
