#ifndef TENSORFLOW_CORE_TENSORFLOW_CORE_COMMON_RUNTIME_TENSOR_WAREHOUSE_H_
#define TENSORFLOW_CORE_TENSORFLOW_CORE_COMMON_RUNTIME_TENSOR_WAREHOUSE_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace executor {

// Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
// TODO(yuanbyu): A better way to do "has_value"?
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

class TensorWarehouse {

 public:
  virtual ~TensorWarehouse() = default;

  virtual Entry &GetTensor(std::string &key) = 0;
  virtual void SetTensor(const std::string &key, Entry &value) = 0;
};

} //namespace executor
} //namespace tensorflow

#endif //TENSORFLOW_CORE_TENSORFLOW_CORE_COMMON_RUNTIME_TENSOR_WAREHOUSE_H_
