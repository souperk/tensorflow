//
// Created by kostas on 20/10/19.
//

#ifndef TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_ON_DEMAND_EXECUTOR_H_
#define TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_ON_DEMAND_EXECUTOR_H_

#include "tensorflow/core/common_runtime/executor.h"

namespace tensorflow{

Status NewOnDemandExecutor(const LocalExecutorParams& params,
                        std::unique_ptr<const Graph> graph,
                        Executor** executor);

}

#endif  // TENSORFLOW_TENSORFLOW_CORE_COMMON_RUNTIME_EXECUTOR_ON_DEMAND_EXECUTOR_H_


