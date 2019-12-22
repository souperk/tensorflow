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

#include "memory_utils.h"

#include <cstdio>
#include <cstring>

size_t ParseNumberFromPropertyLine(char *property_line) {
  // a pointer to the current character processed
  const char *c = property_line;

  // skip descriptor
  while (*c < '0' || *c > '9') c++;

  //parser number part
  size_t result = 0;
  while ('0' <= *c && *c <= '9') {
    result = result * 10 + (*c - '0');
    c++;
  }

  return result;
}

size_t GetCurrentMemoryUsageInKb() {

  FILE *file = fopen("/proc/self/status", "r");
  size_t result = -1;
  char line[128];

  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "VmSize:", 7) == 0) {
      result = ParseNumberFromPropertyLine(line);
      break;
    }
  }

  fclose(file);
  return result;

}

void tensorflow::executor::MemoryWatch::Update() {
  size_t current_memory = GetCurrentMemoryUsageInKb();

  if (current_memory > max_memory_) {
    max_memory_ = current_memory;
  } else if (current_memory < min_memory_) {
    min_memory_ = current_memory;
  }
}
