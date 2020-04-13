// ======================================================================== //
// Copyright 2019-2020 Ingo Wald (NVIDIA) and Bin Wang (LSU)                //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}



class cudaTimer
{
private:
  cudaEvent_t t_start;
  cudaEvent_t t_stop;
  cudaStream_t stream;
#define cudaSafeCall(call)                                              \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (cudaSuccess != err)                                             \
      {                                                                 \
        std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ << "): " \
                  << cudaGetErrorString(err);                           \
        exit(EXIT_FAILURE);                                             \
      }                                                                 \
  } while(0)
public:
  cudaTimer()
  {
    cudaSafeCall(cudaEventCreate(&t_start));
    cudaSafeCall(cudaEventCreate(&t_stop));
  }

  ~cudaTimer()
  {
    cudaSafeCall(cudaEventDestroy(t_start));
    cudaSafeCall(cudaEventDestroy(t_stop));
  }

  void start(cudaStream_t st = 0)
  {
    stream = st;
    cudaSafeCall(cudaEventRecord(t_start, stream));
  }

  float stop()
  {
    float milliseconds = 0;
    cudaSafeCall(cudaEventRecord(t_stop, stream));
    cudaSafeCall(cudaEventSynchronize(t_stop));
    cudaSafeCall(cudaEventElapsedTime(&milliseconds, t_start, t_stop));
    return milliseconds;
  }
};
