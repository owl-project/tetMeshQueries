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

#include "HostTetMesh.cuh"
#include "cudaHelpers.cuh"

namespace advect {
  
  struct DeviceTetMesh {

    void upload(const HostTetMesh &hostTetMesh);
    
    vec3d *d_positions  { nullptr };
    vec3d *d_velocities { nullptr };
    vec4i *d_indices    { nullptr };
    box3d worldBounds;
  };

  template<typename T>
  void allocAndUpload(T *&d_data,
                      const std::vector<T> &data)
  {
    assert(d_data == nullptr);
    cudaCheck(cudaMalloc(&d_data,
                         data.size()*sizeof(data[0])));
    cudaCheck(cudaMemcpy(d_data,
                         data.data(),
                         data.size()*sizeof(data[0]),
                         cudaMemcpyHostToDevice));
  }
  
  inline void DeviceTetMesh::upload(const HostTetMesh &mesh)
  {
    allocAndUpload(d_positions,mesh.positions);
    allocAndUpload(d_velocities,mesh.velocities);
    allocAndUpload(d_indices,mesh.indices);
    worldBounds = mesh.worldBounds;
  }
  
}
