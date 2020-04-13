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

#include "Model.h"
#include "cuda_timer.cuh"
#include "IO.cuh"

namespace advect {
  
  void cudaInitParticles(Particle *d_particles,
                         int numParticles,
                         const box3d &bounds);
  void cudaParticleAdvect(Particle *d_particles,
                          int numParticles,
                          vec4i    *d_tetIndices,
                          vec3d    *d_vertexPositions,
                          vec3d    *d_vertexVelocities,
                          double    dt);
  
  OptixModel *createTestModel(int N)
  {
    //! per tet vertex indices
    std::vector<vec4i>  index;
    //! vertex positions
    std::vector<vec3fa> vertex;
    //! per vertex velocity
    std::vector<vec3fa> velocity;

    testModel(N,index,vertex,velocity);

    // create optix model of the tet mesh, with inside facing
    // triangles to represent tets, with BVH to accel, and all the
    // index/vertex/velocity data uploaded into SBT as a "TetMesh"
    // struct
    OptixModel *model = new OptixModel(index,vertex,velocity);
    return model;
  }

  extern "C" int main(int ac, char **av)
  {
    cudaTimer timer;
    timer.Start();

    int numParticles = 10000;
    int testModelGridSize = 1;
    int numSteps = 1000;
    double dt = 1e-6f;
    
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "--num-particles")
        numParticles = std::atoi(av[++i]);
      else if (arg == "--num-steps")
        numSteps = std::atoi(av[++i]);
      else if (arg == "-dt")
        dt = std::atof(av[++i]);
      else if (arg == "--test-grid-size")
        testModelGridSize = std::atoi(av[++i]);
      else
        throw std::runtime_error("unknown cmdline argument '"+arg+"'");
    }

    // ------------------------------------------------------------------
    // create a host-side model - will eventually use an importer
    // ------------------------------------------------------------------
    HostModel hostModel = HostModel::createTestModel(testModelGridSize);

    // ------------------------------------------------------------------
    // build the query accelerator first, before the cuda kernels
    // allocate their memory.
    // ------------------------------------------------------------------
    OptixTetQuery tetQueryAccelerator(hostModel.positions.data(),
                                      hostModel.positions.size(),
                                      hostModel.indices.data(),
                                      hostModel.indices.size());
    // by now optix should have built all its data,and released
    // whatever temp memory it has used.


    // ------------------------------------------------------------------
    // upload our own cuda data
    // ------------------------------------------------------------------
    DeviceModel deviceTetMesh;
    deviceTetMesh.upload(hostModel);

    // ------------------------------------------------------------------
    // create the device-side particle buffer we're operating on.
    // ------------------------------------------------------------------
    Particle *d_particles = nullptr;
    cudaCheck(cudaMalloc(&d_particles,numParticles*sizeof(Particle)));

    // ------------------------------------------------------------------
    // now run sample advection...
    // ------------------------------------------------------------------
    
    // initialize
    cudaInitParticles(d_particles,numParticles,
                      hostModel.worldBounds);

    // and iterate
    for (int i=0;i<numSteps;i++) {
      // first, compute each particle's current tet
      tetQueryAccelerator.query(d_particles,numParticles);

      // ... and advect
      cudaParticleAdvect(Particle *d_particles,
                         int numParticles,
                         vec4i    *d_tetIndices,
                          vec3d    *d_vertexPositions,
                         vec3d    *d_vertexVelocities,
                         double    dt)

      
    }
   

    
    
    printf("Init RunTime=%lf ms\n", timer.Stop());


    timer.Start();
    // now, we have a buffer of storage for particles that the query
    // kernel can operate on
    for (int i=0;i<numSteps;i++) {
      // std::cout << "--------------------------------------------" << std::endl;
      //std::cout << "step #" << i << std::endl;
      int launchWidth = 64*1024;
      int launchHeight = divRoundUp(maxNumParticles,launchWidth);

      // for now, we'll simply update the position in the raygen prog
      // itself...
      owlRayGenLaunch2D(rayGen,launchWidth,launchHeight);
      // std::cout << "done step #" << i << std::endl;

      //Export particles
      if(i % 10 ==0) writeParticles2VTU(i, d_particles, maxNumParticles);
    }
    std::cout << "#adv: advection steps = " << numSteps << std::endl;
    std::cout << "done ... ignoring proper cleanup for now" << std::endl;

    printf("Particle advection RunTime=%lf ms\n", timer.Stop());
  }
  
}
 
