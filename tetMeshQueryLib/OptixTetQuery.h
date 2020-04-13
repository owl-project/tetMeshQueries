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

#include <vector>
#include <map>
#include <owl/owl.h>
#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"
#include "owl/owl.h"

namespace owl {
  namespace tetQueries {

    using namespace owl::common;

    /*! specifies one query's input/output data, with single precision
      query position.  Queries will only perofrmed for partiles whose
      isActive is not True, so marking particles isActive will make it
      skipped during query. After the query, out_tetID will be set to
      the tet that the particle is in, or -1, if not within any tet (if
      particle is invalid/inactive, out_tetID remains unchanged)
    */
    struct FloatParticle {
      float pos[3];
      /*! if the query is valid (ie, isActive is True) then this
        value will, after the query, be either the ID of the tet
        that 'pos' is in, or -1 if not within any tet */
      int isActive;
    };

    /* specifies one querie's input/output data, with a double-precision
       query position (the query itself will use single-precision float,
       though). Invalid particles (ie, particles in an array for which
       _no_ query should be performed) can be specified by setting
       isActive to False */
    struct DoubleParticle {
      double pos[3];
      /*! if 0, this particle will not be done */
      int isActive;
      int pad_align;
    };

    /*! class that builds a 'shared faces' accel on a given tet mesh,
      and performs (batch-style) point location queries within
      this. "batch" in that sense means you cannot directly call this
      from a cuda kernel, but will have to architect your program in a
      way that the simulatoin coe generates a (device-side) array of
      queries (in CUDA), then calls this kernel (on the host) to do
      all those queries, and then goes back to processing the results
      in a cuda kernel */
    struct OptixTetQuery { 
      OptixTetQuery(const vec3d *vertex, int numVertices,
                    const vec4i *index,  int numIndices);
    
      /*! perform a _synchronous_ query with given device-side array of
        particle */
      void query_sync(FloatParticle  *d_particles, int* out_tetIDs, int numParticles);
      void query_sync(DoubleParticle *d_particles, int* out_tetIDs, int numParticles);

      // NOT IMPLEMENTED YET:
#ifdef HAVE_ASYNC_QUERIES
      /*! if you do want to run queries asynchronously from multiple
        threads, then every thread running queries must have its _own_
        asynch query context */
      struct AsyncQueryContext {
      
        /*! launches a query in the given stream; this will *not*
          synchronize at the end of this call, so you may
          asynchrnously launch another cuda kernel that depends on
          this result into the given this->stream without blocking
          other kernels */
        void query(Particle *d_particles, int numParticles);
      
        /*! the cuda stream that the query will be launched into */
        cudaStream_t    stream;
      private:
        OWLLaunchParams launchParams;
      
        /*! for sanity checking, to make sure the user doesn't
          accidentally run the same context with multiple threads
          after all ... */
        std::mutex      mutex;
      };

      /*! create a new async query context; if you do want to
        asynchronously run quries from multiple host threads in
        parallel, you _have_ to have a separte context per thread */
      AsyncQueryContext *createAsyncQueryContext();
#endif
    
    private:
      OWLGroup   faceBVH = 0;
      OWLContext owl     = 0;
      OWLModule  module  = 0;
      OWLRayGen  rayGen  = 0;
      OWLLaunchParams launchParams = 0;
    };

  } // ::owl::tetQueries
} // ::owl
