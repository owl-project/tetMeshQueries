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

#include <owl/owl.h>
#include "tetMeshQueryLib/internalTypes.h"

namespace owl {
  namespace tetQueries {

    using namespace owl;
    using namespace owl::common;
  
    struct PerRayData
    {
      int tetID;
    };
  
    // closest hit for the shared faces method
    OPTIX_CLOSEST_HIT_PROGRAM(sharedFacesCH)()
    {
      PerRayData    &prd
        = owl::getPRD<PerRayData>();
      const SharedFacesGeom &self = getProgramData<SharedFacesGeom>();
      const int   faceID = optixGetPrimitiveIndex();
      const int   tetID
        = optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
        ? self.tetForFace[faceID].front
        : self.tetForFace[faceID].back;
      prd.tetID = tetID;
    }

    OPTIX_MISS_PROGRAM(miss)()
    {
      /* nothing */
    }

    extern "C" __constant__ LaunchParams optixLaunchParams;
  
    OPTIX_RAYGEN_PROGRAM(queryKernel)()
    {
      int particleID
        = getLaunchIndex().x
        + getLaunchDims().x
        * getLaunchIndex().y;

      LaunchParams &lp = optixLaunchParams;

      if (particleID >= lp.numParticles) return;

      vec3f pos;
      if (lp.isFloat) {
        if (!lp.particlesFloat[particleID].isActive) return;
        float4 particle = (const float4&)lp.particlesFloat[particleID];
        pos = vec3f(particle.x,particle.y,particle.z);
      } else {
        if (!lp.particlesDouble[particleID].isActive) return;
        double4 particle = (const double4&)lp.particlesDouble[particleID];
        pos = vec3f(particle.x,particle.y,particle.z);
      }
    
      // create 'dummy' ray from particle pos, with particle itself as per-ray data
      const RayGen &self = getProgramData<RayGen>();
      PerRayData prd;
      prd.tetID = -1;
      owl::Ray ray(pos,
                   vec3f(1.f, 1e-10f, 1e-10f),
                   0.f, self.maxEdgeLength);//1e10f);self.maxEdgeLength);
      owl::traceRay(self.faces,ray,prd,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);
    
      lp.out_tetIDs[particleID] = prd.tetID;
    }

  } // ::owl::tetQueries
} // ::owl
