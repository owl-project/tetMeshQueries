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

#include "tetMeshQueryLib/OptixTetQuery.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

namespace owl {
  namespace tetQueries {

    /*! this is the gpu representation - do not put any std:: etc stuff
      in here.  note we do NOT store the triangle indices/vertices here;
      optix will keep track of them once we contructed them. */
    struct SharedFacesGeom {
      struct FaceInfo { int front=-1, back=-1; };
      FaceInfo *tetForFace;
    };

    struct RayGen {
      OptixTraversableHandle faces;
      float                  maxEdgeLength;
    };
  
    struct LaunchParams {
      union {
        owl::tetQueries::FloatParticle  *particlesFloat;
        owl::tetQueries::DoubleParticle *particlesDouble;
      };
      int    numParticles;
      int    isFloat;
      int    *out_tetIDs;
    };
  
  } // ::owl::tetQueries
} // ::owl
