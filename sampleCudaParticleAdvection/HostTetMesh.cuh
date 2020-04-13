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

#include "owl/common/math/box.h"
#include <fstream>

namespace advect {

  using namespace owl;
  using namespace owl::common;

  typedef owl::common::box_t<vec3d> box3d;
  
  struct HostTetMesh {
    std::vector<vec3d> positions;
    std::vector<vec3d> velocities;
    std::vector<vec4i> indices;
    box3d worldBounds;

    /*! creates a simple test data set of NxNxN cells with some dummy
        velocity field */
    static HostTetMesh createTestDataSet(int gridSize);

	static HostTetMesh readDataSet(std::string vert_fname, std::string cell_fname, std::string solv_fname="", std::string solc_fname="");
  };

  inline HostTetMesh HostTetMesh::createTestDataSet(int gridSize)
  {
    HostTetMesh model;
    // num vertices along each dim:IO
    int N = gridSize+1;

    vec3f center = vec3f(0.5f*N);
    
    for (int iz=0;iz<N;iz++)
      for (int iy=0;iy<N;iy++)
        for (int ix=0;ix<N;ix++) {
          vec3f pos = vec3f(ix,iy,iz);
          vec3f vel = (pos == center) ? vec3f(1.f,0.f,0.f) : normalize(pos - center);
          model.worldBounds.extend(vec3d(pos));
          model.positions.push_back(vec3d(pos));
          model.velocities.push_back(vec3d(vel));
        }
    for (int iz=0;iz<(N-1);iz++)
      for (int iy=0;iy<(N-1);iy++)
        for (int ix=0;ix<(N-1);ix++) {
          int i0 = ix + N * (iy + N * iz);
          int i1 = i0 + 1;
          int i2 = i0 + N;
          int i3 = i0 + N + 1;

          int i4 = i0 + N*N;
          int i5 = i0 + N*N + 1;
          int i6 = i0 + N*N + N;
          int i7 = i0 + N*N + N + 1;

          model.indices.push_back(vec4i(i0,i1,i3,i5));
          model.indices.push_back(vec4i(i0,i5,i6,i3));
          model.indices.push_back(vec4i(i0,i2,i3,i6));
          model.indices.push_back(vec4i(i0,i4,i5,i6));
          model.indices.push_back(vec4i(i3,i5,i6,i7));
        }
    // fixing winding order of tets:
    for (auto &tet : model.indices) {
      const vec3d &a = model.positions[tet.x];
      const vec3d &b = model.positions[tet.y];
      const vec3d &c = model.positions[tet.z];
      const vec3d &d = model.positions[tet.w];
      if (dot(d-a,cross(b-a,c-a)) < 0.)
        std::swap(tet.z,tet.w);
    }
    
    std::cout << "created test geom w/ " << model.indices.size() << " tets" << std::endl;
	std::cout << "geom bounding box w/ " << model.worldBounds.lower << ", " << model.worldBounds.upper << std::endl;
    return model;
  }

  inline HostTetMesh HostTetMesh::readDataSet(std::string vert_fname, std::string cell_fname, std::string solv_fname, std::string solc_fname)
  {
	  /* Read tets mesh and velocity (vector) field from ascii field

	ASCII file format

	vert.dat
	--------
	NumTetVerts = 16226
	x y z
	-10.0 -17.0 46.0
	-10.0 -17.0 -10.0
	....

	cell.dat
	--------
	NumTetCells = 74430
	id1 id2 id3 id4
	3559 3653 3710 11699
	10154 10852 11785 14048
	...

	solution.dat
	------------
	p u v w
	-0.000518217 -4.31E-17 -8.19E-16 3.38E-17
	-0.000388584 3.77E-17 -3.18E-16 3.95E-17
	...

	*/
	  HostTetMesh model;
	  int NumVerts = 0;
	  int NumTets = 0;

	  std::string word;
	  double number;

	  //Read vertices
	  std::ifstream vfile(vert_fname);
	  if (vfile.is_open()) {
		  vfile >> word >> NumVerts;//Header Line
		  printf("%s %d\n", word.c_str(), NumVerts);
		  vfile >> word >> word >> word;//Comment line

		  model.positions.reserve(NumVerts);
		  for (int i = 0; i < NumVerts; ++i) {
			  vec3d pos;
			  vfile >> pos.x >> pos.y >> pos.z;

			  model.worldBounds.extend(pos);
			  model.positions.push_back(pos);
		  }
		  vfile.close();
	  }

	  //Read Tet indices
	  std::ifstream tfile(cell_fname);
	  if (tfile.is_open()) {
		  tfile >> word >> NumTets;//Header Line
		  printf("%s %d\n", word.c_str(), NumTets);
		  tfile >> word >> word >> word >> word;//Comment line

		  model.indices.reserve(NumTets);
		  for (int i = 0; i < NumTets; ++i) {
			  vec4i tetIDs;
			  tfile >> tetIDs.x >> tetIDs.y >> tetIDs.z >> tetIDs.w;
			  model.indices.push_back(tetIDs);
		  }
		  tfile.close();
	  }
	
	  //Read velocity solutions
	  if (solv_fname.size() > 0) {//Vertx-wise solution
		  std::ifstream sfile(solv_fname);
		  if (sfile.is_open()) {
			  sfile >> word >> word >> word >> word;//Comment line

			  model.velocities.reserve(NumVerts);
			  for (int i = 0; i < NumVerts; ++i) {
				  vec3d vel;
				  sfile >> number >> vel.x >> vel.y >> vel.z; //p u v w
				  if (i < 2) std::cout << "Vert Vel" << i << " " << vel << std::endl;
				  model.velocities.push_back(vel);
			  }
			  sfile.close();
		  }
	  }
	  else {//Cell-wise solution
		  std::ifstream sfile(solc_fname);
		  if (sfile.is_open()) {
			  sfile >> word >> word >> word >> word;//Comment line

			  model.velocities.reserve(NumTets);
			  for (int i = 0; i < NumTets; ++i) {
				  vec3d vel;
				  sfile >> number >> vel.x >> vel.y >> vel.z; //p u v w
				  model.velocities.push_back(vel);
				  if(i<2) std::cout << "Tet Vel" << i << " " << vel<<std::endl;
			  }
			  sfile.close();
		  }
	  }
	  

	  // fixing winding order of tets:
	  for (auto &tet : model.indices) {
		  const vec3d &a = model.positions[tet.x];
		  const vec3d &b = model.positions[tet.y];
		  const vec3d &c = model.positions[tet.z];
		  const vec3d &d = model.positions[tet.w];
		  if (dot(d - a, cross(b - a, c - a)) < 0.)
			  std::swap(tet.z, tet.w);
	  }


	  std::cout << "created test geom w/ " << model.indices.size() << " tets" << std::endl;
	  std::cout << "geom bounding box w/ " << model.worldBounds.lower << ", " << model.worldBounds.upper << std::endl;
	  if (solc_fname.size() > 0) std::cout << "cell-wise uniform velocity enabled!" << std::endl;
	  return model;
  }

}


