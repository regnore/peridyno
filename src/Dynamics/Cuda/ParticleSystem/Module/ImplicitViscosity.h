/**
 * Copyright 2017-2021 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "ParticleApproximation.h"

namespace dyno 
{
	/**
	 * @brief This class implements an implicit solver for artificial viscosity based on the XSPH method.
	 */
	template<typename TDataType>
	class ImplicitViscosity : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(ImplicitViscosity, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ImplicitViscosity();
		~ImplicitViscosity() override;
		
	public:
		DEF_VAR(Real, Viscosity, 0.05, "");

		DEF_VAR(int, InterationNumber, 3, "");

		DEF_VAR_IN(Real, TimeStep, "");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

	public:
		void compute() override;

	private:
		DArray<Coord> mVelOld;
		DArray<Coord> mVelBuf;
	};

	IMPLEMENT_TCLASS(ImplicitViscosity, TDataType)
}