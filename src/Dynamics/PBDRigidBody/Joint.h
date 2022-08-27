#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

namespace dyno
{
	template<typename Real>
	class Joint
	{
	public:

		DYN_FUNC Joint(
			int a,
			int b,
			Real angle,
			Vector<Real, 3> ax = Vector<Real, 3>(1.0f, 0.0f, 0.0f),
			Vector<Real, 3> off1 = Vector<Real, 3>(0.0f),
			Vector<Real, 3> off2 = Vector<Real, 3>(0.0f),
			Real minA = 0.0f,
			Real maxA = 0.0f,
			Real alpha = 0.0f)
		{
			this->bodyId1 = a;
			this->bodyId2 = b;
			this->axis = ax;
			this->offset1 = off1;
			this->offset2 = off2;
			this->minAngle = minA;
			this->maxAngle = maxA;
			this->alpha = alpha;
		}

		int bodyId1;
		int bodyId2;

		Vector<Real, 3> offset1;
		Vector<Real, 3> offset2;
		Vector<Real, 3> axis;

		Vector<Real, 3> a1;
		Vector<Real, 3> b1;
		Vector<Real, 3> b2;

		Real angle;
		Real minAngle;
		Real maxAngle;
		Real alpha;
	};
}