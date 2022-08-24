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
			Vector<Real, 3> off1 = Vector<Real, 3>(0.0f),
			Vector<Real, 3> off2 = Vector<Real, 3>(0.0f),
			Real minA = 0,
			Real maxA = 0)
		{
			bodyId1 = a;
			bodyId2 = b;
			offset1 = off1;
			offset2 = off2;
			minAngle = minA;
			maxAngle = maxA;
		}

		int bodyId1;
		int bodyId2;

		Vector<Real, 3> offset1;
		Vector<Real, 3> offset2;

		Vector<Real, 3> a1;
		Vector<Real, 3> b1;
		Vector<Real, 3> b2;

		Real angle;
		Real minAngle;
		Real maxAngle;
	};
}