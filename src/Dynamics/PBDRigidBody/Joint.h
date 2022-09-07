#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"

namespace dyno
{
	enum JointType
	{
		J_Default = 0,
		J_Hinge
	};

	template<typename Real>
	class Joint
	{
	public:
		DYN_FUNC Joint(){}

		DYN_FUNC Joint(
			int a,
			int b,
			Real alpha = 0.0f)
		{
			this->bodyId1 = a;
			this->bodyId2 = b;
			this->alpha = 0.0f;
		}

		void setHinge(
			Vector<Real, 3> off1 = Vector<Real, 3>(0.0f),
			Vector<Real, 3> off2 = Vector<Real, 3>(0.0f),
			Real minA = 0.0f,
			Real maxA = M_PI,
			Real angle = 0.0f
		)
		{
			this->angle = clamp(angle,minA,maxA);
			this->offset1 = off1;
			this->offset2 = off2;
			this->minAngle = minA;
			this->maxAngle = maxA;
			this->jointType = J_Hinge;
		}

		JointType jointType = J_Default;
		int bodyId1 = -1;
		int bodyId2 = -1;
		Real alpha = 0.0f;
		
		Vector<Real, 3> offset1;
		Vector<Real, 3> offset2;
		Vector<Real, 3> axis;

		Real angle;
		Real minAngle;
		Real maxAngle;
	};

	/*template<typename Real>
	class Hinge : public Joint<Real>
	{
	public:
		DYN_FUNC Hinge(
			int a,
			int b,
			Real angle,
			Vector<Real, 3> off1 = Vector<Real, 3>(0.0f),
			Vector<Real, 3> off2 = Vector<Real, 3>(0.0f),
			Real minA = 0.0f,
			Real maxA = M_PI,
			Real alpha = 0.0f)
			: Joint<Real>(a, b)
		{
			this->angle = angle;
			this->offset1 = off1;
			this->offset2 = off2;
			this->minAngle = minA;
			this->maxAngle = maxA;
			this->alpha = alpha;
			this->jointType = J_Hinge;
		}

		Vector<Real, 3> offset1;
		Vector<Real, 3> offset2;
		Vector<Real, 3> axis;

		Real angle;
		Real minAngle;
		Real maxAngle;
		Real alpha;
	};*/
}