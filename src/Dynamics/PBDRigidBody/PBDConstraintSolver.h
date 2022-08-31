#pragma once
#include "Module/ConstraintModule.h"
#include "PBDRigidBodyShared.h"
#include "Joint.h"

namespace dyno
{
	template<typename TDataType>
	class PBDConstraintSolver : public ConstraintModule
	{
		DECLARE_TCLASS(PBDConstraintSolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		
		typedef typename Quat<Real> TQuat;
		typedef typename TContactPair<Real> ContactPair;
		typedef typename Joint<Real> Joint;
		
		PBDConstraintSolver();
		~PBDConstraintSolver();

		void constrain() override;

	public:
		DEF_VAR(bool, DynamicFrictionEnabled, true, "");

		DEF_VAR(bool, StaticFrictionEnabled, false , "");

		DEF_VAR(Real, RestituteCoef, 0.5f, "");

		DEF_VAR(uint, IterationNumber, 1, "");

		DEF_VAR(uint, NumSubsteps, 1, "");

	public:
		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_ARRAY_IN(Real, Mass, DeviceType::GPU, "Mass of rigid bodies");

		DEF_ARRAY_IN(Coord, Center, DeviceType::GPU, "Center of rigid bodies");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Velocity of rigid bodies");

		DEF_ARRAY_IN(Coord, AngularVelocity, DeviceType::GPU, "Angular velocity of rigid bodies");

		DEF_ARRAY_IN(Matrix, RotationMatrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DEF_ARRAY_IN(Matrix, Inertia, DeviceType::GPU, "Interial matrix");

		DEF_ARRAY_IN(Matrix, InitialInertia, DeviceType::GPU, "Interial matrix");

		DEF_ARRAY_IN(TQuat, Quaternion, DeviceType::GPU, "Quaternion");

		DEF_ARRAY_IN(Real, DynamicFriction, DeviceType::GPU, "Dynamic Friction Coefficient");

		DEF_ARRAY_IN(Real, StaticFriction, DeviceType::GPU, "Static Friction Coefficient");

		DEF_ARRAY_IN(ContactPair, Contacts, DeviceType::GPU, "");

		DEF_ARRAY_IN(Joint, Joint, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, A, DeviceType::GPU, "");

	private:
		void initializeJacobian(Real dt);

		void initialize();

	private:
		DArray<Real> mLambdaN;	//contact impulse
		DArray<Real> mLambdaT;
		DArray<Real> mLambdaJ;
		DArray<Real> mLambdaJA;

		DArray<Real> mContactNumber;
		DArray<Real> mJointNumber;

		DArray<Matrix> mInitialInertia;

		DArray<ContactPair> mAllConstraints;

		DArray<Coord> x_prev;
		DArray<TQuat> q_prev;
		DArray<Coord> v_prev;
		DArray<Coord> w_prev;

		DArray<Real> mAlpha;
	};
}