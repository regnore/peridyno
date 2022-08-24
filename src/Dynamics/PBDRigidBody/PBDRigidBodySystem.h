#pragma once
#include "Node.h"
#include "PBDRigidBodyShared.h"
#include "Joint.h"
#include <vector>
#include <iostream>
namespace dyno
{
	/*!
	*	\class	RigidBodySystem
	*	\brief	Implementation of a rigid body system containing a variety of rigid bodies with different shapes.
	*
	*/
	template<typename TDataType>
	class PBDRigidBodySystem : public Node
	{
		DECLARE_TCLASS(PBDRigidBodySystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TSphere3D<Real> Sphere3D;
		typedef typename TOrientedBox3D<Real> Box3D;
		typedef typename Quat<Real> TQuat;

		typedef typename TContactPair<Real> ContactPair;


		PBDRigidBodySystem(std::string name = "RigidBodySystem");
		virtual ~PBDRigidBodySystem();

		void addBox(
			const BoxInfo& box, 
			const RigidBodyInfo& bodyDef,
			const Real density = Real(1));

		void addSphere(
			const SphereInfo& sphere,
			const RigidBodyInfo& bodyDef, 
			const Real density = Real(1));

		void addTet(
			const TetInfo& tet,
			const RigidBodyInfo& bodyDef,
			const Real density = Real(1));

		void addJoint(
			const Joint<Real>& j
		);

	protected:
		void resetStates() override;

		void updateTopology() override;

	public:
		DEF_VAR(bool, FrictionEnabled, true, "A toggle to control the friction");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Real, Mass, DeviceType::GPU, "Mass of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, Center, DeviceType::GPU, "Center of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Velocity of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, AngularVelocity, DeviceType::GPU, "Angular velocity of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Matrix, RotationMatrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Real, DynamicFriction, DeviceType::GPU, "Dynamic friction coefficient of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Real, StaticFriction, DeviceType::GPU, "Static friction coefficient of rigid bodies");

		//DEF_ARRAY_STATE(Joint, Joint, DeviceType::GPU, "Joints");

		DEF_ARRAY_STATE(Matrix, Inertia, DeviceType::GPU, "Inertia matrix");

		DEF_ARRAY_STATE(TQuat, Quaternion, DeviceType::GPU, "Quaternion");

		DEF_ARRAY_STATE(CollisionMask, CollisionMask, DeviceType::GPU, "Collision mask for each rigid body");

		DEF_ARRAY_STATE(Matrix, InitialInertia, DeviceType::GPU, "Initial inertia matrix");

	private:
		std::vector<RigidBodyInfo> mHostRigidBodyStates;

		std::vector<SphereInfo> mHostSpheres;
		std::vector<BoxInfo> mHostBoxes;
		std::vector<TetInfo> mHostTets;

		DArray<RigidBodyInfo> mDeviceRigidBodyStates;

		DArray<SphereInfo> mDeviceSpheres;
		DArray<BoxInfo> mDeviceBoxes;
		DArray<TetInfo> mDeviceTets;

	public:
		int m_numOfSamples;
		DArray2D<Vec3f> m_deviceSamples;
		DArray2D<Vec3f> m_deviceNormals;

		std::vector<Vec3f> samples;
		std::vector<Vec3f> normals;

		int getSamplingPointSize() { return m_numOfSamples; }

		DArray2D<Vec3f> getSamples() { return m_deviceSamples; }
		DArray2D<Vec3f> getNormals() { return m_deviceNormals; }

		void updateVelocityAngule(Vec3f force, Vec3f torque, float dt);
		void advect(float dt);
		//float m_damping = 0.9f;

		float m_yaw;
		float m_pitch;
		float m_roll;
		float m_recoverSpeed;
		void getEulerAngle(float& yaw, float& pitch, float& roll);
	};
}