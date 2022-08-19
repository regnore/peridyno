#include "PBDIterativeConstraintSolver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(PBDIterativeConstraintSolver, TDataType)

	template<typename TDataType>
	PBDIterativeConstraintSolver<TDataType>::PBDIterativeConstraintSolver()
		: ConstraintModule()
	{
		this->inContacts()->tagOptional(true);
	}

	template<typename TDataType>
	PBDIterativeConstraintSolver<TDataType>::~PBDIterativeConstraintSolver()
	{
	}

	template <typename Coord, typename ContactPair>
	__global__ void TakeOneJacobiIteration(
		DArray<Real> lambda,
		DArray<Coord> accel,
		DArray<Real> d,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Real> mass,
		DArray<ContactPair> nbq,
		DArray<Real> stepInv)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].bodyId1;
		int idx2 = nbq[pId].bodyId2;

		Real eta_i = eta[pId];
		{
			eta_i -= J[4 * pId].dot(accel[idx1 * 2]);
			eta_i -= J[4 * pId + 1].dot(accel[idx1 * 2 + 1]);
			if (idx2 != -1)
			{
				eta_i -= J[4 * pId + 2].dot(accel[idx2 * 2]);
				eta_i -= J[4 * pId + 3].dot(accel[idx2 * 2 + 1]);
			}
		}

		if (d[pId] > EPSILON)
		{
			Real delta_lambda = eta_i / d[pId];
			Real stepInverse = stepInv[idx1];
			if (idx2 != -1)
				stepInverse += stepInv[idx2];
			delta_lambda *= (1.0f / stepInverse);

			//printf("delta_lambda = %.3lf\n", delta_lambda);

			if (nbq[pId].contactType == ContactType::CT_NONPENETRATION || nbq[pId].contactType == ContactType::CT_BOUDNARY) //	PROJECTION!!!!
			{
				Real lambda_new = lambda[pId] + delta_lambda;
				if (lambda_new < 0) lambda_new = 0;

				Real mass_i = mass[idx1];
				if (idx2 != -1)
					mass_i += mass[idx2];

				if (lambda_new > 25 * (mass_i / 0.1)) lambda_new = 25 * (mass_i / 0.1);
				delta_lambda = lambda_new - lambda[pId];
			}

			if (nbq[pId].contactType == ContactType::CT_FRICTION) //	PROJECTION!!!!
			{
				Real lambda_new = lambda[pId] + delta_lambda;
				Real mass_i = mass[idx1];
				if (idx2 != -1)
					mass_i += mass[idx2];

				//if ((lambda_new) > 5 * (mass_i)) lambda_new = 5 * (mass_i);
				//if ((lambda_new) < -5 * (mass_i)) lambda_new = -5 * (mass_i);
				delta_lambda = lambda_new - lambda[pId];
			}

			lambda[pId] += delta_lambda;

			//printf("inside iteration: %d %d %.5lf   %.5lf\n", idx1, idx2, nbq[pId].s4, delta_lambda);

			atomicAdd(&accel[idx1 * 2][0], B[4 * pId][0] * delta_lambda);
			atomicAdd(&accel[idx1 * 2][1], B[4 * pId][1] * delta_lambda);
			atomicAdd(&accel[idx1 * 2][2], B[4 * pId][2] * delta_lambda);

			atomicAdd(&accel[idx1 * 2 + 1][0], B[4 * pId + 1][0] * delta_lambda);
			atomicAdd(&accel[idx1 * 2 + 1][1], B[4 * pId + 1][1] * delta_lambda);
			atomicAdd(&accel[idx1 * 2 + 1][2], B[4 * pId + 1][2] * delta_lambda);

			if (idx2 != -1)
			{
				atomicAdd(&accel[idx2 * 2][0], B[4 * pId + 2][0] * delta_lambda);
				atomicAdd(&accel[idx2 * 2][1], B[4 * pId + 2][1] * delta_lambda);
				atomicAdd(&accel[idx2 * 2][2], B[4 * pId + 2][2] * delta_lambda);

				atomicAdd(&accel[idx2 * 2 + 1][0], B[4 * pId + 3][0] * delta_lambda);
				atomicAdd(&accel[idx2 * 2 + 1][1], B[4 * pId + 3][1] * delta_lambda);
				atomicAdd(&accel[idx2 * 2 + 1][2], B[4 * pId + 3][2] * delta_lambda);
			}
		}
	}

	template <typename Coord>
	__global__ void RB_UpdateVelocity(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> accel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= accel.size() / 2) return;

		velocity[pId] += accel[2 * pId] * dt;
		velocity[pId] += Coord(0, -9.8f, 0) * dt;

		angular_velocity[pId] += accel[2 * pId + 1] * dt;
	}

	template <typename Coord, typename Matrix, typename Quat>
	__global__ void RB_UpdateGesture(
		DArray<Coord> pos,
		DArray<Quat> rotQuat,
		DArray<Matrix> rotMat,
		DArray<Matrix> inertia,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Matrix> inertia_init,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] += velocity[pId] * dt;

		rotQuat[pId] = rotQuat[pId].normalize();
		rotMat[pId] = rotQuat[pId].toMatrix3x3();

		rotQuat[pId] += dt * 0.5f *
			Quat(angular_velocity[pId][0], angular_velocity[pId][1], angular_velocity[pId][2], 0.0)
			*(rotQuat[pId]);

		inertia[pId] = rotMat[pId] * inertia_init[pId] * rotMat[pId].inverse();
		//inertia[pId] = rotMat[pId] * rotMat[pId].inverse();
	}

	template <typename Coord, typename Real>
	__global__ void PBDRB_UpdateXV(
		DArray<Coord> x,
		DArray<Coord> x_prev,
		DArray<Coord> v,
		DArray<Real> mass,
		Coord a_ext,
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= x.size()) return;

		x_prev[pId] = x[pId];
		v[pId] += h * a_ext;
		x[pId] += v[pId] * h;
	}

	template <typename Coord, typename Matrix, typename Quat,typename Real>
	__global__ void PBDRB_UpdateQW(
		DArray<Quat> q,
		DArray<Quat> q_prev,
		DArray<Matrix> R,
		DArray<Matrix> I,
		DArray<Coord> w,
		DArray<Matrix> I_init,
		Coord tau_ext,
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= q.size()) return;

		q_prev[pId] = q[pId];
		w[pId] += h * I[pId].inverse() * (tau_ext - (w[pId].cross(I[pId] * w[pId])));
		q[pId] += h * 0.5f * (Quat(w[pId][0], w[pId][1], w[pId][2], 0.0)* q[pId]);
		q[pId] = q[pId].normalize();

		R[pId] = q[pId].toMatrix3x3();
		I[pId] = R[pId] * I_init[pId] * R[pId].inverse();
	}

	template <typename Coord, typename Quat, typename Real>
	__global__ void PBDRB_CalcVW(
		DArray<Coord> x,
		DArray<Coord> x_prev,
		DArray<Coord> v,
		DArray<Quat> q,
		DArray<Quat> q_prev,
		DArray<Coord> w,
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= x.size()) return;

		v[pId] = (x[pId] - x_prev[pId]) / h;

		Quat dq = q[pId] * (q_prev[pId].inverse());
		w[pId] = (dq.w >=0 ? 1:-1) * 2 * Coord(dq.x , dq.y , dq.z) / h;
	}

	template <typename Coord, typename Quat, typename Matrix, typename Real, typename ContactPair>
	__global__ void PBDRB_SolvePositions(
		DArray<Coord> x,
		DArray<Quat> q,
		DArray<Matrix> R,
		DArray<Matrix>I_init,
		DArray<Real> m,
		DArray<Matrix> I,
		DArray<Real> lambda,
		DArray<Real> alpha,
		DArray<ContactPair> nbq,
		DArray<Real> stepInv,
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbq.size()) return;

		int idx1 = nbq[pId].bodyId1;
		int idx2 = nbq[pId].bodyId2;
		Coord n = -nbq[pId].normal1;
		n /= n.norm();
		Real c = nbq[pId].interpenetration;
		Real tildeAlpha = alpha[pId] / h / h;
			if (idx2 != -1)
			{
				Coord r1 = nbq[pId].pos1 - x[idx1];
				Coord r2 = nbq[pId].pos1 - x[idx2];
				Coord temp3 = r1.cross(n);
				Coord temp4 = r2.cross(n);

				Real w1 = 1.0f / m[idx1] + (Real)(temp3.dot(I[idx1].inverse() * temp3));
				Real w2 = 1.0f / m[idx2] + (Real)(temp4.dot(I[idx2].inverse() * temp4));

				Real dLambda = ((-c - tildeAlpha * lambda[pId]) / (w1 + w2 + tildeAlpha));
				dLambda *= 1 / (stepInv[idx1] + stepInv[idx2]);
				lambda[pId] += dLambda;

				Coord p = dLambda * n;
				x[idx1] += p / m[idx1];
				x[idx2] -= p / m[idx2];
				Coord temp1 = I[idx1].inverse() * (r1.cross(p));
				Coord temp2 = I[idx2].inverse() * (r2.cross(p));
				q[idx1] += Quat(temp1[0], temp1[1], temp1[2], 0) * q[idx1] * 0.5f;
				q[idx2] -= Quat(temp2[0], temp2[1], temp2[2], 0) * q[idx2] * 0.5f;

				R[idx1] = q[idx1].toMatrix3x3();
				I[idx1] = R[idx1] * I_init[idx1] * R[idx1].inverse();

				R[idx2] = q[idx2].toMatrix3x3();
				I[idx2] = R[idx2] * I_init[idx2] * R[idx2].inverse();
			}
			else
			{
				Coord r1 = nbq[pId].pos1 - x[idx1];
				Coord temp3 = r1.cross(n);

				Real w1 = 1.0f / m[idx1] + (Real)(temp3.dot(I[idx1].inverse() * temp3));
				Real w2 = 0.0f;

				Real dLambda = ((-c - tildeAlpha * lambda[pId]) / (w1 + w2 + tildeAlpha));
				dLambda /= stepInv[idx1];
				lambda[pId] += dLambda;

				Coord p = dLambda * n;
				x[idx1] += p / m[idx1];
				Coord temp1 = I[idx1].inverse() * (r1.cross(p));
				q[idx1] += Quat(temp1[0], temp1[1], temp1[2], 0) * q[idx1] * 0.5f;

				R[idx1] = q[idx1].toMatrix3x3();
				I[idx1] = R[idx1] * I_init[idx1] * R[idx1].inverse();
			}
	}

	template<typename TDataType>
	void PBDIterativeConstraintSolver<TDataType>::constrain()
	{
		uint num = this->inCenter()->size();
		Real dt = this->inTimeStep()->getData();
		uint numSubsteps = this->varNumSubsteps()->getData();
		Real h = dt / numSubsteps;

		if (this->x_prev.size() == 0)
			this->x_prev.resize(num);

		if (this->q_prev.size() == 0)
			this->q_prev.resize(num);

		Coord g = Coord(0.0f, -9.8f, 0.0f);
		Coord tau = Coord(0.0f, 0.0f, 0.0f);

		for (int ii = 0; ii < numSubsteps; ii++)
		{
			////construct j
			//if (!this->inContacts()->isEmpty())
			//{
			//	initializeJacobian(dt);

			//	int size_constraints = mAllConstraints.size();
				/*for (int i = 0; i < this->varIterationNumber()->getData(); i++)
				{
					cuExecute(size_constraints,
						TakeOneJacobiIteration,
						mLambda,
						mAccel,
						mD,
						mJ,
						mB,
						mEta,
						this->inMass()->getData(),
						mAllConstraints,
						mContactNumber);
				}
			}*/

			cuExecute(num,
				PBDRB_UpdateXV,
				this->inCenter()->getData(),
				x_prev,
				this->inVelocity()->getData(),
				this->inMass()->getData(),
				g,
				h);

			cuExecute(num,
				PBDRB_UpdateQW,
				this->inQuaternion()->getData(),
				q_prev,
				this->inRotationMatrix()->getData(),
				this->inInertia()->getData(),
				this->inAngularVelocity()->getData(),
				this->inInitialInertia()->getData(),
				tau,
				h);

			uint numC = 0;
			if (this->inContacts()->size() > 0)
			{
				this->initialize();
				numC = this->inContacts()->size();
			}

			if (numC > 0)
			{
				for (int i = 0 ; i < this->varIterationNumber()->getData() ; i++) 
				{
					cuExecute(numC,
						PBDRB_SolvePositions,
						this->inCenter()->getData(),
						this->inQuaternion()->getData(),
						this->inRotationMatrix()->getData(),
						this->inInitialInertia()->getData(),
						this->inMass()->getData(),
						this->inInertia()->getData(),
						this->mLambda,
						this->mAlpha,
						this->mAllConstraints,
						this->mContactNumber,
						h);
				}
			}

			cuExecute(num,
				PBDRB_CalcVW,
				this->inCenter()->getData(),
				this->x_prev,
				this->inVelocity()->getData(),
				this->inQuaternion()->getData(),
				this->q_prev,
				this->inAngularVelocity()->getData(),
				h);

			this->mLambda.reset();
			this->mAllConstraints.reset();
		}
	}

	template <typename ContactPair>
	__global__ void CalculateNbrCons(
		DArray<ContactPair> nbc,
		DArray<Real> nbrCnt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;

		int idx1 = nbc[pId].bodyId1;
		int idx2 = nbc[pId].bodyId2;

		if (idx1 != -1)
			atomicAdd(&nbrCnt[idx1], 1.0f);
		if (idx2 != -1)
			atomicAdd(&nbrCnt[idx2], 1.0f);
	}

	template <typename Coord, typename Matrix, typename ContactPair>
	__global__ void CalculateJacobians(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<ContactPair> nbc)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbc[pId].bodyId1;
		int idx2 = nbc[pId].bodyId2;

		//printf("%d %d\n", idx1, idx2);

		if (nbc[pId].contactType == ContactType::CT_NONPENETRATION) // contact, collision
		{
			Coord p1 = nbc[pId].pos1;
			Coord p2 = nbc[pId].pos2;
			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = -n;
			J[4 * pId + 3] = -(r2.cross(n));

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = -n / mass[idx2];
			B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
		}
		else if (nbc[pId].contactType == ContactType::CT_BOUDNARY) // boundary
		{
			Coord p1 = nbc[pId].pos1;
			//	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ %d %.3lf %.3lf %.3lf\n", idx1, p1[0], p1[1], p1[2]);

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];


			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = Coord(0);
			J[4 * pId + 3] = Coord(0);

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = Coord(0);
			B[4 * pId + 3] = Coord(0);
		}
		else if (nbc[pId].contactType == ContactType::CT_FRICTION) // friction
		{
			Coord p1 = nbc[pId].pos1;
			//printf("~~~~~~~ %.3lf %.3lf %.3lf\n", p1[0], p1[1], p1[2]);


			Coord p2 = Coord(0);
			if (idx2 != -1)
				p2 = nbc[pId].pos2;

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = Coord(0);
			if (idx2 != -1)
				r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			if (idx2 != -1)
			{
				J[4 * pId + 2] = -n;
				J[4 * pId + 3] = -(r2.cross(n));
			}
			else
			{
				J[4 * pId + 2] = Coord(0);
				J[4 * pId + 3] = Coord(0);
			}
			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			if (idx2 != -1)
			{
				B[4 * pId + 2] = -n / mass[idx2];
				B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
			}
			else
			{
				B[4 * pId + 2] = Coord(0);
				B[4 * pId + 3] = Coord(0);
			}
		}
	}

	template <typename Coord>
	__global__ void CalculateDiagonals(
		DArray<Real> D,
		DArray<Coord> J,
		DArray<Coord> B)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= J.size() / 4) return;

		Real d = Real(0);
		d += J[4 * tId].dot(B[4 * tId]);
		d += J[4 * tId + 1].dot(B[4 * tId + 1]);
		d += J[4 * tId + 2].dot(B[4 * tId + 2]);
		d += J[4 * tId + 3].dot(B[4 * tId + 3]);

		D[tId] = d;
	}

	// ignore zeta !!!!!!
	template <typename Coord, typename ContactPair>
	__global__ void CalculateEta(
		DArray<Real> eta,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> J,
		DArray<Real> mass,
		DArray<ContactPair> nbq,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].bodyId1;
		int idx2 = nbq[pId].bodyId2;
		//printf("from ita %d\n", pId);
		Real ita_i = Real(0);
		if (true) // test dist constraint
		{
			ita_i -= J[4 * pId].dot(velocity[idx1]);
			ita_i -= J[4 * pId + 1].dot(angular_velocity[idx1]);
			if (idx2 != -1)
			{
				ita_i -= J[4 * pId + 2].dot(velocity[idx2]);
				ita_i -= J[4 * pId + 3].dot(angular_velocity[idx2]);
			}
		}
		eta[pId] = ita_i / dt;
		if (nbq[pId].contactType == ContactType::CT_NONPENETRATION || nbq[pId].contactType == ContactType::CT_BOUDNARY)
		{
			eta[pId] += min(nbq[pId].interpenetration, nbq[pId].interpenetration) / dt / dt / 15.0f;
		}
	}

	template <typename ContactPair>
	__global__ void SetupFrictionConstraints(
		DArray<ContactPair> nbq,
		int contact_size)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= contact_size) return;

		Coord3D n = nbq[pId].normal1;
		n /= n.norm();

		Coord3D n1, n2;
		if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
		{
			n1 = Coord3D(0, n[2], -n[1]);
			n1 /= n1.norm();
			n2 = n1.cross(n);
			n2 /= n2.norm();
		}
		else if (abs(n[0]) > EPSILON)
		{
			n1 = Coord3D(n[2], 0, -n[0]);
			n1 /= n1.norm();
			n2 = n1.cross(n);
			n2 /= n2.norm();
		}

		nbq[pId * 2 + contact_size].bodyId1 = nbq[pId].bodyId1;
		nbq[pId * 2 + contact_size].bodyId2 = nbq[pId].bodyId2;
		nbq[pId * 2 + contact_size] = nbq[pId];
		nbq[pId * 2 + contact_size].contactType = ContactType::CT_FRICTION;
		nbq[pId * 2 + contact_size].normal1 = n1;

		nbq[pId * 2 + 1 + contact_size].bodyId1 = nbq[pId].bodyId1;
		nbq[pId * 2 + 1 + contact_size].bodyId2 = nbq[pId].bodyId2;
		nbq[pId * 2 + 1 + contact_size] = nbq[pId];
		nbq[pId * 2 + 1 + contact_size].contactType = ContactType::CT_FRICTION;
		nbq[pId * 2 + 1 + contact_size].normal1 = n2;
	}

	template<typename TDataType>
	void PBDIterativeConstraintSolver<TDataType>::initialize()
	{
		if (this->inContacts()->isEmpty())
			return;

		auto& contacts = this->inContacts()->getData();
		int sizeOfContacts = contacts.size();
		int sizeOfConstraints = sizeOfContacts;

		mAllConstraints.resize(sizeOfConstraints);

		if (contacts.size() > 0)
			mAllConstraints.assign(contacts, contacts.size(), 0, 0);

		mLambda.resize(sizeOfConstraints);
		mAlpha.resize(sizeOfConstraints);

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);

		mLambda.reset();
		mContactNumber.reset();
		mAlpha.reset();

		if (sizeOfConstraints == 0) return;

		cuExecute(sizeOfConstraints,
			CalculateNbrCons,
			mAllConstraints,
			mContactNumber
		);
	}

	template<typename TDataType>
	void PBDIterativeConstraintSolver<TDataType>::initializeJacobian(Real dt)
	{
		//int sizeOfContacts = mBoundaryContacts.size() + contacts.size();

		if (this->inContacts()->isEmpty())
			return;

		auto& contacts = this->inContacts()->getData();
		int sizeOfContacts = contacts.size();
 		int sizeOfConstraints = sizeOfContacts;
		if (this->varFrictionEnabled()->getData())
		{
			sizeOfConstraints += 2 * sizeOfContacts;
		}

		mAllConstraints.resize(sizeOfConstraints);

		if (contacts.size() > 0)
			mAllConstraints.assign(contacts, contacts.size(), 0, 0);

		if (this->varFrictionEnabled()->getData())
		{
			cuExecute(sizeOfContacts,
				SetupFrictionConstraints,
				mAllConstraints,
				sizeOfContacts);
		}


		mJ.resize(4 * sizeOfConstraints);
		mB.resize(4 * sizeOfConstraints);
		mD.resize(sizeOfConstraints);
		mEta.resize(sizeOfConstraints);
		mLambda.resize(sizeOfConstraints);

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);

		mJ.reset();
		mB.reset();
		mD.reset();
		mEta.reset();
		mLambda.reset();
		mContactNumber.reset();

		if (sizeOfConstraints == 0) return;

// 		if (contacts.size() > 0)
// 			mAllConstraints.assign(contacts, contacts.size());
// 
// 		if (mBoundaryContacts.size() > 0)
// 			mAllConstraints.assign(mBoundaryContacts, mBoundaryContacts.size(), contacts.size(), 0);

// 		if (this->varFrictionEnabled()->getData())
// 		{
// 			cuExecute(sizeOfContacts,
// 				SetupFrictionConstraints,
// 				mAllConstraints,
// 				sizeOfContacts);
// 		}
		cuExecute(sizeOfConstraints,
			CalculateNbrCons,
			mAllConstraints,
			mContactNumber
		);

		cuExecute(sizeOfConstraints,
			CalculateJacobians,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mAllConstraints);

		cuExecute(sizeOfConstraints,
			CalculateDiagonals,
			mD,
			mJ,
			mB);

		cuExecute(sizeOfConstraints,
			CalculateEta,
			mEta,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mJ,
			this->inMass()->getData(),
			mAllConstraints,
			dt);
	}



	DEFINE_CLASS(PBDIterativeConstraintSolver);
}