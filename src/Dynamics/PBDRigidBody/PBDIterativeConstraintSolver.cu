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
		DArray<Coord> v_prev,
		DArray<Quat> q,
		DArray<Quat> q_prev,
		DArray<Coord> w,
		DArray<Coord> w_prev,
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= x.size()) return;

		v_prev[pId] = v[pId];
		w_prev[pId] = w[pId];
		v[pId] = (x[pId] - x_prev[pId]) / h;

		Quat dq = q[pId] * (q_prev[pId].inverse());
		w[pId] = (dq.w >=0 ? 1:-1) * 2 * Coord(dq.x , dq.y , dq.z) / h;
	}

	template <typename Matrix, typename Quat>
	__global__ void PBDRB_UpdateRI(
		DArray<Quat> q,
		DArray<Matrix> I_init,
		DArray<Matrix> I,
		DArray<Matrix> R)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= q.size()) return;

		R[pId] = q[pId].toMatrix3x3();
		I[pId] = R[pId] * I_init[pId] * R[pId].inverse();
	}

	template <typename Coord, typename Quat, typename Matrix, typename Real, typename ContactPair>
	__global__ void PBDRB_SolvePositions(
		DArray<Coord> x,
		DArray<Coord> x_prev,
		DArray<Quat> q,
		DArray<Quat> q_prev,
		DArray<Real> m,
		DArray<Matrix> I,
		DArray<Real> lambdaN,
		DArray<Real> lambdaT,
		DArray<Real> alpha,
		DArray<Real> miuS,
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

		if (c > 0) 
		{
			if (idx2 != -1)
			{
				Coord r1 = nbq[pId].pos1 - x[idx1];
				Coord r2 = nbq[pId].pos1 - x[idx2];

				Coord p1 = x[idx1] + r1;
				Coord p2 = x[idx2] + r2;

				Coord p1Bar = x_prev[idx1] + (q_prev[idx1] - q[idx1]).rotate(r1);
				Coord p2Bar = x_prev[idx2] + (q_prev[idx2] - q[idx2]).rotate(r2);

				Coord temp3 = r1.cross(n);
				Coord temp4 = r2.cross(n);

				Real w1 = 1.0f / m[idx1] + (Real)(temp3.dot(I[idx1].inverse() * temp3));
				Real w2 = 1.0f / m[idx2] + (Real)(temp4.dot(I[idx2].inverse() * temp4));

				Real dLambdaN = ((-c - tildeAlpha * lambdaN[pId]) / (w1 + w2 + tildeAlpha));
				dLambdaN *= 1 / (stepInv[idx1] + stepInv[idx2]);
				lambdaN[pId] += dLambdaN;

				Coord p = dLambdaN * n;

				Coord temp1 = I[idx1].inverse() * (r1.cross(p));
				Coord temp2 = I[idx2].inverse() * (r2.cross(p));
				Quat temp5 = Quat(temp1[0], temp1[1], temp1[2], 0) * q[idx1] * 0.5f;
				Quat temp6 = Quat(temp2[0], temp2[1], temp2[2], 0) * q[idx2] * 0.5f;

				atomicAdd(&x[idx1][0], p[0] / m[idx1]);
				atomicAdd(&x[idx1][1], p[1] / m[idx1]);
				atomicAdd(&x[idx1][2], p[2] / m[idx1]);

				atomicAdd(&x[idx2][0], -p[0] / m[idx2]);
				atomicAdd(&x[idx2][1], -p[1] / m[idx2]);
				atomicAdd(&x[idx2][2], -p[2] / m[idx2]);

				atomicAdd(&q[idx1].x, temp5.x);
				atomicAdd(&q[idx1].y, temp5.y);
				atomicAdd(&q[idx1].z, temp5.z);
				atomicAdd(&q[idx1].w, temp5.w);

				atomicAdd(&q[idx2].x, -temp6.x);
				atomicAdd(&q[idx2].y, -temp6.y);
				atomicAdd(&q[idx2].z, -temp6.z);
				atomicAdd(&q[idx2].w, -temp6.w);

				Coord dP = (p1 - p1Bar) - (p2 - p2Bar);
				if (dP.norm() < EPSILON || dP.norm() > -EPSILON)
					dP = Coord(0.0f);
				Coord dP_t = dP - (dP.dot(n)) * n;
				if (dP_t.norm() < EPSILON || dP_t.norm() > -EPSILON)
					dP_t = Coord(0.0f);

				Real dLambdaT = ((-dP_t.norm() - tildeAlpha * lambdaT[pId]) / (w1 + w2 + tildeAlpha));
				dLambdaT *= 1 / (stepInv[idx1] + stepInv[idx2]);
				lambdaT[pId] += dLambdaT;

				if (lambdaT[pId] > (miuS[idx1] + miuS[idx2]) / 2 * lambdaN[pId])
				{
					Coord temp7 = I[idx1].inverse() * (r1.cross(dP_t));
					Coord temp8 = I[idx2].inverse() * (r2.cross(dP_t));
					Quat temp9 = Quat(temp7[0], temp7[1], temp7[2], 0) * q[idx1] * 0.5f;
					Quat temp10 = Quat(temp8[0], temp8[1], temp8[2], 0) * q[idx2] * 0.5f;

					atomicAdd(&x[idx1][0], dP_t[0] / m[idx1]);
					atomicAdd(&x[idx1][1], dP_t[1] / m[idx1]);
					atomicAdd(&x[idx1][2], dP_t[2] / m[idx1]);

					atomicAdd(&x[idx2][0], -dP_t[0] / m[idx2]);
					atomicAdd(&x[idx2][1], -dP_t[1] / m[idx2]);
					atomicAdd(&x[idx2][2], -dP_t[2] / m[idx2]);

					atomicAdd(&q[idx1].x, temp9.x);
					atomicAdd(&q[idx1].y, temp9.y);
					atomicAdd(&q[idx1].z, temp9.z);
					atomicAdd(&q[idx1].w, temp9.w);

					atomicAdd(&q[idx2].x, -temp10.x);
					atomicAdd(&q[idx2].y, -temp10.y);
					atomicAdd(&q[idx2].z, -temp10.z);
					atomicAdd(&q[idx2].w, -temp10.w);
				}
			}
			else
			{
				Coord r1 = nbq[pId].pos1 - x[idx1];
				Coord p1 = x[idx1] + r1;
				Coord p1Bar = x_prev[idx1] + (q_prev[idx1] - q[idx1]).rotate(r1);
				//printf("%f\t%f\t%f\n",p1Bar[0], p1Bar[1], p1Bar[2]);
				Coord temp3 = r1.cross(n);

				Real w1 = 1.0f / m[idx1] + (Real)(temp3.dot(I[idx1].inverse() * temp3));
				Real w2 = 0.0f;
				Real dLambdaN = ((-c - tildeAlpha * lambdaN[pId]) / (w1 + w2 + tildeAlpha));
				dLambdaN /= stepInv[idx1];
				lambdaN[pId] += dLambdaN;
				Coord p = dLambdaN * n;
				Coord temp1 = I[idx1].inverse() * (r1.cross(p));
				Quat temp2 = 0.5f * Quat(temp1[0], temp1[1], temp1[2], 0) * q[idx1];
				atomicAdd(&x[idx1][0], p[0] / m[idx1]);
				atomicAdd(&x[idx1][1], p[1] / m[idx1]);
				atomicAdd(&x[idx1][2], p[2] / m[idx1]);

				atomicAdd(&q[idx1].x, temp2.x);
				atomicAdd(&q[idx1].y, temp2.y);
				atomicAdd(&q[idx1].z, temp2.z);
				atomicAdd(&q[idx1].w, temp2.w);

				Coord dP = (p1 - p1Bar);
				if (dP.norm() < EPSILON || dP.norm() > -EPSILON)
					dP = Coord(0.0f);
				Coord dP_t = dP - (dP.dot(n)) * n;
				if (dP_t.norm() < EPSILON || dP_t.norm() > -EPSILON)
					dP_t = Coord(0.0f);

				Real dLambdaT = ((-dP_t.norm() - tildeAlpha * lambdaT[pId]) / (w1 + w2 + tildeAlpha));
				dLambdaT *= 1 / stepInv[idx1];
				lambdaT[pId] += dLambdaT;

				if (lambdaT[pId] > miuS[idx1] * lambdaN[pId])
				{
					Coord temp7 = I[idx1].inverse() * (r1.cross(dP_t));
					Quat temp9 = Quat(temp7[0], temp7[1], temp7[2], 0) * q[idx1] * 0.5f;

					atomicAdd(&x[idx1][0], dP_t[0] / m[idx1]);
					atomicAdd(&x[idx1][1], dP_t[1] / m[idx1]);
					atomicAdd(&x[idx1][2], dP_t[2] / m[idx1]);

					atomicAdd(&q[idx1].x, temp9.x);
					atomicAdd(&q[idx1].y, temp9.y);
					atomicAdd(&q[idx1].z, temp9.z);
					atomicAdd(&q[idx1].w, temp9.w);

				}
			}
		}
	}

	template <typename Coord, typename Matrix, typename Real, typename ContactPair>
	__global__ void PBDRB_SolveVelocities(
		DArray<Coord> x,
		DArray<Coord> v,
		DArray<Coord> v_prev,
		DArray<Coord> w,
		DArray<Coord> w_prev,
		DArray<Real> m,
		DArray<Matrix> I,
		DArray<Real> lambda,
		DArray<Real> miu,
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

		if (idx2 != -1)
		{
			Coord r1 = nbq[pId].pos1 - x[idx1];
			Coord r2 = nbq[pId].pos1 - x[idx2];

			Coord vv = (v[idx1] + w[idx1].cross(r1)) - (v[idx2] + w[idx2].cross(r2));
			Real v_n = n.dot(vv);
			Coord v_t = vv - v_n*n;

			Coord vv_prev = (v_prev[idx1] + w_prev[idx1].cross(r1)) - (v_prev[idx2] + w_prev[idx2].cross(r2));
			Real v_n_prev = n.dot(vv_prev) > 0 ? 0.0f : abs(n.dot(vv_prev));

			Coord dv = Coord(0.0f);
			if (abs(v_n) < h * 9.8f * 2)
				dv += n * (-v_n) / (stepInv[idx1]+ stepInv[idx2]);
			else
				dv += n * (-v_n - 0.5f * v_n_prev) / (stepInv[idx1]+ stepInv[idx2]);

			if (v_t.norm() > 0)
			{
				Real miu_d = (miu[idx1] + miu[idx2]) / 2;
				dv += (v_t / v_t.norm() * min(h * miu_d * (lambda[pId] * 1000.0f / h / h), v_t.norm())) ;

				Coord temp1 = r1.cross(n);
				Coord temp2 = r2.cross(n);

				Real w1 = 1.0f / m[idx1] + (Real)(temp1.dot(I[idx1].inverse() * temp1));
				Real w2 = 1.0f / m[idx2] + (Real)(temp2.dot(I[idx2].inverse() * temp2));
				Coord p = dv / (w1 + w2);

				Coord temp3 = I[idx1].inverse() * (r1.cross(p));
				Coord temp4 = I[idx2].inverse() * (r2.cross(p));

				atomicAdd(&v[idx1][0], p[0] / m[idx1]);
				atomicAdd(&v[idx1][1], p[1] / m[idx1]);
				atomicAdd(&v[idx1][2], p[2] / m[idx1]);

				atomicAdd(&v[idx2][0], -p[0] / m[idx2]);
				atomicAdd(&v[idx2][1], -p[1] / m[idx2]);
				atomicAdd(&v[idx2][2], -p[2] / m[idx2]);

				atomicAdd(&w[idx1][0], temp3[0]);
				atomicAdd(&w[idx1][1], temp3[1]);
				atomicAdd(&w[idx1][2], temp3[2]);

				atomicAdd(&w[idx2][0], -temp4[0]);
				atomicAdd(&w[idx2][1], -temp4[1]);
				atomicAdd(&w[idx2][2], -temp4[2]);
			}
		}
		else 
		{
			Coord r1 = nbq[pId].pos1 - x[idx1];

			Coord vv = v[idx1] + w[idx1].cross(r1);
			Real v_n = n .dot(vv);
			Coord v_t = vv - v_n * n;

			Coord vv_prev = v_prev[idx1] + w_prev[idx1].cross(r1);
			Real v_n_prev = n.dot(vv_prev) > 0 ? 0.0f: abs(n.dot(vv_prev));

			Coord dv = Coord(0.0f);
			if(abs(v_n)<h*9.8f*2)
				dv += n * (-v_n) / (stepInv[idx1]);
			else
				dv += n * (-v_n - 0.5f * v_n_prev) / (stepInv[idx1]);

			//printf("%.10f\t%.10f\n", v_n, v_n_prev);

			if (v_t.norm() > 0)
			{
				Real miu_d = miu[idx1];
				dv += (v_t / v_t.norm() * min(h * miu_d * (lambda[pId] * 1000.0f / h / h), v_t.norm()));

				Coord temp1 = r1.cross(n);

				Real w1 = 1.0f / m[idx1] + temp1.dot(I[idx1].inverse() * temp1);
				Real w2 = 0.0f;
				Coord p = dv / (w1 + w2);

				Coord temp3 = I[idx1].inverse() * (r1.cross(p));

				atomicAdd(&v[idx1][0], p[0] / m[idx1]);
				atomicAdd(&v[idx1][1], p[1] / m[idx1]);
				atomicAdd(&v[idx1][2], p[2] / m[idx1]);
				//printf("%.10f\t%.10f\t%.10f\n", v[idx1][0], v[idx1][1], v[idx1][2]);

				atomicAdd(&w[idx1][0], temp3[0]);
				atomicAdd(&w[idx1][1], temp3[1]);
				atomicAdd(&w[idx1][2], temp3[2]);
			}
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

		if (this->v_prev.size() == 0)
			this->v_prev.resize(num);

		if (this->w_prev.size() == 0)
			this->w_prev.resize(num);

		Coord g = Coord(0.0f, -9.8f, 0.0f);
		Coord tau = Coord(0.0f, 0.0f, 0.0f);
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
			cuExecute(numC,
				PBDRB_SolvePositions,
				this->inCenter()->getData(),
				this->x_prev,
				this->inQuaternion()->getData(),
				this->q_prev,
				this->inMass()->getData(),
				this->inInertia()->getData(),
				this->mLambdaN,
				this->mLambdaT,
				this->mAlpha,
				this->inStaticFriction()->getData(),
				this->mAllConstraints,
				this->mContactNumber,
				h);

			cuExecute(
				num,
				PBDRB_UpdateRI,
				this->inQuaternion()->getData(),
				this->inInitialInertia()->getData(),
				this->inInertia()->getData(),
				this->inRotationMatrix()->getData());
		}

		cuExecute(
			num,
			PBDRB_CalcVW,
			this->inCenter()->getData(),
			x_prev,
			this->inVelocity()->getData(),
			this->v_prev,
			this->inQuaternion()->getData(),
			q_prev,
			this->inAngularVelocity()->getData(),
			this->w_prev,
			h);

		if (this->varFrictionEnabled()->getData())
		{
			if (numC > 0)
			{
				cuExecute(
					numC,
					PBDRB_SolveVelocities,
					this->inCenter()->getData(),
					this->inVelocity()->getData(),
					this->v_prev,
					this->inAngularVelocity()->getData(),
					this->w_prev,
					this->inMass()->getData(),
					this->inInertia()->getData(),
					this->mLambdaN,
					this->inDynamicFriction()->getData(),
					this->mAllConstraints,
					this->mContactNumber,
					h);
			}
		}

		this->mLambdaN.reset();
		this->mLambdaT.reset();
		this->mAllConstraints.reset();
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

		mLambdaN.resize(sizeOfConstraints);
		mLambdaT.resize(sizeOfConstraints);
		mAlpha.resize(sizeOfConstraints);

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);

		mLambdaN.reset();
		mLambdaT.reset();
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
		mLambdaN.resize(sizeOfConstraints);

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);

		mJ.reset();
		mB.reset();
		mD.reset();
		mEta.reset();
		mLambdaN.reset();
		mContactNumber.reset();

		if (sizeOfConstraints == 0) 
			return;

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