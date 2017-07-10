/*
 * springdynamics.cpp
 *
 *  Created on: 28 May 2015
 *      Author: jadarve
 */

#include "spherepix/springdynamics.h"

#include <iostream>
#include <memory>
#include <cstring>

#include "Eigen/Dense"
#include "spherepix/image.h"
#include "spherepix/geometry.h"
#include "spherepix/arithmetic.h"



// using namespace std;
// using namespace Eigen;

namespace spherepix {

void springSystemTimeStep(const Image<float>& etas_in,
                          const Image<float>& etasVelocity_in,
                          const Image<float>& etasAcceleration_in,
                          Image<float>& etas_out,
                          Image<float>& etasVelocity_out,
                          Image<float>& etasAcceleration_out,
                          const float dt,
                          const float M,
                          const float C,
                          const float K,
                          const float L) {

	const bool checkCoordinates = false;

	int height = etas_in.height();
	int width = etas_in.width();

	#ifdef _OPENMP
	// set the number of threads reserved to Eigen to 1, so that
	// each Eigen operation in the parallel for is executed in
	// the same thread.
	int eigenThreads = Eigen::nbThreads();
	Eigen::setNbThreads(1);

		#pragma omp parallel default(shared)
		{
			#pragma omp for
	#endif
	for (int r = 0; r < height; r ++) {

		#ifdef _OPENMP
			#pragma omp parallel shared(r, height, etas_in, etasVelocity_in, etasAcceleration_in, etas_out, etasVelocity_out, etasAcceleration_out)
			{
				#pragma omp for
		#endif
		for (int c = 0; c < width; c ++) {

			Eigen::Vector3f eta = readVector3f<checkCoordinates>(etas_in, r, c);
			Eigen::Vector3f eta_vel = readVector3f<checkCoordinates>(etasVelocity_in, r, c);

			Eigen::Matrix2Xf B = getOrthonormalBasis(etas_in, r, c);
			Eigen::MatrixXf BT = B.transpose();

			Eigen::Vector2f beta_vel = B * eta_vel;

			// Damping force
			Eigen::Vector2f C_total = C * beta_vel;

			// Elastic force
			Eigen::Vector2f V_total(0.0f, 0.0f);


			// for the 8-neighborhood of eta
			for (int rr = -1; rr < 2; rr ++) {
				int rr_p = r + rr;

				for (int cc = -1; cc < 2; cc ++) {
					int cc_p = c + cc;

					// check neighbor boundaries
					if (rr_p >= 0 && rr_p < height && cc_p >= 0 && cc_p < width) {

						if (rr != 0 || cc != 0) {

							Eigen::Vector3f eta_j = readVector3f<checkCoordinates>(etas_in, rr_p, cc_p);
							Eigen::Vector3f mu_j = etaToMu_orthographic(eta, eta_j);
							Eigen::Vector2f beta_j = B * mu_j;
							float beta_j_norm = beta_j.norm();

							Eigen::Vector2f beta_j_normalized = beta_j_norm > 0.0f? 
								beta_j.normalized() : Eigen::Vector2f(0.0f, 0.0f);

							// for diagonal springs
							float Lfactor = abs(rr) + abs(cc) > 1 ? sqrtf(2.0f) : 1.0f;

							// add the elastic force of this connection to the total
							V_total += K * (beta_j_norm - L * Lfactor) * beta_j_normalized;
						}
					}
				}
			}

			// Euler integration
			Eigen::Vector2f beta_acc_p = (V_total - C_total) / M;
			Eigen::Vector2f beta_vel_p = beta_vel + dt * beta_acc_p;
			Eigen::Vector2f beta_pos_p = dt * beta_vel_p;

			// State conversion to original coordinates
			Eigen::Vector3f mu_pos_p = BT * beta_pos_p;

			// Convert position from mu to eta coordinates
			Eigen::Vector3f eta_pos_p = muToEta_orthographic(eta, mu_pos_p);
			eta_pos_p.normalize();

			// Rotation matrix from eta to eta_pos_p
			Eigen::Matrix3f R = rotationMatrix(eta, eta_pos_p);

			// Convert velocity and acceleration from eta TS to eta_pos_p TS
			Eigen::Vector3f mu_acc_p = R * BT * beta_acc_p;
			Eigen::Vector3f mu_vel_p = R * BT * beta_vel_p;

			// Results packaging
			writeVector3f<checkCoordinates>(eta_pos_p, r, c, etas_out);
			writeVector3f<checkCoordinates>(mu_vel_p, r, c, etasVelocity_out);
			writeVector3f<checkCoordinates>(mu_acc_p, r, c, etasAcceleration_out);

			if (isNaN(eta_pos_p)) {
				std::cout << "-----------------------------------------------------------" << std::endl;
				std::cout << "(" << r << ", " << c << ")" << std::endl;
				std::cout << "beta_pos: " << beta_pos_p.transpose() << std::endl;
				std::cout << "beta_vel_p: " << beta_vel_p.transpose() << std::endl;
				std::cout << "beta_acc_p: " << beta_acc_p.transpose() << std::endl;
				std::cout << "V_total: " << V_total.transpose() << std::endl;
				std::cout << "C_total: " << C_total.transpose() << std::endl;
				std::cout << "eta_vel: " << eta_vel.transpose() << std::endl;
				std::cout << "eta: " << eta.transpose() << std::endl;
				std::cout << "eta_pos_p: " << eta_pos_p.transpose() << std::endl;
				std::cout << "mu_vel_p: " << mu_vel_p.transpose() << std::endl;
				std::cout << "mu_acc_p: " << mu_acc_p.transpose() << std::endl;
				std::cout << "R: " << R << std::endl;
				// std::cout << "BT: " << BT << std::endl;
				std::cout << "BT: " << B.transpose() << std::endl;

				Eigen::Vector3f eta_1 = readVector3f<checkCoordinates>(etas_in, 5, 28);
				std::cout << "eta_1: " << eta_1.transpose() << std::endl;
				Eigen::Vector3f mu_1 = etaToMu_orthographic(eta, eta_1);
				std::cout << "mu_1: " << mu_1.transpose() << std::endl;
				std::cout << "mu_1 normalized: " << mu_1.normalized().transpose() << std::endl;
			}
		}
		#ifdef _OPENMP
		}
		#endif
	}
	#ifdef _OPENMP
	}
	// restore the number of threads allocated to Eigen
	Eigen::setNbThreads(eigenThreads);
	#endif
}


void runSpringSystem(const Image<float>& etas_in,
                     const Image<float>& etasVelocity_in,
                     const Image<float>& etasAcceleration_in,
                     Image<float>& etas_out,
                     Image<float>& etasVelocity_out,
                     Image<float>& etasAcceleration_out,
                     const float dt,
                     const float M,
                     const float C,
                     const float K,
                     const float L,
                     const int N) {

	int height = etas_in.height();
	int width = etas_in.width();

	// buffers to store the iteration results
	Image<float> etas_0(height, width, 3);
	Image<float> etasVelocity_0(height, width, 3);
	Image<float> etasAcceleration_0(height, width, 3);

	Image<float> etas_1(height, width, 3);
	Image<float> etasVelocity_1(height, width, 3);
	Image<float> etasAcceleration_1(height, width, 3);

	// copy initial conditions to *_0 buffers
	etas_0.copyFrom(etas_in);
	etasVelocity_0.copyFrom(etasVelocity_in);
	etasAcceleration_0.copyFrom(etasAcceleration_in);

	// run the iterations
	bool swap = false;
	for (int n = 0; n < N; n ++) {
		if (swap) {
			springSystemTimeStep(etas_1,
			                     etasVelocity_1,
			                     etasAcceleration_1,
			                     etas_0,
			                     etasVelocity_0,
			                     etasAcceleration_0,
			                     dt, M, C, K, L);
		} else {
			springSystemTimeStep(etas_0,
			                     etasVelocity_0,
			                     etasAcceleration_0,
			                     etas_1,
			                     etasVelocity_1,
			                     etasAcceleration_1,
			                     dt, M, C, K, L);
		}

		swap = !swap;
	}

	// copy results to output arrays
	if (swap) {
		etas_out.copyFrom(etas_1);
		etasVelocity_out.copyFrom(etasVelocity_1);
		etasAcceleration_out.copyFrom(etasAcceleration_1);
	} else {
		etas_out.copyFrom(etas_0);
		etasVelocity_out.copyFrom(etasVelocity_0);
		etasAcceleration_out.copyFrom(etasAcceleration_0);
	}
}

}; // namespace spherepix