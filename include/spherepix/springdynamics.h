/**
 * springdynamics.h
 */

#ifndef SPHEREPIX_SPRINGDYNAMICS_HPP_
#define SPHEREPIX_SPRINGDYNAMICS_HPP_

#include "image.h"

namespace spherepix {

/**
 * \brief Runs one time step iteration of the spring system
 *
 * \param etas_in input spherical coordinates grid, (x, y, z) normalized
 *
 * \param etasVelocity_in input velocity grid. (vx, vy, vz) lying
 *		on the tangent space of etas_in
 *
 * \param etasAcceleration_in input acceleration grid. (ax, ay, az) lying
 *		on the tangent space of etas_in
 *
 * \param etas_out output spherical coordinates grid, (x, y, z) normalized
 *
 * \param etasVelocity_out output velocity grid. (vx, vy, vz) lying
 *		on the tangent space of etas_out
 *
 * \param etasAcceleration_out output acceleration grid. (ax, ay, az) lying
 *		on the tangent space of etas_out
 *
 * \param dt time step
 * \param M point mass
 * \param C damping coefficient
 * \param K spring elasticity constant
 * \param L spring rest longitude
 */
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
                          const float L);

/**
 * \brief Runs the spring system for several iterations
 *
 * \param etas_in input spherical coordinates grid, (x, y, z) normalized
 *
 * \param etasVelocity_in input velocity grid. (vx, vy, vz) lying
 *		on the tangent space of etas_in
 *
 * \param etasAcceleration_in input acceleration grid. (ax, ay, az) lying
 *		on the tangent space of etas_in
 *
 * \param etas_out output spherical coordinates grid, (x, y, z) normalized
 *
 * \param etasVelocity_out output velocity grid. (vx, vy, vz) lying
 *		on the tangent space of etas_out
 *
 * \param etasAcceleration_out output acceleration grid. (ax, ay, az) lying
 *		on the tangent space of etas_out
 *
 * \paramm dt time step
 * \param M point mass
 * \param C damping coefficient
 * \param K spring elasticity constant
 * \param L spring rest longitude
 * \param N number of iterations
 */
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
                     const int N);


}; // namespace spherepix

#endif // SPHEREPIX_SPRINGDYNAMICS_HPP_
