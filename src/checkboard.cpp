/*
 * checkboard.cpp
 *
 *  Created on: 8 Jun 2015
 *      Author: jadarve
 */

#include "checkboard.h"
#include "geometry.h"
#include "util.h"

#include <vector>
#include <iostream>

using namespace std;
using namespace Eigen;

Eigen::Vector2f findInterpolationCoodinates(const Eigen::Vector3f eta, const float* etas, const int height, const int width) {

	// center of the 2D search space
	int centerH = (height / 2) -1;
	int centerW = (width / 2) -1;

	// increment
	int incH = height / 4;
	int incW = width / 4;

	// children index in the subdivision
	int childrenIndex = 0;

	while(incH > 0 || incW > 0) {

//		cout << "(cH, cW): (" << centerH << ", " << centerW
//				<< ")\t(iH, iW): (" << incH << ", " << incW << ")\tchild: " << childrenIndex << endl;

		Vector3f eta_00 = readVector3f(etas, centerH - incH, centerW - incW, width);
		Vector3f eta_01 = readVector3f(etas, centerH - incH, centerW + incW, width);
		Vector3f eta_10 = readVector3f(etas, centerH + incH, centerW - incW, width);
		Vector3f eta_11 = readVector3f(etas, centerH + incH, centerW + incW, width);

		// FIXME: change the retractions to geodesic so the search can support wide angles
//		float n_00 = etaToMu_orthographic(eta_00, eta).norm();
//		float n_01 = etaToMu_orthographic(eta_01, eta).norm();
//		float n_10 = etaToMu_orthographic(eta_10, eta).norm();
//		float n_11 = etaToMu_orthographic(eta_11, eta).norm();

		float n_00 = etaToMu_geodesic(eta_00, eta).norm();
		float n_01 = etaToMu_geodesic(eta_01, eta).norm();
		float n_10 = etaToMu_geodesic(eta_10, eta).norm();
		float n_11 = etaToMu_geodesic(eta_11, eta).norm();

		// find the index of minimum norm
		vector<float> norm_vector = {n_00, n_01, n_10, n_11};
		childrenIndex = min_element(norm_vector.begin(), norm_vector.end()) - norm_vector.begin();

		// assign the new center position according to the children index
		// with minimum distance
		switch(childrenIndex) {

			case 0:
				centerH = centerH - incH;
				centerW = centerW - incW;
				break;
			case 1:
				centerH = centerH - incH;
				centerW = centerW + incW;
				break;
			case 2:
				centerH = centerH + incH;
				centerW = centerW - incW;
				break;
			case 3:
				centerH = centerH + incH;
				centerW = centerW + incW;
				break;
		}

		incH /= 2;
		incW /= 2;
	}

//	cout << "(cH, cW): (" << centerH << ", " << centerW
//					<< ")\t(iH, iW): (" << incH << ", " << incW << ")\tchild: " << childrenIndex << endl;

	// the last centerH/W is the nearest pixel position to eta

	// get an interpolation basis
	Vector3f eta_0 = readVector3f(etas, centerH, centerW, width);

//	Matrix2Xf B = getOrthonormalBasis(etas, eta_0, centerH, centerW, width);

//	Matrix2Xf Bi = getInterpolationBasis(etas, eta_0, centerH, centerW, height, width);
//	Vector3f mu = etaToMu_geodesic(eta_0, eta);
//	Vector2f betaInterp = Bi*mu;

	pair<Matrix2Xf, float> Bpair = getOrthonormalBasisWithNorm(etas, eta_0, centerH, centerW, width);
	Matrix2Xf B = Bpair.first;
	float norm = Bpair.second;

	Vector3f mu = etaToMu_geodesic(eta_0, eta);

	// interpolated beta coordinates
	Vector2f betaInterp = (B*mu)/norm;

	// (column, row) coordinates
	Vector2f beta = Vector2f((float)centerH + betaInterp.y(), (float)centerW + betaInterp.x());

//	cout << "eta_0:\n" << eta_0 << endl;
//	cout << "eta:\n" << eta << endl;
//	cout << "B:\n" << B << endl;
//	cout << "norm B:\n" << norm << endl;
//	cout << "Bi:\n" << Bi << endl;
//	cout << "mu interp:\n" << mu << endl;
//	cout << "beta interp:\n" << betaInterp << endl;
//	cout << "beta:\n" << beta << endl;

	return beta;
}

Eigen::Vector2f findInterpolationCoodinatesExhaustive(const Eigen::Vector3f eta, const float* etas, const int height, const int width) {

	// center of the 2D search space
	int centerH = (height / 2) -1;
	int centerW = (width / 2) -1;

	// closest point in the grid to the query eta
	Vector3f etaClosest = readVector3f(etas, centerH, centerW, width);

	float minDist = etaToMu_geodesic(etaClosest, eta).norm();
	minDist = isnanf(minDist)? 0.0f : minDist;

//	float minDist = etaToMu_orthographic(etaClosest, eta).norm();

	bool stop = false;

	while(!stop) {

//		cout << "(cH, cW): (" << centerH << ", " << centerW << ") dist: " << minDist << endl;

		// assume this is the last iteration of the search
		stop = true;

		// initialize the values of the new center position and minDist
		// with the old values
		float newMinDist = minDist;
		int newCenterH = centerH, newCenterW = centerW;

		// check the distance of all 8 neighbors of (centerH, centerW)
		for(int r = -1; r < 2; r ++) {
			int rr = centerH + r;

			for(int c = -1; c < 2; c ++) {
				int cc = centerW + c;

				// if the neighbor point is within range of grid
				if(rr >= 0 && rr < height && cc >= 0 && cc < width) {

					Vector3f etaNeighbor = readVector3f(etas, rr, cc, width);
					float dist = etaToMu_orthographic(etaNeighbor, eta).norm();
					dist = isnanf(dist)? 0.0f : dist;

//					float dist = etaToMu_orthographic(etaNeighbor, eta).norm();

					// if the distance between the neighbor and eta
					// is less than the current
					if(dist < newMinDist) {
						newMinDist = minDist;
						newCenterH = rr;
						newCenterW = cc;

						// at least one neighbor was found to have a smaller
						// distance to eta than (centerH, centerW)
//						stop = false;
					}

				}
			}
		}

		stop = centerH == newCenterH && centerW == newCenterW;

		centerH = newCenterH;
		centerW = newCenterW;
		minDist = newMinDist;
	}


	etaClosest = readVector3f(etas, centerH, centerW, width);

	// FIXME: Here I am assuming the neighboring pixel is locally orthogonal. It may not be the case
	pair<Matrix2Xf, float> Bpair = getOrthonormalBasisWithNorm(etas, etaClosest, centerH, centerW, width);
	Matrix2Xf B = Bpair.first;
	float baseNorm = Bpair.second;

	Vector3f mu = etaToMu_geodesic(etaClosest, eta);
	float muNorm = mu.norm();
	mu = isnanf(muNorm) ? Vector3f::Zero() : mu;
//	Vector3f mu = etaToMu_orthographic(etaClosest, eta);

	// interpolated beta coordinates
	Vector2f betaInterp = (B*mu)/baseNorm;

	// (column, row) coordinates
	Vector2f beta = Vector2f((float)centerH + betaInterp.y(), (float)centerW + betaInterp.x());

//	cout << "beta interp:\n" << betaInterp << endl;
//	cout << "beta:\n" << beta << endl;

	return beta;

	// interpolation basis
//	Matrix2Xf I = getInterpolationBasis(etas, etaClosest, centerH, centerW, height, width);
////	Vector3f mu = etaToMu_geodesic(etaClosest, eta);
//	Vector3f mu = etaToMu_orthographic(etaClosest, eta);
//	Vector2f betaInterp = I*mu;
//
//	Vector2f beta = Vector2f((float)centerH + betaInterp.y(), (float)centerW + betaInterp.x());
//	return beta;
}

void castBetaCoordinates(const float* etas, float* betas,
		const int etasHeight, const int etasWidth,
		const float* otherEtas, const int otherHeight, const int otherWidth) {

	for(int r = 0; r < etasHeight; r ++) {
		for(int c = 0; c < etasWidth; c ++) {
			// read eta from etas
			Vector3f eta = readVector3f(etas, r, c, etasWidth);

			// search for the interpolation coordinate of eta in otherEtas array
			Vector2f betaInterp = findInterpolationCoodinates(eta, otherEtas, otherHeight, otherWidth);
			float bx = betaInterp.x();
			float by = betaInterp.y();

			// if the beta coordinate falls within the dimensions of
			// otherEtas grid, write the interpolated coordinates,
			// otherwise fill with negative numbers
			if(bx >= 0.0f && bx < otherWidth && by >=0 && by < otherHeight) {
				writeVector2f(betas, betaInterp, r, c, etasWidth);
			} else {
				writeVector2f(betas, Vector2f(-1.0f, -1.0f), r, c, etasWidth);
			}
		}
	}
}

void castBetaCoordinatesExhaustive(	const float* etas, float* betas,
									const int etasHeight, const int etasWidth,
									const float* otherEtas, const int otherHeight,
									const int otherWidth) {

	for(int r = 0; r < etasHeight; r ++) {
		for(int c = 0; c < etasWidth; c ++) {
			// read eta from etas
			Vector3f eta = readVector3f(etas, r, c, etasWidth);

			// search for the interpolation coordinate of eta in otherEtas array
			Vector2f betaInterp = findInterpolationCoodinatesExhaustive(eta, otherEtas, otherHeight, otherWidth);
			float bx = betaInterp.x();
			float by = betaInterp.y();

			// if the beta coordinate falls within the dimensions of
			// otherEtas grid, write the interpolated coordinates,
			// otherwise fill with negative numbers
			if(bx >= 0.0f && bx < otherWidth && by >=0 && by < otherHeight) {
				writeVector2f(betas, betaInterp, r, c, etasWidth);
			} else {
				writeVector2f(betas, Vector2f(-1.0f, -1.0f), r, c, etasWidth);
			}
		}
	}
}


void convolve_row(		const float* img, const int height, const int width, const int channels,
						const float* belt_left, const float* belt_right, const int beltWidth,
						const float* mask, const int maskLength,
						float* img_output) {

	int offset = maskLength / 2;

//	cout << "convolve_row(): belt width: " << beltWidth << endl;

	// for each pixel in the face
	for(int r = 0; r < height; r ++) {
		for(int c = 0; c < width; c ++) {

			// convolution accumulator
			float convSum = 0.0f;

			// for each point in the mask
			for(int cm = -offset; cm <= offset; cm ++) {

				// column position in the image frame
				int cc = c + cm;

				// mask coefficient
				// reads mask coefficients in reverse order as scipy.ndimage does
				float coeff = mask[maskLength -1 - (cm + offset)];

				float imgValue;

				// if column position falls within the range of the face
				// perform the convolution as usual
				if(cc >= 0 && cc < width) {
					imgValue = img[r*width + cc];
				} else {			// get image value from belts

					if(cc < 0) { 	// left face (cc < 0)
						imgValue = belt_left[r*beltWidth + (beltWidth + cc)];		// cc is a negative number
//						cout << "left: [" << r << ", " << (beltWidth + cc) << "]: c " << c << endl;
					}
					else { 			// right face (cc >= width)
						imgValue = belt_right[r*beltWidth + (cc - width)];
//						cout << "right: [" << r << ", " << (cc - width) << "]: c " << c << endl;
					}
				}

				convSum += coeff*imgValue;
			}

			// store the convolution value
			img_output[r*width + c] = convSum;
		}
	}
}

void convolve_col(	const float* img, const int height, const int width, const int channels,
					const float* belt_top, const float* belt_bottom, const int beltWidth,
					const float* mask, const int maskLength,
					float* img_output) {

	int offset = maskLength / 2;

//	cout << "beltWidth: " << beltWidth << endl;

		// for each pixel in the face
	for(int r = 0; r < height; r ++) {
		for(int c = 0; c < width; c ++) {

			// convolution accumulator
			float convSum = 0.0f;

			// for each point in the mask
			for(int rm = -offset; rm <= offset; rm ++) {

				// row position in the image frame
				int rr = r + rm;

				// mask coefficient
				// reads mask coefficients in reverse order as scipy.ndimage does
				float coeff = mask[maskLength -1 - (rm + offset)];

				float imgValue = 0.0f;

				// if row position falls within the range of the face
				// perform the convolution as usual
				if(rr >= 0 && rr < height) {
					imgValue = img[rr*width + c];
				} else {			// get image value from belts

					if(rr < 0) { 	// top face (rr < 0)
						int rowB = (beltWidth + rr);		// rr is negative
						imgValue = belt_top[rowB*width + c];
//						cout << "top: [" << rowB << ", " << c << "]: " << imgValue << endl;
					}
					else { 			// bottom face (rr >= height)
						int rowB = height -rr;
						imgValue = belt_bottom[rowB*width + c];
//						cout << "bottom: [" << rowB << ", " << c << "]: " << imgValue << endl;
					}
//					imgValue = 0.0f;
//					cout << "[" << coeff << ", " << imgValue << ", " << convSum << "] ";
				}


				convSum += coeff*imgValue;
//				convSum += imgValue;
			}
//			cout << endl;

			// store the convolution value
			img_output[r*width + c] = convSum;
		}
	}
}
