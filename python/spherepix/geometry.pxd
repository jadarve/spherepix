from libcpp cimport bool

cimport eigen
cimport image

cdef extern from 'spherepix/geometry.h' namespace 'spherepix':
    
    #################################################
    # PROJECTION MATRIX TO TANGENT SPACE
    #################################################
    eigen.Matrix3f_cpp projectionMatrix_cpp 'spherepix::projectionMatrix'(
        const eigen.Vector3f_cpp& eta)
    
    
    #################################################
    # TANGENT SPACE PROJECTIONS
    #################################################

    eigen.Vector3f_cpp etaToMu_orthographic_cpp 'spherepix::etaToMu_orthographic'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& eta)

    eigen.Vector3f_cpp etaToMu_perspective_cpp 'spherepix::etaToMu_perspective'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& eta)

    eigen.Vector3f_cpp etaToMu_geodesic_cpp 'spherepix::etaToMu_geodesic'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& eta)

    eigen.Vector3f_cpp etaToMu_cordal_cpp 'spherepix::etaToMu_cordal'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& eta)


    ##################################################
    # RETRACTIONS
    ##################################################

    eigen.Vector3f_cpp muToEta_orthographic_cpp 'spherepix::muToEta_orthographic'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& mu)

    eigen.Vector3f_cpp muToEta_perspective_cpp 'spherepix::muToEta_perspective'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& mu)

    eigen.Vector3f_cpp muToEta_geodesic_cpp 'spherepix::muToEta_geodesic'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& mu)

    eigen.Vector3f_cpp muToEta_cordal_cpp 'spherepix::muToEta_cordal'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& mu)


    ##################################################
    # ROTATION MATRICES
    ##################################################

    eigen.Matrix3f_cpp rotationMatrix_cpp 'spherepix::rotationMatrix'(
        const eigen.Vector3f_cpp& eta_0, const eigen.Vector3f_cpp& eta_1)

    eigen.Matrix3f_cpp rotationMatrixAxisAngle_cpp 'spherepix::rotationMatrixAxisAngle'(
        const eigen.Vector3f_cpp& axis, const float& theta)


    ##################################################
    # ORTHONORMAL COORDINATES
    ##################################################

    eigen.Matrix2Xf_cpp getOrthonormalBasis_cpp 'spherepix::getOrthonormalBasis'(
        const image.Image_cpp[float]& etas, const int row, const int col)

    eigen.Matrix2Xf_cpp getOrthonormalBasis_cpp 'spherepix::getOrthonormalBasis'(
        const image.Image_cpp[float]& etas, const int row, const int col, float& norm_out)

    eigen.Vector2f_cpp muToBetapix_cpp 'spherepix::muToBetapix'(
        const eigen.Matrix2Xf_cpp& B, const float Bnorm, const eigen.Vector3f_cpp& mu);

    eigen.Vector3f_cpp betapixToMu_cpp 'spherepix::betapixToMu'(
        const eigen.Matrix2Xf_cpp& B, const float Bnorm, const eigen.Vector2f_cpp& beta);

    void betapixFieldToMu_cpp 'spherepix::betapixFieldToMu'(
        const image.Image_cpp[float]& etas, const image.Image_cpp[float]& field, image.Image_cpp[float]& output)

    image.Image_cpp[float] betapixFieldToMu_cpp 'spherepix::betapixFieldToMu'(
        const image.Image_cpp[float]& etas, const image.Image_cpp[float]& field)

    void muFieldToBetapix_cpp 'spherepix::muFieldToBetapix'(
        const image.Image_cpp[float]& etas, const image.Image_cpp[float]& field, image.Image_cpp[float]& output)

    image.Image_cpp[float] muFieldToBetapix_cpp 'spherepix::muFieldToBetapix'(
        const image.Image_cpp[float]& etas, const image.Image_cpp[float]& field)


    ##################################################
    # COORDINATES INTERPOLATION
    ##################################################

    eigen.Vector2f_cpp findInterpolationCoodinates_cpp 'spherepix::findInterpolationCoodinates'(
        const eigen.Vector3f_cpp& eta, const image.Image_cpp[float]& etaGrid,
        const bool flipVertical);


    ##################################################
    # UTILITY FUNCTIONS
    ##################################################
    
    void retractField_orthographic_cpp 'spherepix::retractField_orthographic'(
        const image.Image_cpp[float]& etas, const image.Image_cpp[float]& field, image.Image_cpp[float]& output)

    image.Image_cpp[float] retractField_orthographic_cpp 'spherepix::retractField_orthographic'(
        const image.Image_cpp[float]&etas, const image.Image_cpp[float]&field)

    void transformCoordinates_cpp 'spherepix::transformCoordinates'(
        const image.Image_cpp[float]& etas, const eigen.Matrix3f_cpp& T, image.Image_cpp[float]& etas_out)

    image.Image_cpp[float] transformCoordinates_cpp 'spherepix::transformCoordinates'(
        const image.Image_cpp[float]& etas, const eigen.Matrix3f_cpp& T)

    void dotProductField3_cpp 'spherepix::dotProductField3'(
        const image.Image_cpp[float]& field1, const image.Image_cpp[float]& field2, image.Image_cpp[float]& output)

    image.Image_cpp[float] dotProductField3_cpp 'spherepix::dotProductField3'(
        const image.Image_cpp[float]& field1, const image.Image_cpp[float]& field2)

    void crossProductField3_cpp 'spherepix::crossProductField3'(
        const image.Image_cpp[float]& field1, const image.Image_cpp[float]& field2, image.Image_cpp[float]& output)

    image.Image_cpp[float] crossProductField3_cpp 'spherepix::crossProductField3'(
        const image.Image_cpp[float]& field1, const image.Image_cpp[float]& field2)

    void angleBetweenNeighbors_cpp 'spherepix::angleBetweenNeighbors'(
        const image.Image_cpp[float]& etas, image.Image_cpp[float]& theta_out)

    image.Image_cpp[float] angleBetweenNeighbors_cpp 'spherepix::angleBetweenNeighbors'(
        const image.Image_cpp[float]& etas)

    void distanceBetweenNeighbors_cpp 'spherepix::distanceBetweenNeighbors'(
        const image.Image_cpp[float]& etas, image.Image_cpp[float]& distance_out)

    image.Image_cpp[float] distanceBetweenNeighbors_cpp 'spherepix::distanceBetweenNeighbors'(
        const image.Image_cpp[float]& etas)
