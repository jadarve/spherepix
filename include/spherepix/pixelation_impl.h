
template<typename T>
SphericalImage<T> convolve2D(const Pixelation& pix,
                             const SphericalImage<T>& img,
                             const Image<T>& mask) {

    // output image
    SphericalImage<T> output(pix, img.depth());

    // for each face in the image
    // call convolve row
    for (int faceIndex = 0; faceIndex < pix.faceCount(); faceIndex ++) {
        convolveFace2D(pix, img, faceIndex, mask, output[faceIndex]);
    }

    return output;
}

template<typename T>
SphericalImage<T> convolveRow(const Pixelation& pix,
                              const SphericalImage<T>& img,
                              const Image<T>& mask) {

    // output image
    SphericalImage<T> output(pix, img.depth());

    // for each face in the image
    // call convolve row
    for (int faceIndex = 0; faceIndex < pix.faceCount(); faceIndex ++) {
        convolveFaceRow(pix, img, faceIndex, mask, output[faceIndex]);
    }

    return output;
}

template<typename T>
SphericalImage<T> convolveColumn(const Pixelation& pix,
                                 const SphericalImage<T>& img,
                                 const Image<T>& mask) {

    // output image
    SphericalImage<T> output(pix, img.depth());

    // for each face in the image
    // call convolve row
    for (int faceIndex = 0; faceIndex < pix.faceCount(); faceIndex ++) {
        convolveFaceColumn(pix, img, faceIndex, mask, output[faceIndex]);
    }

    return output;
}


template<typename T>
void convolveFace2D(const Pixelation& pix, const SphericalImage<T>& sphericalImg,
                    const int faceIndex, const Image<T>& mask, Image<T>& output) {


    const int height = sphericalImg.faceHeight();
    const int width = sphericalImg.faceWidth();

    const int MHeight = mask.height();
    const int MWidth = mask.width();

    Image<T> img = sphericalImg[faceIndex];
    const int depth = img.depth();

    // pixel value
    std::vector<T> pixelValue(img.depth());

    // convolution sum
    std::vector<T> convolutionSum(img.depth());

    // convolves the interior region
    for (int r = MHeight / 2; r < height - MHeight / 2; r ++) {
        for (int c = MWidth / 2; c < width - MWidth / 2; c ++) {

            // clear convolution sum
            convolutionSum.assign(img.depth(), 0);

            // scan convolution mask
            for (int rm = -MHeight / 2; rm <= MHeight / 2; rm ++) {
                int rr = r + rm;
                for (int cm = -MWidth / 2; cm <= MWidth / 2; cm ++) {
                    int cc = c + cm;

                    // mask coefficient
                    T coeff = mask(rm + MHeight / 2, cm + MWidth / 2);

                    // convolution sum for each channel
                    img.get(rr, cc, pixelValue);
                    for (int d = 0; d < depth; d ++) {
                        convolutionSum[d] += coeff * pixelValue[d];
                    }
                }
            }

            // set convolution output
            output.set(r, c, convolutionSum);
        }
    } // interior region

    // TODO: top, bottom, left, right regions... :S
}

template<typename T>
void convolveFaceRow(const Pixelation& pix, const SphericalImage<T>& sphericalImg,
                     const int faceIndex, const Image<T>& mask, Image<T>& output) {

    // clear output
    output.clear();

    const int height = sphericalImg.faceHeight();
    const int width = sphericalImg.faceWidth();
    const int Mhalf = mask.width() / 2;

    Image<T> img = sphericalImg[faceIndex];
    const int depth = img.depth();

    // pixel value
    std::vector<T> pixelValue(img.depth());

    // convolution sum
    std::vector<T> convolutionSum(img.depth());

    // convolves the interior region
    for (int r = 0; r < height; r ++) {
        for (int c = Mhalf; c < width - Mhalf; c ++) {

            // clear convolution sum
            convolutionSum.assign(img.depth(), 0);

            // traverses convolution mask
            for (int cm = -Mhalf; cm <= Mhalf; cm ++) {

                int cc = c + cm;

                // mask coefficient
                // T coeff = mask[M - 1 - (cm + Mhalf)];
                T coeff = mask[cm + Mhalf];

                // convolution sum for each channel
                img.get(r, cc, pixelValue);
                for (int d = 0; d < depth; d ++) {
                    convolutionSum[d] += coeff * pixelValue[d];
                }
            }

            // set convolution output
            output.set(r, c, convolutionSum);
        }
    } // interior region

    // connectivity graph
    const Image<int> connGraph = pix.faceConnectivityGraph();

    // left and right images
    const Image<T> imgLeft = sphericalImg[connGraph(faceIndex, FaceNeighbor::LEFT)];
    const Image<T> imgRight = sphericalImg[connGraph(faceIndex, FaceNeighbor::RIGHT)];

    // beta interpolation coordinates
    const int beltWidth = pix.interpolationBeltWidth();
    const Image<float> betasLeft = pix.interpolationCoordinates(faceIndex, FaceNeighbor::LEFT);
    const Image<float> betasRight = pix.interpolationCoordinates(faceIndex, FaceNeighbor::RIGHT);


    //#############
    // left side
    //#############
    for (int r = 0; r < height; r ++) {
        for (int c = 0; c < Mhalf; c ++) {

            // clear convolution sum
            convolutionSum.assign(depth, 0);

            for (int cm = -Mhalf; cm <= Mhalf; cm ++) {

                int cc = c + cm;

                // T coeff = mask[M - 1 - (cm +Mhalf)];
                T coeff = mask[cm + Mhalf];

                if (cc < 0) {

                    // read interpolation coordinates
                    float bRow = betasLeft(r, beltWidth + cc, 0);
                    float bCol = betasLeft(r, beltWidth + cc, 1);

                    // interpolate image value and apply convolution
                    interpolate(imgLeft, bRow, bCol, pixelValue);
                } else {
                    // read from image
                    img.get(r, cc, pixelValue);
                }

                // apply convolution to each channel
                for (int i = 0; i < depth; i ++) {
                    convolutionSum[i] += coeff * pixelValue[i];
                }
            }

            // set convolution output
            output.set(r, c, convolutionSum);
        }
    } // left side

    //#############
    // right side
    //#############
    for (int r = 0; r < height; r ++) {
        for (int c = width - Mhalf - 1; c < width; c ++) {

            // clear convolution sum
            convolutionSum.assign(depth, 0);

            for (int cm = -Mhalf; cm <= Mhalf; cm ++) {

                int cc = c + cm;

                // T coeff = mask[M - 1 - (cm +Mhalf)];
                T coeff = mask[cm + Mhalf];

                if (cc >= width) {

                    // read interpolation coordinates FIXME: check
                    float bRow = betasRight(r, beltWidth - cc - 1, 0);
                    float bCol = betasRight(r, beltWidth - cc - 1, 1);

                    // interpolate image value and apply convolution
                    interpolate(imgRight, bRow, bCol, pixelValue);
                } else {
                    // read from image
                    img.get(r, cc, pixelValue);
                }

                // apply convolution to each channel
                for (int i = 0; i < depth; i ++) {
                    convolutionSum[i] += coeff * pixelValue[i];
                }
            }

            // set convolution output
            output.set(r, c, convolutionSum);
        }
    } // right side
}


template<typename T>
void convolveFaceColumn(const Pixelation& pix, const SphericalImage<T>& sphericalImg,
                        const int faceIndex, const Image<T>& mask, Image<T>& output) {

    // clear output
    output.clear();

    const int height = sphericalImg.faceHeight();
    const int width = sphericalImg.faceWidth();
    const int Mhalf = mask.width() / 2;


    Image<T> img = sphericalImg[faceIndex];
    const int depth = img.depth();

    // pixel value
    std::vector<T> pixelValue(img.depth());

    // convolution sum
    std::vector<T> convolutionSum(img.depth());

    // convolves the interior region
    for (int r = Mhalf; r < height - Mhalf; r ++) {

        // FIXME: check limits
        for (int c = 0; c < width; c ++) {

            // clear convolution sum
            convolutionSum.assign(img.depth(), 0);

            // traverses convolution mask
            for (int rm = -Mhalf; rm <= Mhalf; rm ++) {

                int rr = r + rm;

                // mask coefficient
                // T coeff = mask[M - 1 - (rm + Mhalf)];
                T coeff = mask[rm + Mhalf];

                // convolution sum for each channel
                img.get(rr, c, pixelValue);
                for (int d = 0; d < depth; d ++) {
                    convolutionSum[d] += coeff * pixelValue[d];
                }
            }

            // set convolution output
            output.set(r, c, convolutionSum);
        }
    } //interior region

    // connectivity graph
    const Image<int> connGraph = pix.faceConnectivityGraph();

    // left and right images
    const Image<T> imgTop = sphericalImg[connGraph(faceIndex, FaceNeighbor::TOP)];
    const Image<T> imgBottom = sphericalImg[connGraph(faceIndex, FaceNeighbor::BOTTOM)];

    // beta interpolation coordinates
    const int beltWidth = pix.interpolationBeltWidth();
    const Image<float> betasTop = pix.interpolationCoordinates(faceIndex, FaceNeighbor::TOP);
    const Image<float> betasBottom = pix.interpolationCoordinates(faceIndex, FaceNeighbor::BOTTOM);


    //#############
    // top side
    //#############
    for (int r = 0; r < Mhalf; r ++) {
        for (int c = 0; c < width; c ++) {

            // clear convolution sum
            convolutionSum.assign(depth, 0);

            for (int rm = -Mhalf; rm <= Mhalf; rm ++) {

                int rr = r + rm;

                // T coeff = mask[M - 1 - (rm + Mhalf)];
                T coeff = mask[rm + Mhalf];

                if (rr < 0) {

                    // read interpolation coordinates FIXME: check
                    float bRow = betasTop(beltWidth + rr, c, 0);
                    float bCol = betasTop(beltWidth + rr, c, 1);

                    // interpolate image value and apply convolution
                    interpolate(imgTop, bRow, bCol, pixelValue);
                } else {
                    // read from image
                    img.get(rr, c, pixelValue);
                }

                // apply convolution to each channel
                for (int i = 0; i < depth; i ++) {
                    convolutionSum[i] += coeff * pixelValue[i];
                }
            }

            // set convolution output
            output.set(r, c, convolutionSum);
        }
    } // top region

    //#############
    // bottom side
    //#############
    for (int r = height - Mhalf - 1; r < height; r ++) {
        for (int c = 0; c < width; c ++) {

            // clear convolution sum
            convolutionSum.assign(depth, 0);

            for (int rm = -Mhalf; rm <= Mhalf; rm ++) {

                int rr = r + rm;

                // mask coefficient
                T coeff = mask[rm + Mhalf];

                if (rr >= height) {

                    // read interpolation coordinates FIXME: check
                    float bRow = betasBottom(beltWidth - rr - 1, c, 0);
                    float bCol = betasBottom(beltWidth - rr - 1, c, 1);

                    // interpolate image value and apply convolution
                    interpolate(imgBottom, bRow, bCol, pixelValue);
                } else {
                    // read from image
                    img.get(rr, c, pixelValue);
                }

                // apply convolution to each channel
                for (int i = 0; i < depth; i ++) {
                    convolutionSum[i] += coeff * pixelValue[i];
                }
            }

            // set convolution output
            output.set(r, c, convolutionSum);
        }
    } // bottom region
}