//#########################################################
// UTILITY METHODS
//#########################################################

/**
 * \brief reads a Vector3f from an image
 */
template<bool checkCoordinates>
Eigen::Vector3f readVector3f(const Image<float>& image, const int row, const int col) {

    if (checkCoordinates) {
        if (!image.checkCoordinates(row, col)) {
            std::cerr << "readVector3f(): coordinates out of bounds: ("
                      << row << ", " << col << ") shape [" << image.height() << ", " << image.width() << "]" << std::endl;
        }
        if (image.depth() != 3) {
            std::cerr << "readVector3f(): depth should be 3, got " << image.depth() << std::endl;
        }
    }

    const size_t offset = row * image.rowLength() + col * image.depth();
    return Eigen::Vector3f(image[offset], image[offset + 1], image[offset + 2]);
}

/**
 * \brief writes a Vector3f to an image
 */
template<bool checkCoordinates>
void writeVector3f(const Eigen::Vector3f& v, const int row, const int col, Image<float>& image) {

    if (checkCoordinates) {
        if (!image.checkCoordinates(row, col)) {
            std::cerr << "writeVector3f(): coordinates out of bounds: ("
                      << row << ", " << col << ") shape [" << image.height() << ", " << image.width() << "]" << std::endl;
        }
        if (image.depth() != 3) {
            std::cerr << "writeVector3f(): depth should be 3, got " << image.depth() << std::endl;
        }
    }

    const size_t offset = row * image.rowLength() + col * image.depth();

    image[offset] = v.x();
    image[offset + 1] = v.y();
    image[offset + 2] = v.z();
}

/**
 * \brief reads a Vector2f from an image
 */
template<bool checkCoordinates>
Eigen::Vector2f readVector2f(const Image<float>& image, const int row, const int col) {

    if (checkCoordinates) {
        if (!image.checkCoordinates(row, col)) {
            std::cerr << "readVector2f(): coordinates out of bounds: ("
                      << row << ", " << col << ") shape [" << image.height() << ", " << image.width() << "]" << std::endl;
        }
        if (image.depth() != 2) {
            std::cerr << "readVector2f(): depth should be 2, got " << image.depth() << std::endl;
        }
    }

    const size_t offset = row * image.rowLength() + col * image.depth();

    return Eigen::Vector2f(image[offset], image[offset + 1]);
}

/**
 * \brief writes a Vector2f to an image
 */
template<bool checkCoordinates>
void writeVector2f(const Eigen::Vector2f& v, const int row, const int col, Image<float>& image) {

    if (checkCoordinates) {
        if (!image.checkCoordinates(row, col)) {
            std::cerr << "writeVector2f(): coordinates out of bounds: ("
                      << row << ", " << col << ") shape [" << image.height() << ", " << image.width() << "]" << std::endl;
        }
        if (image.depth() != 2) {
            std::cerr << "writeVector2f(): depth should be 2, got " << image.depth() << std::endl;
        }
    }

    const size_t offset = row * image.rowLength() + col * image.depth();
    
    image[offset] = v.x();
    image[offset + 1] = v.y();
}



//#########################################################
// INTERPOLATION METHODS
//#########################################################

template<typename T>
void interpolate(const Image<T>& img, const float row, const float col, std::vector<T>& out) {

    // TODO
    int rowInt = rint(row);
    int colInt = rint(col);
    if(rowInt >= 0 && rowInt < img.height() && colInt >= 0 && colInt < img.width()) {
        for(int d = 0; d < img.depth(); d ++) {
            out[d] = img(rowInt, colInt, d);
        }    
    }
}

template<typename T>
std::vector<T> interpolate(const Image<T>& img, const float row, const float col) {

    // TODO
    std::vector<T> out(img.depth());
    int rowInt = rint(row);
    int colInt = rint(col);
    if(rowInt >= 0 && rowInt < img.height() && colInt >= 0 && colInt < img.width()) {
        for(int d = 0; d < img.depth(); d ++) {
            out[d] = img(rowInt, colInt, d);
        }
    }
    return out;
}

template<typename T>
T interpolate(const Image<T>& img, const float row, const float col) {

    // TODO
    int rowInt = rint(row);
    int colInt = rint(col);
    if(rowInt >= 0 && rowInt < img.height() && colInt >= 0 && colInt < img.width()) {
        return img(rowInt, colInt);
    } else {
        return 0;
    }
}

template<typename T>
void interpolateImage(const Image<T>& inputImage, const Image<float>& coordinates,
                      Image<T>& outputImage) {

    // clean output
    outputImage.fill(0);
    
    std::vector<T> pixelValue(inputImage.depth());

    for (int r = 0; r < coordinates.height(); r ++) {
        for (int c = 0; c < coordinates.width(); c ++) {
            Eigen::Vector2f beta = readVector2f(coordinates, r, c);
            float br = beta.y();
            float bc = beta.x();

            if (br >= 0 && br < inputImage.height() && bc >= 0 && bc < inputImage.width()) {
                interpolate(inputImage, br, bc, pixelValue);
                outputImage.set(r, c, pixelValue);
            }
        }
    }
}


//#########################################################
// CONVOLUTION METHODS
//#########################################################

template<typename T>
Image<T> convolve2D(const Image<T>& img,
                    const Image<T>& mask) {

    const int height = img.height();
    const int width = img.width();
    const int depth = img.depth();

    Image<T> output(height, width, depth);
    
    convolve2D(img, mask, output);
    return output;
}

template<typename T>
void convolve2D(const Image<T>& img, const Image<T>& mask, Image<T>& output) {

    const int height = img.height();
    const int width = img.width();
    const int depth = img.depth();

    const int MHeight = mask.height();
    const int MWidth = mask.width();

    // clean output
    output.fill(0);

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
}

template<typename T>
Image<T> convolveRow(const Image<T>& img,
                     const Image<T>& mask) {

    const int height = img.height();
    const int width = img.width();
    const int depth = img.depth();

    Image<T> output(height, width, depth);

    convolveRow(img, mask, output);
    return output;
}

template<typename T>
void convolveRow(const Image<T>& img, const Image<T>& mask, Image<T>& output) {

    const int height = img.height();
    const int width = img.width();
    const int depth = img.depth();

    const int Mhalf = mask.width() / 2;

    // clear output
    output.clear();

    // pixel value
    std::vector<T> pixelValue(depth);

    // convolution sum
    std::vector<T> convolutionSum(depth);

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
}

template<typename T>
Image<T> convolveColumn(const Image<T>& img,
                        const Image<T>& mask) {

    const int height = img.height();
    const int width = img.width();
    const int depth = img.depth();
    
    Image<T> output(height, width, depth);

    convolveColumn(img, mask, output);
    return output;
}

template<typename T>
void convolveColumn(const Image<T>& img, const Image<T>& mask, Image<T>& output) {

    const int height = img.height();
    const int width = img.width();
    const int depth = img.depth();

    const int Mhalf = mask.width() / 2;

    // clear output
    output.clear();

    // pixel value
    std::vector<T> pixelValue(depth);

    // convolution sum
    std::vector<T> convolutionSum(depth);

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
}