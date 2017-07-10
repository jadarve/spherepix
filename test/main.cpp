#include <iostream>
#include <fstream>
#include <vector>

#include <spherepix/image.h>
#include <spherepix/camera.h>
#include <spherepix/pixelation.h>
#include <Eigen/Dense>

void numericsTest() {
    Eigen::Vector2f V_total(-1.39383e-05, 7.17674e-06);
    Eigen::Vector2f C_total(-7.66636e-08,  1.24335e-07);
    float M = 1.0f;

    Eigen::Vector2f beta_acc_p = (V_total - C_total) / M;

    std::cout << "beta_acc_p: " << beta_acc_p.transpose() << std::endl;
}

void testMin() {
    std::vector<int> v {7, 8, 5, 2, 0, 9, 10};
    int childrenIndex = std::min_element(v.begin(), v.end()) - v.begin(); 
    std::cout << "min index: " << childrenIndex << std::endl;
    std::cout << "element: " << v[childrenIndex] << std::endl;
}

void tesCamera() {

    std::cout << "camera test" << std::endl;
    spherepix::PinholeCamera cam(35, 480, 640, 18, 18);

    std::cout << "height: " << cam.height() << std::endl;
    std::cout << "width: " << cam.width() << std::endl;

    std::cout << "camera test: finished" << std::endl;
}

void testSpringDynamics() {
    int N = 512;
    spherepix::Image<float> input = spherepix::createFace_equiangular(N);
    spherepix::Image<float> face = spherepix::regularizeCoordinates(input, 0.0f, 1000);

    // spherepix::Image<float> win = input.subImage(0, 0, 3, 3);
    // spherepix::Image<float> cop(3,3,3);
    // cop.copyFrom(win);
    // for(int i =0; i < win.length(); ++i) {
    //     std::cout << i << ": " << win[i] << " : " << cop[i] << std::endl;
    // }

    face.save("pix512_2.bin");

    std::cout << "ALL GOOD" << std::endl;
}

int main(int argc, char const *argv[])
{
    // std::cout << "imageTest: start" << std::endl;

    // spherepix::Image<float> img {2, 5, 3};
    // std::cout << "width: " << img.width() << std::endl;
    // std::cout << "height: " << img.height() << std::endl;
    // std::cout << "depth: " << img.depth() << std::endl;
    // std::cout << "length: " << img.length() << std::endl;
    // std::cout << "pitch: " << img.pitch() << std::endl;

    // for (int i = 0; i < img.length(); ++i) {
    //     img[i] = i;
    //     std::cout << "[" << i << "]: " << img[i] << std::endl;
    // }

    // for (int r = 0; r < img.height(); ++ r) {
    //     for (int c = 0; c < img.width(); ++ c) {
    //         std::cout << "[" << r << ", " << c << "]: " << img(r, c, 0) << std::endl;
    //     }
    // }


    // float* data = new float[100];
    // spherepix::Image<float> imgExt {10, 10, 1, data};

    numericsTest();
    testMin();
    tesCamera();
    // testSpringDynamics();

    spherepix::Pixelation pix = spherepix::PixelationFactory::createPixelation(spherepix::PixelationMode::MODE_1, 64, 3);
    spherepix::SphericalImage<float> img(pix, 3);

    std::cout << "sphImg face height: " << img.faceHeight() << std::endl;

    spherepix::Image<float>* face = &img[0];
    std::cout << "face depth: " << face->depth() << std::endl;

    return 0;
}