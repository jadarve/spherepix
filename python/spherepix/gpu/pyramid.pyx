"""
    spherepix.gpu.pyramid
    ---------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""


cimport spherepix.gpu.pixelation as gpix
import spherepix.gpu.pixelation as gpix

cdef class ImagePyramid:
    

    def __init__(self, gpix.PixelationFace face,
        gpix.PixelationFaceImage inputImage=None,
        int levels=1):
        """Creates a new GPU ImagePyramid instance.

        If inputImage is set, then the constructor also
        performs a call to configure() to configure the
        pipeline stage.

        Parameters
        ----------

        face : PixelationFace.
            Pixelation face.

        inputImage : PixelationFaceImage, optional.
            Input image from which pyramid levels are computed.
            Defaults to None.

        levels : int, optional.
            Number of pyramid levels to compute. Defaults to 1.
        """


    def __cinit__(self, gpix.PixelationFace face,
        gpix.PixelationFaceImage inputImage=None,
        int levels=1):
        
        self.pyr = ImagePyramid_cpp(face.face, inputImage.img, levels)


    def configure(self):

        self.pyr.configure()


    def compute(self):
        
        self.pyr.compute()


    def elapsedTime(self):

        return self.pyr.elapsedTime()


    def setInputImage(self, gpix.PixelationFaceImage img):

        self.pyr.setInputImage(img.img)


    def getImage(self, int level):

        cdef gpix.PixelationFaceImage img = gpix.PixelationFaceImage()
        img.img = self.pyr.getImage(level)
        return img


    property levels:
        def __get__(self):
            return self.pyr.getLevels()

        def __set__(self, value):
            self.pyr.setLevels(value)

        def __del__(self):
            pass
