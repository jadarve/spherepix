"""
    spherepix.gpu.camera
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

from libcpp.memory cimport shared_ptr

cimport spherepix.camera as scam
import spherepix.camera as scam

cimport spherepix.gpu.pixelation as gpix
import spherepix.gpu.pixelation as gpix


cdef class GPUCamera:
    
    def __cinit__(self, scam.Camera camera):
        self.cam = GPUCamera_cpp(camera.cam)

    def __dealloc__(self):
        pass


    property height:
        def __get__(self):
            return self.cam.height()

        def __set__(self, value):
            raise RuntimeError('camera height cannot be set')

        def __del__(self):
            pass


    property width:
        def __get__(self):
            return self.cam.width()

        def __set__(self, value):
            raise RuntimeError('camera width cannot be set')

        def __del__(self):
            pass


    property shape:
        def __get__(self):
            return (self.height, self.width)

        def __set__(self, value):
            raise RuntimeError('camera shape cannot be set')

        def __del__(self):
            pass


cdef class GPUFaceImageMapper:
    
    def __init__(self, GPUCamera cam, gpix.PixelationFace face, 
        gpix.PixelationFaceImage inputImage=None,
        gpix.PixelationFaceImage faceBetas=None):
        """Creates a new GPU face image mapper for a given camera

        Parameters
        ----------
        cam : GPUCamera
            GPU camera object

        face : PixelationFace
            Pixelation face on which the image will be mapped.

        inputImage : PixelationFaceImage
            Input image to be mapped

        faceBetas : Face interpolation coordinates, optional.
            If different than None, it will use these coordinates
            to map the image and cam can be None,otherwise the 
            provided camera will be used. 
        """

        # nothing to do, self.mapper is constructed at __cinit__()
        pass


    def __cinit__(self, GPUCamera cam, gpix.PixelationFace face,
        gpix.PixelationFaceImage inputImage=None,
        gpix.PixelationFaceImage faceBetas=None):


        if faceBetas is None:
            self.mapper = GPUFaceImageMapper_cpp(cam.cam, face.face)

        else:
            self.mapper = GPUFaceImageMapper_cpp(faceBetas.img, face.face)

        if inputImage != None:
            self.setInputImage(inputImage)
            self.configure()


    def __dealloc__(self):
        # nothing to do
        pass


    def configure(self):
        """Configure pipeline stage.
        """

        self.mapper.configure()


    def compute(self):
        """Computes mapped image.
        """

        self.mapper.compute()


    def elapsedTime(self):
        return self.mapper.elapsedTime()


    def setInputImage(self, gpix.PixelationFaceImage img):
        """Set input image to be mapped.

        Parameters
        ----------
        img : PixelationFaceImage
            GPU image to be mapped by the camera.
        """

        self.mapper.setInputImage(img.img)


    def getMappedImage(self):
        """Returns the mapped image

        Returns
        -------
        output : PixelationFaceImage
            Mapped image. Image buffer allocated in GPU memory space.
        """
        
        cdef gpix.PixelationFaceImage output = gpix.PixelationFaceImage()
        output.img = self.mapper.getMappedImage()
        return output


    def getInterpolationCoordinates(self):
        """Returns interpolation coordinates.

        Returns
        -------
        betas : PixelationFaceImage
            Interpolation coordinates [col, row] for each pixel in the face.
        """

        cdef gpix.PixelationFaceImage betas = gpix.PixelationFaceImage()
        betas.img = self.mapper.getInterpolationCoordinates()
        return betas
