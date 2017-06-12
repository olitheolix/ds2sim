import io
import os
import pyhorde
import pyhorde3d_res
import PIL.Image
import numpy as np
import ds2server.ds2logger


class Engine(pyhorde.PyHorde3D):
    def __init__(self, width: int, height: int):
        # Initialise Horde.
        super().__init__()

        self.logit = ds2server.ds2logger.getLogger('Horde3d')

        # Sanity checks.
        assert isinstance(width, int)
        assert isinstance(height, int)

        # No lights upon startup.
        self.lights = {}
        self.frameCnt = 0
        self.geo_ids, self.img_ids, self.xml_ids = set(), set(), set()
        self.models = {}
        self.guidedIDs = {}
        self.skybox_node = None
        self.model_version = 0

        self.cam = self.setupHorde()
        self.resize(width, height)

        self.logit.info('Initialised Horde')

    def setupHorde(self):
        # Global Horde options.
        self.h3dSetOption(self.h3dOptions.LoadTextures, 1)
        self.h3dSetOption(self.h3dOptions.TexCompression, 0)
        self.h3dSetOption(self.h3dOptions.MaxAnisotropy, 4)
        self.h3dSetOption(self.h3dOptions.ShadowMapSize, 2048)
        self.h3dSetOption(self.h3dOptions.FastAnimation, 1)

        # Define the resources that we will load manually.
        rt = self.h3dResTypes
        resources = [
            ('base', rt.SceneGraph, 'models/platform/platform.scene.xml'),
            ('sky', rt.SceneGraph, 'models/skybox/skybox_ds2.scene.xml'),
            ('light', rt.Material, 'materials/light.material.xml'),
            ('shader', rt.Pipeline, 'pipelines/deferred.pipeline.xml'),
        ]
        del rt

        # Manually load the just listed resources.
        path = os.path.dirname(os.path.abspath(pyhorde3d_res.__file__))
        self.resources = {}
        for name, rtype, fname in resources:
            res = self.h3dAddResource(rtype, name, 0)
            self.resources[name] = res
            fname = os.path.join(path, fname)
            self.h3dLoadResource(res, open(fname, 'rb').read())

        # Load all those resources whose name denotes a path (that includes
        # shaders, light materials, etc.)
        if not self.h3dUtLoadResourcesFromDisk(path):
            self.logit.error('Could not load main resources')
        else:
            self.logit.info('Resources loaded')
        del path

        # Add the one and only camera Horde has. The minions will set its
        # camera matrix to render the scenes for all DS2 cameras.
        root = self.h3dRootNode
        camera = self.h3dAddCameraNode(root, 'Camera', self.resources['shader'])

        # Add skybox.
        self.skybox_node = self.h3dAddNode(root, self.resources['sky'])
        self.h3dSetNodeFlags(self.skybox_node, self.h3dNodeFlags.NoCastShadow, True)

        # Add the platform.
        platform = self.h3dAddNode(root, self.resources['base'])
        self.h3dSetNodeTransform(platform, 0, -10, 0, 0, 0, 0, 0.23, 0.23, 0.23)
        self.h3dUtDumpMessages()
        return camera

    def loadCubes(self):
        rt = self.h3dResTypes
        path = os.path.dirname(os.path.abspath(pyhorde3d_res.__file__))
        path = os.path.join(path, 'models', 'cube')

        fname = os.path.join(path, 'cube.scene.xml')
        res = self.h3dAddResource(rt.SceneGraph, 'cube', 0)
        self.h3dLoadResource(res, open(fname, 'rb').read())
        if not self.h3dUtLoadResourcesFromDisk(path):
            self.logit.error('Could not load cube resources')
        else:
            self.logit.info('Loaded cube resource')

        self.model = self.h3dAddNode(self.h3dRootNode, res)

    def addLights(self):
        root = self.h3dRootNode
        for i in range(3):
            lname = f'Light{i}'
            res = self.resources['light']
            light = self.h3dAddLightNode(root, lname, res, "LIGHTING", "SHADOWMAP")
            self.h3dSetNodeParamF(light, self.h3dLight.RadiusF, 0, 55)
            self.h3dSetNodeParamF(light, self.h3dLight.FovF, 0, 90)
            self.h3dSetNodeParamI(light, self.h3dLight.ShadowMapCountI, 3)
            self.h3dSetNodeParamF(light, self.h3dLight.ShadowSplitLambdaF, 0, 0.9)
            self.h3dSetNodeParamF(light, self.h3dLight.ShadowMapBiasF, 0, 0.001)
            self.lights[lname] = light

            tm = np.eye(4).flatten().astype(np.float32).tobytes()
            tm = np.fromstring(tm, np.float32)
            if tm.shape != (16, ):
                self.logit.error(f'Invalid transform data for light <{lname}>')
                continue

            # Update light transformation.
            self.h3dSetNodeTransMat(light, tm)

    def resize(self, width, height):
        self.h3dSetNodeParamI(self.cam, self.h3dCamera.ViewportXI, 0)
        self.h3dSetNodeParamI(self.cam, self.h3dCamera.ViewportYI, 0)
        self.h3dSetNodeParamI(self.cam, self.h3dCamera.ViewportWidthI, width)
        self.h3dSetNodeParamI(self.cam, self.h3dCamera.ViewportHeightI, height)

        # Set virtual camera parameters
        near, far = 0.1, 5000
        self.h3dSetupCameraView(self.cam, 45.0, width / height, near, far)
        self.h3dResizePipelineBuffers(self.resources['shader'], width, height)

    def renderToImage(self):
        self.frameCnt += 1

        # Render scene
        self.h3dRender(self.cam)
        self.h3dFinalizeFrame()
        self.h3dClearOverlays()

        # Tell Horde to write the screenshot to memory.
        self.h3dUtDumpMessages()

        width, height = self.h3dScreenshotDimensions()
        f32buf = np.zeros(width * height * 4, np.float32)
        img_buf = np.zeros(width * height * 3, np.uint8)

        assert self.h3dScreenshot(f32buf, img_buf)
        img = np.zeros((width, height, 3), np.uint8)
        img[:, :, 0] = img_buf[0::3].reshape(width, height)
        img[:, :, 1] = img_buf[1::3].reshape(width, height)
        img[:, :, 2] = img_buf[2::3].reshape(width, height)

        # Compress image to JPEG.
        buf = io.BytesIO()
        img = PIL.Image.fromarray(np.flipud(img))
        img.save(buf, 'jpeg', quality=90)

        buf.seek(0)
        return buf.read()

    def renderScene(self, cmat, width, height):
        assert isinstance(cmat, bytes)

        # Center the SkyBox at the camera. The scale of the skybox is related
        # to the `far` plane of the camera; it must be less than `far` /
        # sqrt(3) to be fully visible.
        pos = np.fromstring(cmat, np.float32).tolist()
        pos = pos[-4:-1]
        scale = 3 * [int(0.9 * 5000 / np.sqrt(3))]
        self.h3dSetNodeTransform(self.skybox_node, *pos, 0, 0, 0, *scale)
        del pos, scale

        # Update the camera position.
        self.h3dSetNodeTransMat(self.cam, np.fromstring(cmat, np.float32))

        return self.renderToImage()
