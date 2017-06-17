import os
import pyhorde
import numpy as np
import ds2sim.logger


class Engine(pyhorde.PyHorde3D):
    def __init__(self, width, height, fov=45):
        self.logit = ds2sim.logger.getLogger('Horde3d')
        self.fov = fov

        # Initialise Horde.
        super().__init__(width, height)

        self.res_idx = 0
        self.frameCnt = 0

        # No lights upon startup.
        self.lights = {}
        self.models = {}
        self.resources = {}

        # Load default resources.
        self.cam = self.setupHorde()

        self.logit.info('Initialised Horde')

    def _setRenderSize(self, cam, width, height):
        assert width > 0 and height > 0

        self.h3dSetNodeParamI(cam, self.h3dCamera.ViewportXI, 0)
        self.h3dSetNodeParamI(cam, self.h3dCamera.ViewportYI, 0)
        self.h3dSetNodeParamI(cam, self.h3dCamera.ViewportWidthI, width)
        self.h3dSetNodeParamI(cam, self.h3dCamera.ViewportHeightI, height)

        # Set virtual camera parameters
        near, far = 0.1, 5000
        self.h3dSetupCameraView(cam, self.fov, width / height, near, far)
        self.h3dResizePipelineBuffers(self.resources['shader'], width, height)

    def _renderToImage(self):
        self.frameCnt += 1

        # Render the scene.
        self.h3dRender(self.cam)
        self.h3dFinalizeFrame()
        self.h3dClearOverlays()

        # Update log file, in case some
        self.h3dUtDumpMessages()

        # Query the size of the last screenshot, so that we can allocate a
        # buffer into which Horde can copy the data from the GPU. We will need
        # to allocate two buffer. The fact that we need two is an
        # implementation detail. FYI: the floating point one will hold the RGBA
        # image data (float values in [0, 1]). The C++ code will unpack the
        # RGBA data into our second RGB buffer, and convert it to uint8 in the
        # process. Some NumPy magic would have been just as efficient, and one
        # day I will fix it.
        width, height = self.h3dScreenshotDimensions()
        f32buf = np.zeros(width * height * 4, np.float32)
        img_buf = np.zeros(width * height * 3, np.uint8)

        # Request the screenshot data and unpack it into the usual
        # Height x Width x 3 format.
        assert self.h3dScreenshot(f32buf, img_buf)
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :, 0] = img_buf[0::3].reshape(height, width)
        img[:, :, 1] = img_buf[1::3].reshape(height, width)
        img[:, :, 2] = img_buf[2::3].reshape(height, width)
        return img

    def setupHorde(self):
        """Load basic/shared resources, and create the basic models.

        Only basic resource like the shaders, the skybox, and the disk-like
        platform will be loaded here.

        The platform and skybox will also be added to the scene. This suffices
        to render a non-trivial scene immediately afterwards.
        """
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

        # Manually load the resources specified above.
        path = ds2sim.getResourcePath()
        self.resources.clear()
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

        # Add the camera. Note: there will only be a single camera.
        root = self.h3dRootNode
        camera = self.h3dAddCameraNode(root, 'Camera', self.resources['shader'])

        # Add the skybox to the scene.
        self.models['skybox'] = self.h3dAddNode(root, self.resources['sky'])
        self.h3dSetNodeFlags(
            self.models['skybox'], self.h3dNodeFlags.NoCastShadow, True)

        # Update Horde's log file.
        self.h3dUtDumpMessages()
        return camera

    def loadDefaultResources(self):
        """ Load the resources for 10 cubes.

        Each cube has the same geometry, but a different texture, one for each
        digit [0-9]. To respective resource names are eg 'cube_5'.

        NOTE: this is a convenience method to load the resources for the
        platform and all cubes.

        Returns:
            dict: {cube_nr: handle}, where 'cube_nr' is an integer in [0, 9].
        """
        rt = self.h3dResTypes
        path = os.path.join(ds2sim.getResourcePath(), 'models', 'cube')

        fname = os.path.join(path, 'cube.scene.xml')
        scn_cube_xml_template = open(fname, 'r').read()
        fname = os.path.join(path, 'cube.material.xml')
        mat_cube_xml_template = open(fname, 'r').read()
        del fname

        ret = {}
        for i in range(10):
            self.res_idx += 1
            res_name = f'cube_{i}'
            mat_name = f'mat_cube_{i}'
            scn_cube_xml = scn_cube_xml_template.replace(
                'material="cube.material.xml"', f'material="{mat_name}"')
            res = self.h3dAddResource(rt.SceneGraph, res_name, 0)
            self.h3dLoadResource(res, scn_cube_xml.encode('utf8'))
            self.resources[self.res_idx] = res
            ret[i] = res
            del res

            mat_cube_xml = mat_cube_xml_template.replace('0.jpg', f'{i}.jpg')
            res = self.h3dAddResource(rt.Material, mat_name, 0)
            self.h3dLoadResource(res, mat_cube_xml.encode('utf8'))

        # Add the platform resource.
        ret['base'] = self.resources['base']

        if not self.h3dUtLoadResourcesFromDisk(path):
            self.logit.error('Could not load cube resources')
            return {}

        self.logit.info('Loaded all cube resources')
        self.h3dUtDumpMessages()
        return ret

    def _loadDemoScene(self, num_cubes, seed=0):
        """Setup a demo scene.

        This convenience function will populate the scene with an artificial
        sun, a platform, and `num_cubes` randomly chosen cubes.

        Args:
            num_cubes (int): number of cubes to randomly place in the scene
            seed (int): seed value to ensure repeatable scenes.

        Returns:
            None
        """
        # Load the default resource, like the platform, or the various cubes.
        # This will return resource handles. We will still have to add them to
        # the scene.
        default_resources = self.loadDefaultResources()

        # Add a default light. Then place it far away to mimic a sun.
        lname = self.addLight()
        pos = 2000 * np.array([0, 1, 1], np.float32)
        tm = np.eye(4)
        tm[2, :3] = pos / np.linalg.norm(pos)
        tm[3, :3] = pos
        self.setNodeTransMat(lname, tm.flatten().astype(np.float32).tobytes())
        del lname, pos, tm

        # Add the platform.
        node = self.addNode(default_resources['base'])
        self.setNodeTransPes(node, [0, -35, 0], [0, 0, 0], [1, .2, 1])
        del node

        # Initialise the random generator to ensure reproducible results.
        np.random.seed(seed)

        # Draw the number of each cube at random.
        cube_num = np.random.choice(np.arange(10), num_cubes)

        # Create random positions and orientations for each cube.
        cube_pos = 50 * np.random.uniform(-1, 1, size=(num_cubes, 3))
        cube_rot = 180 * np.random.uniform(-1, 1, size=(num_cubes, 3))

        # Each cube has the same size in the scene.
        scale = 2 * np.ones(3)

        # Add each cube to the scene and set its transform.
        for idx, (pos, rot, num) in enumerate(zip(cube_pos, cube_rot, cube_num)):
            node = self.addNode(default_resources[num])
            self.setNodeTransPes(node, pos, rot, scale)

    def addNode(self, resource, parent=None):
        parent = parent or self.h3dRootNode
        return self.h3dAddNode(parent, resource)

    def addLight(self):
        self.res_idx += 1
        root = self.h3dRootNode

        lname = f'Light{self.res_idx}'
        res = self.resources['light']
        light = self.h3dAddLightNode(root, lname, res, "LIGHTING", "SHADOWMAP")
        self.h3dSetNodeParamF(light, self.h3dLight.RadiusF, 0, 4000)
        self.h3dSetNodeParamF(light, self.h3dLight.FovF, 0, 90)
        self.h3dSetNodeParamI(light, self.h3dLight.ShadowMapCountI, 3)
        self.h3dSetNodeParamF(light, self.h3dLight.ShadowSplitLambdaF, 0, 0.9)
        self.h3dSetNodeParamF(light, self.h3dLight.ShadowMapBiasF, 0, 0.001)
        self.h3dSetNodeParamF(light, self.h3dLight.ColorF3, 0, 1.0)
        self.h3dSetNodeParamF(light, self.h3dLight.ColorF3, 1, 1.0)
        self.h3dSetNodeParamF(light, self.h3dLight.ColorF3, 2, 1.0)

        self.lights[self.res_idx] = light
        return light

    def setNodeTransMat(self, node, tm):
        assert isinstance(tm, bytes)
        assert len(tm) == 16 * 4
        return self.h3dSetNodeTransMat(node, np.fromstring(tm, np.float32))

    def setNodeTransPes(self, node, pos, euler, scale):
        assert len(pos) == len(euler) == len(scale) == 3
        self.h3dSetNodeTransform(node, *pos, *euler, *scale)

    def renderScene(self, cmat, width, height, skybox=True):
        assert isinstance(cmat, bytes)

        assert width > 0 and height > 0
        self._setRenderSize(self.cam, width, height)

        # Center the SkyBox at the camera. The scale of the skybox is related
        # to the `far` plane of the camera; it must be less than `far` /
        # sqrt(3) to be fully visible.
        if skybox:
            pos = np.fromstring(cmat, np.float32).tolist()
            pos = pos[-4:-1]
            scale = 3 * [int(0.9 * 5000 / np.sqrt(3))]
        else:
            pos = scale = (0, 0, 0)
        self.h3dSetNodeTransform(self.models['skybox'], *pos, 0, 0, 0, *scale)
        del pos, scale

        # Update the camera position, then render the scene and return the image.
        self.h3dSetNodeTransMat(self.cam, np.fromstring(cmat, np.float32))
        return self._renderToImage()
