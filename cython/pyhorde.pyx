from collections import namedtuple
cimport pyhorde


H3DOptions = namedtuple(
    'H3DOptions',
    (
        'MaxLogLevel '
        'MaxNumMessages '
        'TrilinearFiltering '
        'MaxAnisotropy '
        'TexCompression '
        'SRGBLinearization '
        'LoadTextures '
        'FastAnimation '
        'ShadowMapSize '
        'SampleCount '
        'WireframeMode '
        'DebugViewMode '
        'DumpFailedShaders '
        'GatherTimeStats'
    )
)

H3DResTypes = namedtuple(
    'H3DResTypes',
    (
        'Undefined '
        'SceneGraph '
        'Geometry '
        'Animation '
        'Material '
        'Code '
        'Shader '
        'Texture '
        'ParticleEffect '
	    'Pipeline'
    )
)

H3DNodeTypes = namedtuple(
    'H3DNodeTypes',
    (
        'Undefined '
        'Group '
        'Model '
        'Mesh '
        'Joint '
        'Light '
        'Camera '
        'Emitter'
    )
)

H3DNodeFlags = namedtuple(
    'H3DNodeFlags',
    (
        'NoDraw '
        'NoCastShadow '
        'NoRayQuery '
        'Inactive'
    )
)

H3DLight = namedtuple(
    'H3DLight',
    (
        'MatResI '
        'RadiusF '
        'FovF '
        'ColorF3 '
        'ColorMultiplierF '
        'ShadowMapCountI '
        'ShadowSplitLambdaF '
        'ShadowMapBiasF '
        'LightingContextStr '
        'ShadowContextStr '
    )
)

H3DCamera = namedtuple(
    'H3DCamera',
    (
        'PipeResI '
        'OutTexResI '
        'OutBufIndexI '
        'LeftPlaneF '
        'RightPlaneF '
        'BottomPlaneF '
        'TopPlaneF '
        'NearPlaneF '
        'FarPlaneF '
        'ViewportXI '
        'ViewportYI '
        'ViewportWidthI '
        'ViewportHeightI '
        'OrthoI '
        'OccCullingI '
    )
)


cdef class PyHorde3D:
    def __init__(self, width=512, height=512):
        # The actual engine (will not be created in this ctor but a dedicated
        # method since EGL needs to be initialised first).
        self.eglDpy = NULL

        # Create OpenGL context and initialise Horde. The initial resolution of
        # 128x128 is just because we need an initial resolution. We may
        # afterwards change the resolution freely.
        self.eglDpy = initEGL(width, height)
        assert self.eglDpy != NULL
        assert h3dInit() is True

        # Root node to Horde3D scene.
        self.h3dRootNode = H3DRootNode

        # Expose constants to Python.
        self.h3dOptions = H3DOptions(
            MaxLogLevel,
            MaxNumMessages,
            TrilinearFiltering,
            MaxAnisotropy,
            TexCompression,
            SRGBLinearization,
            LoadTextures,
            FastAnimation,
            ShadowMapSize,
            SampleCount,
            WireframeMode,
            DebugViewMode,
            DumpFailedShaders,
            GatherTimeStats
        )
        self.h3dResTypes = H3DResTypes(
            Undefined,
            SceneGraph,
            Geometry,
            Animation,
            Material,
            Code,
            Shader,
            Texture,
            ParticleEffect,
	        Pipeline,
        )

        self.h3dNodeTypes = H3DNodeTypes(
            Undefined,
            Group,
            Model,
            Mesh,
            Joint,
            Light,
            Camera,
            Emitter,
        )

        self.h3dNodeFlags = H3DNodeFlags(
            NoDraw,
            NoCastShadow,
            NoRayQuery,
            Inactive,
        )

        self.h3dLight = H3DLight(
	        MatRes,
	        RadiusF,
	        FovF,
	        ColorF3,
	        ColorMultiplierF,
	        ShadowMapCountI,
	        ShadowSplitLambdaF,
	        ShadowMapBiasF,
	        LightingContextStr,
	        ShadowContextStr,
        )

        self.h3dCamera = H3DCamera(
		    PipeResI,
		    OutTexResI,
		    OutBufIndexI,
		    LeftPlaneF,
		    RightPlaneF,
		    BottomPlaneF,
		    TopPlaneF,
		    NearPlaneF,
		    FarPlaneF,
		    ViewportXI,
		    ViewportYI,
		    ViewportWidthI,
		    ViewportHeightI,
		    OrthoI,
		    OccCullingI,
        )

    def __dealloc__(self):
        h3dRelease()
        if self.eglDpy != NULL:
            releaseEGL(self.eglDpy)

    def h3dScreenshotFile(self, str fname):
        cdef string c_fname = fname.encode('utf8')
        return h3dutScreenshot(c_fname.c_str())

    def h3dSetOption(self, ListH3DOptions param, float value):
        return h3dSetOption(param, value)

    def h3dAddResource(self, int rtype, str name, int flags):
        cdef string s = name.encode('utf8')
        return h3dAddResource(rtype, s.c_str(), flags)

    def h3dAddNode(self, H3DNode parent, H3DRes sceneGraphRes):
        return h3dAddNodes(parent, sceneGraphRes)

    def h3dRemoveNode(self, H3DNode node):
        h3dRemoveNode(node)

    def h3dAddCameraNode(self, H3DNode parent, str name, H3DRes pipelineRes ):
        cdef string s = name.encode('utf8')
        return h3dAddCameraNode(parent, s.c_str(), pipelineRes)

    def h3dSetNodeTransform(self, H3DNode node, float tx, float ty, float tz,
                         float rx, float ry, float rz, float sx, float sy,
                         float sz):
        return h3dSetNodeTransform(node, tx, ty, tz, rx, ry, rz, sx, sy, sz)

    def h3dSetNodeTransMat(self, H3DNode node, float[:] mat44):
        return h3dSetNodeTransMat(node, &mat44[0])

    def h3dGetNodeTransform(self, H3DNode node):
        cdef float v[9];
        h3dGetNodeTransform(
            node, &(v[0]), &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8])
        return [float(v[_]) for _ in range(9)]

    def h3dGetNodeTransMat(self, H3DNode node):
        # Horde does not copy the TM into an array of our choosing, but gives
        # us a pointer to its the internal data structure.
        cdef const float *ptr
        h3dGetNodeTransMats(node, &ptr, NULL)
        return <unsigned char[:16 * sizeof(float)]><unsigned char*>(&(ptr[0]))

    def h3dAddLightNode(self, H3DNode parent,
                     str name,
                     H3DRes materialRes,
                     str lightingContext,
                     str shadowContext):
        cdef string cname = name.encode('utf8')
        cdef string clc = lightingContext.encode('utf8')
        cdef string csc = shadowContext.encode('utf8')

        return h3dAddLightNode(parent, cname.c_str(),
                               materialRes, clc.c_str(), csc.c_str())

    def h3dSetNodeParamI(self, H3DNode node, int param, int value):
        return h3dSetNodeParamI(node, param, value)

    def h3dSetNodeParamF(self, H3DNode node, int param, int compIdx, float value):
        return h3dSetNodeParamF(node, param, compIdx, value)

    def h3dGetNodeParamI(self, H3DNode node, int param):
        return h3dGetNodeParamI(node, param)

    def h3dGetNodeParamF(self, H3DNode node, int param, int compIdx):
        return h3dGetNodeParamF(node, param, compIdx)

    def h3dClearOverlays(self):
        return h3dClearOverlays()

    def h3dFinalizeFrame(self):
        return h3dFinalizeFrame()

    def h3dRender(self, H3DNode cameraNode):
        return h3dRender(cameraNode)

    def h3dResizePipelineBuffers(self, H3DRes pipeRes, int width, int height ):
        return h3dResizePipelineBuffers(pipeRes, width, height)

    def h3dSetupCameraView(self, H3DNode cameraNode, float fov, float aspect,
                        float nearDist, float farDist ):
        return h3dSetupCameraView(cameraNode, fov, aspect,
                        nearDist, farDist)

    def h3dUtLoadResourcesFromDisk(self, str contentDir):
        cdef string ccd = contentDir.encode('utf8')
        return h3dutLoadResourcesFromDisk(ccd.c_str())

    def h3dUtDumpMessages(self):
        return h3dutDumpMessages()

    def h3dScreenshotDimensions(self):
        cdef int width, height;
        h3dutGetScreenshotParam(&width, &height)
        return width, height

    def h3dScreenshot(self, float[:] f32buf, unsigned char[:] ui8buf):
        cdef int width, height;
        h3dutGetScreenshotParam(&width, &height)
        assert len(f32buf) == width * height * 4
        assert len(ui8buf) == width * height * 3

        return h3dutScreenshotRaw(
            <char*?>(&f32buf[0]), len(f32buf) * sizeof(float),
            &ui8buf[0], len(ui8buf)
        )

    def h3dCloneResource(self, H3DRes res, str name):
        cdef string c_name = name.encode('utf8')
        h3dCloneResource(res, c_name.c_str())

    def h3dSetResParamI(self, H3DRes res, int elem, int elemIdx, int param, int value):
        h3dSetResParamI(res, elem, elemIdx, param, value)

    def h3dLoadResource(self, H3DRes res, bytes data):
        return h3dLoadResource(res, data, len(data))

    def h3dFindResource(self, int rtype, str name):
        tmp = name.encode('utf8')
        cdef char *c_name = tmp
        return h3dFindResource(rtype, c_name)

    def h3dGetNextResource(self, rtype, H3DRes start):
        return h3dGetNextResource(rtype, start)

    def h3dFindNodes(self, H3DNode startNode, char *name, int rtype):
        tmp = name.encode('utf8')
        cdef char *c_name = tmp
        return h3dFindNodes(startNode, c_name, rtype)

    def h3dGetNodeChild(self, H3DNode node, int index):
        return h3dGetNodeChild(node, index)

    def h3dGetNodeParent(self, H3DNode node):
        return h3dGetNodeParent(node)

    def h3dSetNodeParent(self, H3DNode node, H3DNode parent) -> bool:
        return h3dSetNodeParent(node, parent)

    def h3dGetNodeType(self, H3DNode node):
        return h3dGetNodeType(node)

    def h3dSetNodeFlags(self, H3DNode node, int flags, bint recursive):
        h3dSetNodeFlags(node, flags, recursive)
