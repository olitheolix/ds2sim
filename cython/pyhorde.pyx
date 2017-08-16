from collections import namedtuple
cimport pyhorde


H3DRenderDevice = namedtuple('H3DRenderDevice', 'OpenGL2 OpenGL4')
H3DModelUpdateFlags = namedtuple('H3DModelUpdateFlags', 'Animation Geometry')

H3DEmitter = namedtuple(
    'H3DEmmitter',
    (
        'MatResI '
        'PartEffResI '
        'MaxCountI '
        'RespawnCountI '
        'DelayF '
        'EmmissionRateF '
        'SpreadAngleF '
        'ForceF3 '
    )
)


H3DPartEffRes = namedtuple(
    'H3DPartEffRes',
    (
	    'ParticleElem '
	    'ChanMoveVelElem '
	    'ChanRotVelElem '
	    'ChanSizeElem '
	    'ChanColRElem '
	    'ChanColGElem '
	    'ChanColBElem '
	    'ChanColAElem '
	    'PartLifeMinF '
	    'PartLifeMaxF '
	    'ChanStartMinF '
	    'ChanStartMaxF '
	    'ChanEndRateF '
	    'ChanDragElem '
    )
)

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
        assert h3dInit(OpenGL2) is True

        # Root node to Horde3D scene.
        self.h3dRootNode = H3DRootNode

        # Expose constants to Python.
        self.h3dRenderDevice = H3DRenderDevice(OpenGL2, OpenGL4)
        self.h3dModelUpdateFlags = H3DModelUpdateFlags(Animation, Geometry)
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

        self.h3dPartEffRes = H3DPartEffRes(
	        ParticleElem,
	        ChanMoveVelElem,
	        ChanRotVelElem,
	        ChanSizeElem,
	        ChanColRElem,
	        ChanColGElem,
	        ChanColBElem,
	        ChanColAElem,
	        PartLifeMinF,
	        PartLifeMaxF,
	        ChanStartMinF,
	        ChanStartMaxF,
	        ChanEndRateF,
	        ChanDragElem,
        )

        self.h3dEmitter = H3DEmitter(
            MatResI,
            PartEffResI,
            MaxCountI,
            RespawnCountI,
            DelayF,
            EmissionRateF,
            SpreadAngleF,
            ForceF3,
        )

    def __dealloc__(self):
        if self.eglDpy != NULL:
            h3dRelease()
            releaseEGL(self.eglDpy)
            self.eglDpy = NULL

    def shutdown(self):
        if self.eglDpy != NULL:
            h3dRelease()
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

    def h3dAddCameraNode(self, H3DNode parent, str name, H3DRes pipelineRes):
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

    def h3dResizePipelineBuffers(self, H3DRes pipeRes, int width, int height):
        return h3dResizePipelineBuffers(pipeRes, width, height)

    def h3dSetupCameraView(self, H3DNode cameraNode, float fov, float aspect,
                        float nearDist, float farDist):
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

    def h3dScreenshot(self, unsigned char[:] ui8buf):
        cdef int width, height;
        h3dutGetScreenshotParam(&width, &height)
        assert len(ui8buf) == width * height * 3
        return h3dutScreenshotRaw(&ui8buf[0], len(ui8buf))

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

    def h3dFindNodes(self, H3DNode startNode, str name, int rtype):
        tmp = name.encode('utf8')
        cdef char *c_name = tmp
        return h3dFindNodes(startNode, c_name, rtype)

    def h3dGetNodeFindResult(self, int index):
        return h3dGetNodeFindResult(index)

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

    def h3dCheckNodeVisibility(self, H3DNode node, H3DNode camera):
        return h3dCheckNodeVisibility(node, camera, True, False)

    def h3dSetMaterialUniform(self, H3DRes res, str name,
                              float a, float b, float c, float d):
        tmp = name.encode('utf8')
        cdef char *c_name = tmp
        h3dSetMaterialUniform(res, c_name, a, b, c, d)

    def h3dAddEmitterNode(self, H3DNode parent, str name, H3DRes materialRes,
                          H3DRes particleEffectRes, int maxParticleCount,
                          int respawnCount):
        tmp = name.encode('utf8')
        cdef char *c_name = tmp
        return h3dAddEmitterNode(
            parent, c_name, materialRes, particleEffectRes,
            maxParticleCount, respawnCount)

    def h3dUpdateEmitter(self, H3DNode emitterNode, float timeDelta):
        h3dUpdateEmitter(emitterNode, timeDelta)

    def h3dHasEmitterFinished(self, H3DNode emitterNode):
        return h3dHasEmitterFinished(emitterNode)

    def h3dUpdateModel(self, H3DNode modelNode, int flags):
        h3dUpdateModel(modelNode, flags)

    def h3dSetupModelAnimStage(self, H3DNode modelNode, int stage, H3DRes animationRes,
                               int layer, str startNode, bint additive):
        tmp = startNode.encode('utf8')
        cdef char *c_startNode = tmp
        return h3dSetupModelAnimStage(
            modelNode, stage, animationRes, layer, c_startNode, additive)
