from libc.stdint cimport uint8_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map


cdef extern from "horde3d/Horde3D.h":
    ctypedef int H3DRes
    ctypedef int H3DNode

    ctypedef enum ListH3DOptions "H3DOptions::List":
        MaxLogLevel        "H3DOptions::MaxLogLevel"
        MaxNumMessages     "H3DOptions::MaxNumMessages"
        TrilinearFiltering "H3DOptions::TrilinearFiltering"
        MaxAnisotropy      "H3DOptions::MaxAnisotropy"
        TexCompression     "H3DOptions::TexCompression"
        SRGBLinearization  "H3DOptions::SRGBLinearization"
        LoadTextures       "H3DOptions::LoadTextures"
        FastAnimation      "H3DOptions::FastAnimation"
        ShadowMapSize      "H3DOptions::ShadowMapSize"
        SampleCount        "H3DOptions::SampleCount"
        WireframeMode      "H3DOptions::WireframeMode"
        DebugViewMode      "H3DOptions::DebugViewMode"
        DumpFailedShaders  "H3DOptions::DumpFailedShaders"
        GatherTimeStats    "H3DOptions::GatherTimeStats"

    ctypedef enum ListH3DResTypes "H3DResTypes::List":
        Undefined      "H3DResTypes::Undefined"
        SceneGraph     "H3DResTypes::SceneGraph"
        Geometry       "H3DResTypes::Geometry"
        Animation      "H3DResTypes::Animation"
        Material       "H3DResTypes::Material"
        Code           "H3DResTypes::Code"
        Shader         "H3DResTypes::Shader"
        Texture        "H3DResTypes::Texture"
        ParticleEffect "H3DResTypes::ParticleEffect"
        Pipeline       "H3DResTypes::Pipeline"

    ctypedef enum ListH3NodeTypes "H3DNodeTypes::List":
        Undefined  "H3DNodeTypes::Undefined"
        Group      "H3DNodeTypes::Group"
        Model      "H3DNodeTypes::Model"
        Mesh       "H3DNodeTypes::Mesh"
        Joint      "H3DNodeTypes::Joint"
        Light      "H3DNodeTypes::Light"
        Camera     "H3DNodeTypes::Camera"
        Emitter    "H3DNodeTypes::Emitter"

    ctypedef enum ListH3NodeFlags "H3DNodeFlags::List":
        NoDraw        "H3DNodeFlags::NoDraw"
        NoCastShadow  "H3DNodeFlags::NoCastShadow"
        NoRayQuery    "H3DNodeFlags::NoRayQuery"
        Inactive      "H3DNodeFlags::Inactive"

    ctypedef enum ListH3DResTypes "H3DLight::List":
        MatRes              "H3DLight::MatResI"
        RadiusF             "H3DLight::RadiusF"
        FovF                "H3DLight::FovF"
        ColorF3             "H3DLight::ColorF3"
        ColorMultiplierF    "H3DLight::ColorMultiplierF"
        ShadowMapCountI     "H3DLight::ShadowMapCountI"
        ShadowSplitLambdaF  "H3DLight::ShadowSplitLambdaF"
        ShadowMapBiasF      "H3DLight::ShadowMapBiasF"
        LightingContextStr  "H3DLight::LightingContextStr"
        ShadowContextStr    "H3DLight::ShadowContextStr"

    ctypedef enum ListH3DCamera "H3DCamera::List":
        PipeResI           "H3DCamera::PipeResI"
        OutTexResI         "H3DCamera::OutTexResI"
        OutBufIndexI       "H3DCamera::OutBufIndexI"
        LeftPlaneF         "H3DCamera::LeftPlaneF"
        RightPlaneF        "H3DCamera::RightPlaneF"
        BottomPlaneF       "H3DCamera::BottomPlaneF"
        TopPlaneF          "H3DCamera::TopPlaneF"
        NearPlaneF         "H3DCamera::NearPlaneF"
        FarPlaneF          "H3DCamera::FarPlaneF"
        ViewportXI         "H3DCamera::ViewportXI"
        ViewportYI         "H3DCamera::ViewportYI"
        ViewportWidthI     "H3DCamera::ViewportWidthI"
        ViewportHeightI    "H3DCamera::ViewportHeightI"
        OrthoI             "H3DCamera::OrthoI"
        OccCullingI        "H3DCamera::OccCullingI"


    const H3DNode H3DRootNode

    bint h3dInit()
    void h3dRelease()
    bint h3dSetOption(ListH3DOptions param, float value)
    H3DRes h3dFindResource(int type, char *name)
    H3DRes h3dGetNextResource(int type, H3DRes start)
    H3DRes h3dAddResource(int type, const char *name, int flags)
    H3DRes h3dCloneResource(H3DRes res, char *name)
    void h3dSetResParamI(H3DRes res, int elem, int elemIdx, int param, int value)
    void h3dRemoveNode(H3DNode node)
    H3DNode h3dAddNodes(H3DNode parent, H3DRes sceneGraphRes)
    void h3dSetNodeFlags(H3DNode node, int flags, bint recursive)
    int h3dFindNodes(H3DNode startNode, char *name, int type)
    int h3dGetNodeType(H3DNode node)
    H3DNode h3dGetNodeChild(H3DNode node, int index)
    bint h3dSetNodeParent(H3DNode node, H3DNode parent)
    H3DNode h3dGetNodeParent(H3DNode node)
    H3DNode h3dAddCameraNode(H3DNode parent, const char *name,
                             H3DRes pipelineRes)
    void h3dSetNodeTransform(H3DNode node, float tx, float ty, float tz,
                             float rx, float ry, float rz, float sx, float sy,
                             float sz)
    void h3dGetNodeTransform(H3DNode node, float *tx, float *ty, float *tz,
                             float *rx, float *ry, float *rz, float *sx, float *sy,
                             float *sz)
    void h3dGetNodeTransMats(H3DNode node, float **relMat, float **absMat)
    void h3dSetNodeTransMat(H3DNode, const float *mat4x4)
    H3DNode h3dAddLightNode(H3DNode parent, const char *name,
                            H3DRes materialRes,
                            const char *lightingContext,
                            const char *shadowContext)

    void h3dSetNodeParamI( H3DNode node, int param, int value )
    int h3dGetNodeParamI( H3DNode node, int param )
    float h3dGetNodeParamF( H3DNode node, int param, int compIdx )
    void h3dSetNodeParamF( H3DNode node, int param, int compIdx, float value )

    void h3dClearOverlays()
    void h3dFinalizeFrame()
    void h3dRender( H3DNode cameraNode )
    void h3dResizePipelineBuffers( H3DRes pipeRes, int width, int height )
    void h3dSetupCameraView( H3DNode cameraNode, float fov, float aspect,
                             float nearDist, float farDist )


cdef extern from "horde3d/Horde3DUtils.h":
    bint h3dutScreenshot(const char *)
    bint h3dLoadResource(H3DRes res, char *data, int size)
    bint h3dutLoadResourcesFromDisk( const char *contentDir)
    bint h3dutDumpMessages()
    void h3dutGetScreenshotParam( int*,  int* )
    bint h3dutScreenshotRaw(char *, int , unsigned char *, int)


# This _must_ be included _after_ anything that includes protobuf files.
# Strange errors abound during compilation otherwise.
cdef extern from "EGL/egl.h":
    ctypedef void *EGLDisplay

cdef extern from "glutils.hpp":
    EGLDisplay initEGL(int, int)
    void releaseEGL(EGLDisplay &)


cdef class PyHorde3D:
    cdef object keepalive
    cdef EGLDisplay eglDpy
    cdef readonly h3dRootNode
    cdef readonly h3dOptions
    cdef readonly h3dResTypes
    cdef readonly h3dNodeTypes
    cdef readonly h3dNodeFlags
    cdef readonly h3dLight
    cdef readonly h3dCamera
