#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <memory>
#include <EGL/egl.h>


EGLDisplay initEGL(const int width, const int height) {
  const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
  };

  const EGLint pbufferAttribs[] = {
    EGL_WIDTH, width,
    EGL_HEIGHT, height,
    EGL_NONE,
  };

  // 1. Initialize EGL
  EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  EGLint major, minor;
  eglInitialize(eglDpy, &major, &minor);

  // Print EGL version number.
  printf("EGL version: %d.%d\n", major, minor);

  // 2. Select an appropriate display configuration.
  EGLint numConfigs;
  EGLConfig eglCfg;
  eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

  // 3. Create a surface.
  EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

  // 4. Bind the API.
  eglBindAPI(EGL_OPENGL_API);

  // 5. Create an OpenGL context and make it current.
  EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
  eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

  return eglDpy;
}

void releaseEGL(EGLDisplay &eglDpy) {
  eglTerminate(eglDpy);
}
