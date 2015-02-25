#ifndef __RUNTIMEGL_H__
#define __RUNTIMEGL_H__
#include <cstdio>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
// OpenGL Graphics includes
#include "GL/glew.h"
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include "GL/freeglut.h"
#endif
#include <cuda_gl_interop.h>

#define SDK_CHECK_ERROR_GL() if (!sdkCheckErrorGL(__FILE__, __LINE__)) exit(EXIT_FAILURE);
inline bool sdkCheckErrorGL(const char *file, const int line)
{
	bool ret_val = true;
	// check for error
	GLenum gl_error = glGetError();
	if (gl_error != GL_NO_ERROR)
	{
#ifdef _WIN32
		char tmpStr[512];
		// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line when the user double clicks on the error line in the Output pane. Like any compile error.
		sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, gluErrorString(gl_error));
		fprintf(stderr, "%s", tmpStr);
#endif
		fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
		fprintf(stderr, "%s\n", gluErrorString(gl_error));
		ret_val = false;
	}
	return ret_val;
}

#endif // __RUNTIMEGL_H__