#ifdef VISUAL
#include <cstdlib>
#include "VisualHost.h"

float Visual::RotateX;
float Visual::RotateY;
float Visual::TranslateX;
float Visual::TranslateY;
float Visual::TranslateZ;
//
IVisualRender *Visual::_render;
int Visual::_mouseLastX;
int Visual::_mouseLastY;
int Visual::_mouseState;
//StopWatchInterface *Visual::_timer;
int Visual::_fpsCount; // FPS count for averaging
int Visual::_fpsLimit; // FPS limit for sampling
float Visual::_avgFps;

void Visual::Display()
{
	//sdkStartTimer(&_timer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_render->Display();
	//sdkStopTimer(&_timer);
	glutSwapBuffers();
	glutPostRedisplay();
	ComputeFPS();
}

void Visual::Keyboard(unsigned char key, int, int)
{
	_render->Keyboard(key);
	switch (key)
	{
	case 27:
		glutLeaveMainLoop();
		break;
	}
}

void Visual::Mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
		_mouseState |= 1 << button;
	else if (state == GLUT_UP)
		_mouseState = 0;
	_mouseLastX = x;
	_mouseLastY = y;
}

void Visual::Motion(int x, int y)
{
	float dx = (float)(x - _mouseLastX);
	float dy = (float)(y - _mouseLastY);
	if (_mouseState & 2)
	{
		RotateX += dy * 0.2f;
		RotateY += dx * 0.2f;
	}
	if (_mouseState & 1)
	{
		TranslateX += dx * 1.5f;
		TranslateY += dy * -1.5f;
	}
	if (_mouseState & 4)
		TranslateZ += dy * -0.2f;
	_mouseLastX = x;
	_mouseLastY = y;
}

void Visual::TimerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);
}

void Visual::ComputeFPS()
{
	_fpsCount++;
	if (_fpsCount == _fpsLimit)
	{
		_avgFps = 1.f; //1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		_fpsCount = 0;
		_fpsLimit = (int)(_avgFps > 1.f ? _avgFps : 1.f);
		//sdkResetTimer(&timer);
	}
	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", _avgFps);
	glutSetWindowTitle(fps);
}

bool Visual::InitGL(IVisualRender *render, int *argc, char **argv)
{
	_render = render;
	InitState();
	glutInit(argc, argv);
	int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
	int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition((screenWidth - WindowWidth) / 2, (screenHeight - WindowHeight) / 2);
	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Cuda GL Interop (VBO)");
	// register callbacks
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	glutDisplayFunc(Display);
	glutKeyboardFunc(Keyboard);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);

	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0"))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glPointSize(5);

	// viewport
	glViewport(0, 0, WindowWidth, WindowHeight);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(80.0, (GLfloat)WindowWidth / (GLfloat)WindowHeight, 0.01, 500.0);

	SDK_CHECK_ERROR_GL();
	return true;
}

void Visual::Main()
{
	_render->Initialize();
	// start rendering mainloop
	glutMainLoop();
}

#endif