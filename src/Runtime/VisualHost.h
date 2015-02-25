#ifndef _VISUAL_H__
#define _VISUAL_H__
#pragma warning(disable: 4996)
#include "RuntimeGL.h"

// constants
#define REFRESH_DELAY 10 //ms
const unsigned int WindowWidth = 800;
const unsigned int WindowHeight = 600;

struct quad4
{ 
	float4 av, ac;
	float4 bv, bc;
	float4 cv, cc;
	float4 dv, dc;
};
static __inline__ __device__ quad4 make_quad4(
	float4 av, float4 ac,
	float4 bv, float4 bc,
	float4 cv, float4 cc,
	float4 dv, float4 dc)
{
	quad4 q; q.av = av; q.ac = ac; q.bv = bv; q.bc = bc; q.cv = cv; q.cc = cc; q.dv = dv; q.dc = dc; return q;
}

class IVisualRender
{
public:
	virtual void Dispose() = 0;
	virtual void Display() = 0;
	virtual void Keyboard(unsigned char key) = 0;
	virtual void Initialize() = 0;
};

class Visual
{
public:
	static float RotateX;
	static float RotateY;
	static float TranslateX;
	static float TranslateY;
	static float TranslateZ;

protected:
	static IVisualRender* _render;
	// mouse controls
	static int _mouseLastX;
	static int _mouseLastY;
	static int _mouseState;
	//StopWatchInterface *_timer;
	static int _fpsCount; // FPS count for averaging
	static int _fpsLimit; // FPS limit for sampling
	static float _avgFps;

	static void Display();
	static void Keyboard(unsigned char key, int, int);
	static void Mouse(int button, int state, int x, int y);
	static void Motion(int x, int y);
	static void TimerEvent(int value);
	static void ComputeFPS();

public:
	//inline ~Visual() { Dispose(); }
	inline static void Dispose()
	{
		//sdkDeleteTimer(&_timer);
		if (_render)
		{
			_render->Dispose();
			delete _render; _render = 0;
		}
	}
	static void Main();
	inline static void InitState()
	{
		// mouse controls
		_mouseState = 0;
		RotateX = 180.0f, RotateY = 0.0f;
		TranslateX = -200.0f, TranslateY = 150.0f, TranslateZ = -200.0f;
		//TranslateX = -30f, TranslateY = 20.0f, TranslateZ = -50.0f;
		//TranslateX = -3f, TranslateY = 5.0f, TranslateZ = -12.0f;
		//
		//_timer = nullptr;
		_fpsCount = 0; // FPS count for averaging
		_fpsLimit = 1; // FPS limit for sampling
		_avgFps = 0.0f;
	}
	static bool InitGL(IVisualRender* render, int *argc, char **argv);
};

#endif // __VISUAL_H__
