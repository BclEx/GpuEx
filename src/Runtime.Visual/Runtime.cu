#define __EMBED__ 1
#include "VisualHost.h"
#include "Runtime.h"
#include "RuntimeTypes.h"

///////////////////////////////////////////////////////////////////////////////
// VISUAL
#pragma region VISUAL

#if __CUDACC__
//#define MAX(a,b) (a > b ? a : b)
#define BLOCKPITCH 64
#define HEADERPITCH 4
#define BLOCKREFCOLOR make_float4(1, 0, 0, 1)
#define HEADERCOLOR make_float4(0, 1, 0, 1)
#define BLOCKCOLOR make_float4(0, 0, .7, 1)
#define BLOCK2COLOR make_float4(0, 0, 1, 1)
#define MARKERCOLOR make_float4(1, 1, 0, 1)

__global__ static void RenderHeap(struct runtimeHeap_s *heap, quad4 *b, unsigned int offset)
{
	int index = offset;
	// heap
	b[index] = make_quad4(
		make_float4(00, 1, 1, 1), HEADERCOLOR,
		make_float4(10, 1, 1, 1), HEADERCOLOR,
		make_float4(10, 0, 1, 1), HEADERCOLOR,
		make_float4(00, 0, 1, 1), HEADERCOLOR);
	// free
	float x1, y1;
	if (heap->blockPtr)
	{
		size_t offset = ((char *)heap->blockPtr - (char *)heap->blocks);
		offset %= heap->blocksLength;
		offset /= heap->blockSize;
		//
		unsigned int x = offset % BLOCKPITCH;
		unsigned int y = offset / BLOCKPITCH;
		x1 = x * 10; y1 = y * 20 + 2;
		b[index + 1] = make_quad4(
			make_float4(x1 + 0, y1 + 1, 1, 1), MARKERCOLOR,
			make_float4(x1 + 1, y1 + 1, 1, 1), MARKERCOLOR,
			make_float4(x1 + 1, y1 + 0, 1, 1), MARKERCOLOR,
			make_float4(x1 + 0, y1 + 0, 1, 1), MARKERCOLOR);
	}
}

__global__ static void RenderBlock(struct runtimeHeap_s *heap, quad4 *b, size_t blocks, unsigned int blocksY, unsigned int offset)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int blockIndex = y * BLOCKPITCH + x;
	if (blockIndex >= blocks)
		return;
	runtimeBlockHeader *hdr = (runtimeBlockHeader *)(heap->blocks + blockIndex * heap->blockSize);
	int index = blockIndex * 2 + offset;
	// block
	float x2 = x * 10; float y2 = y * 20 + 2;
	if (hdr->magic != RUNTIME_MAGIC || hdr->fmtoffset >= heap->blockSize)
	{
		b[index] = make_quad4(
			make_float4(x2 + 0, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 00, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 0, y2 + 00, 1, 1), BLOCKCOLOR);
	}
	else
	{
		b[index] = make_quad4(
			make_float4(x2 + 0, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 0, 1, 1), HEADERCOLOR,
			make_float4(x2 + 0, y2 + 0, 1, 1), HEADERCOLOR);
		b[index + 1] = make_quad4(
			make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK2COLOR,
			make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK2COLOR,
			make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK2COLOR,
			make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK2COLOR);
	}
}

__global__ static void Keypress(struct runtimeHeap_s *heap, unsigned char key)
{
	_runtimeSetHeap(heap);
	switch (key)
	{
	case 'a': _printf("Test\n"); break;
	case 'A': printf("Test\n"); break;
	case 'b': _printf("Test %d\n", threadIdx.x); break;
	case 'B': printf("Test %d\n", threadIdx.x); break;
	case 'c': _assert(true); break;
	case 'C': assert(true); break;
	case 'd': _assert(false); break;
	case 'D': assert(false); break;
	case 'e': _transfer("test", 1); break;
	case 'f': _throw("test", 1); break;
	case 'g': {
		va_list2<int, const char *> args;
		_va_start(args, 1, "2");
		int a1 = _va_arg(args, int);
		char *a2 = _va_arg(args, char*);
		_va_end(args);
		break; }
	}
}

inline size_t GetRuntimeRenderQuads(size_t blocks)
{ 
	return 2 + (blocks * 2);
}

static void LaunchRuntimeRender(cudaDeviceHeap &host, float4 *b, size_t blocks)
{
	cudaErrorCheck(cudaDeviceHeapSelect(host));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	RenderHeap<<<heapGrid, heapBlock>>>((runtimeHeap *)host.heap, (quad4 *)b, 0);
	//
	dim3 blockBlock(16, 16, 1);
	dim3 blockGrid((unsigned int)MAX(BLOCKPITCH / 16, 1), (unsigned int)MAX(blocks / BLOCKPITCH / 16, 1), 1);
	RenderBlock<<<blockGrid, blockBlock>>>((runtimeHeap *)host.heap, (quad4 *)b, blocks, (unsigned int)blocks / BLOCKPITCH, 2);
}

static void LaunchRuntimeKeypress(cudaDeviceHeap &host, unsigned char key)
{
	if (key == 'z')
	{
		cudaDeviceHeapSynchronize(host);
		return;
	}
	cudaErrorCheck(cudaDeviceHeapSelect(host));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	Keypress<<<heapGrid, heapBlock>>>((runtimeHeap *)host.heap, key);
}

// _vbo variables
static GLuint _runtimeVbo;
static GLsizei _runtimeVboSize;
static struct cudaGraphicsResource *_runtimeVboResource;

static void RuntimeRunCuda(cudaDeviceHeap &host, size_t blocks, struct cudaGraphicsResource **resource)
{
	// map OpenGL buffer object for writing from CUDA
	cudaErrorCheck(cudaGraphicsMapResources(1, resource, nullptr));
	float4 *b;
	size_t size;
	cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void **)&b, &size, *resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", size);
	LaunchRuntimeRender(host, b, blocks);
	// unmap buffer object
	cudaErrorCheck(cudaGraphicsUnmapResources(1, resource, nullptr));
}

static void RuntimeCreateVBO(size_t blocks, GLuint *vbo, struct cudaGraphicsResource **resource, unsigned int vbo_res_flags)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	// initialize buffer object
	_runtimeVboSize = (GLsizei)GetRuntimeRenderQuads(blocks) * 4;
	unsigned int size = _runtimeVboSize * 2 * sizeof(float4);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// register this buffer object with CUDA
	cudaErrorCheck(cudaGraphicsGLRegisterBuffer(resource, *vbo, vbo_res_flags));
	SDK_CHECK_ERROR_GL();
}

static void RuntimeDeleteVBO(GLuint *vbo, struct cudaGraphicsResource *resource)
{
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(resource);
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

void RuntimeVisualRender::Dispose()
{
	if (_runtimeVbo)
		RuntimeDeleteVBO(&_runtimeVbo, _runtimeVboResource);
}

void RuntimeVisualRender::Keyboard(unsigned char key)
{
	LaunchRuntimeKeypress(_runtimeHost, key);
}

void RuntimeVisualRender::Display()
{
	size_t blocks = _runtimeHost.blocksLength / _runtimeHost.blockSize;
	// run CUDA kernel to generate vertex positions
	RuntimeRunCuda(_runtimeHost, blocks, &_runtimeVboResource);

	//gluLookAt(0, 0, 200, 0, 0, 0, 0, 1, 0);
	//glScalef(.02, .02, .02);
	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(Visual::TranslateX, Visual::TranslateY, Visual::TranslateZ);
	glRotatef(Visual::RotateX, 1.0, 0.0, 0.0);
	glRotatef(Visual::RotateY, 0.0, 1.0, 0.0);

	// render from the _vbo
	glBindBuffer(GL_ARRAY_BUFFER, _runtimeVbo);
	glVertexPointer(4, GL_FLOAT, sizeof(float4) * 2, 0);
	glColorPointer(4, GL_FLOAT, sizeof(float4) * 2, (GLvoid*)sizeof(float4));

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_QUADS, 0, _runtimeVboSize);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void RuntimeVisualRender::Initialize()
{
	size_t blocks = _runtimeHost.blocksLength / _runtimeHost.blockSize;
	// create VBO
	RuntimeCreateVBO(blocks, &_runtimeVbo, &_runtimeVboResource, cudaGraphicsMapFlagsWriteDiscard);
	// run the cuda part
	RuntimeRunCuda(_runtimeHost, blocks, &_runtimeVboResource);
}

#undef MAX
#undef BLOCKPITCH
#undef HEADERPITCH
#undef BLOCKREFCOLOR
#undef HEADERCOLOR
#undef BLOCKCOLOR
#undef BLOCK2COLOR
#undef MARKERCOLOR

#else
void RuntimeVisualRender::Dispose() { }
void RuntimeVisualRender::Keyboard(unsigned char key) { }
void RuntimeVisualRender::Display() { }
void RuntimeVisualRender::Initialize() { }
#endif

#pragma endregion
