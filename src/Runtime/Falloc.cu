#define __EMBED__ 1
#if __CUDACC__
#include "Falloc.cu.h"
#else
#include "Falloc.cpu.h"
#endif
#include "FallocHost.h"

///////////////////////////////////////////////////////////////////////////////
// VISUAL
#pragma region VISUAL
#ifdef VISUAL
#if __CUDACC__
#define MAX(a,b) (a > b ? a : b)
#define BLOCKPITCH 64
#define HEADERPITCH 4
#define BLOCKREFCOLOR make_float4(.7, 0, 0, 1)
#define BLOCKREF2COLOR make_float4(1, 0, 0, 1)
#define HEADERCOLOR make_float4(0, 1, 0, 1)
#define BLOCKCOLOR make_float4(0, 0, .7, 1)
#define BLOCK2COLOR make_float4(0, 0, 1, 1)
#define BLOCK3COLOR make_float4(0, .4, 1, 1)
#define BLOCK4COLOR make_float4(0, .7, 1, 1)
#define BLOCK5COLOR make_float4(.5, .7, 1, 1)
#define MARKERCOLOR make_float4(1, 1, 0, 1)

__inline__ __device__ static void OffsetBlockRef(unsigned int x, unsigned int y, float *x1, float *y1)
{
	x += (y % HEADERPITCH) * BLOCKPITCH; 
	y /= HEADERPITCH;
	*x1 = x * 1; *y1 = y * 1 + 1;
}

__global__ static void RenderHeap(quad4 *b, fallocHeap *heap, unsigned int offset)
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
	if (heap->freeBlockPtr)
	{
		size_t offset = ((char *)heap->freeBlockPtr - (char *)heap->blockRefs);
		offset %= heap->blockRefsLength;
		offset /= sizeof(fallocBlockRef);
		//
		OffsetBlockRef(offset % BLOCKPITCH, offset / BLOCKPITCH, &x1, &y1);
		b[index + 1] = make_quad4(
			make_float4(x1 + .0, y1 + .2, 1, 1), MARKERCOLOR,
			make_float4(x1 + .2, y1 + .2, 1, 1), MARKERCOLOR,
			make_float4(x1 + .2, y1 + .0, 1, 1), MARKERCOLOR,
			make_float4(x1 + .0, y1 + .0, 1, 1), MARKERCOLOR);
	}
	// retn
	if (heap->retnBlockPtr)
	{
		size_t offset = ((char *)heap->retnBlockPtr - (char *)heap->blockRefs);
		offset %= heap->blockRefsLength;
		offset /= sizeof(fallocBlockRef);
		//
		OffsetBlockRef(offset % BLOCKPITCH, offset / BLOCKPITCH, &x1, &y1);
		b[index + 2] = make_quad4(
			make_float4(x1 + .7, y1 + .9, 1, 1), MARKERCOLOR,
			make_float4(x1 + .9, y1 + .9, 1, 1), MARKERCOLOR,
			make_float4(x1 + .9, y1 + .7, 1, 1), MARKERCOLOR,
			make_float4(x1 + .7, y1 + .7, 1, 1), MARKERCOLOR);
	}
}

__global__ static void RenderBlock(quad4 *b, size_t blocks, unsigned int blocksY, fallocHeap *heap, unsigned int offset)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int blockIndex = y * BLOCKPITCH + x;
	if (blockIndex >= blocks)
		return;
	fallocBlockRef *ref = (fallocBlockRef *)((char *)heap->blockRefs + blockIndex * sizeof(fallocBlockRef));
	fallocBlockHeader *hdr = (fallocBlockHeader *)(heap->blocks + blockIndex * heap->blockSize);
	int index = blockIndex * 4 + offset;
	//
	float x1, y1; OffsetBlockRef(x, y, &x1, &y1);
	if (ref->block == nullptr)
	{
		b[index] = make_quad4(
			make_float4(x1 + 0.0, y1 + 0.9, 1, 1), BLOCKREFCOLOR,
			make_float4(x1 + 0.9, y1 + 0.9, 1, 1), BLOCKREFCOLOR,
			make_float4(x1 + 0.9, y1 + 0.0, 1, 1), BLOCKREFCOLOR,
			make_float4(x1 + 0.0, y1 + 0.0, 1, 1), BLOCKREFCOLOR);
	}
	else
	{
		b[index] = make_quad4(
			make_float4(x1 + 0.0, y1 + 0.9, 1, 1), BLOCKREF2COLOR,
			make_float4(x1 + 0.9, y1 + 0.9, 1, 1), BLOCKREF2COLOR,
			make_float4(x1 + 0.9, y1 + 0.0, 1, 1), BLOCKREF2COLOR,
			make_float4(x1 + 0.0, y1 + 0.0, 1, 1), BLOCKREF2COLOR);
	}
	// block
	float x2 = x * 10; float y2 = y * 20 + (blocksY / HEADERPITCH) + 3;
	if (hdr->magic != FALLOC_MAGIC)
	{
		b[index + 1] = make_quad4(
			make_float4(x2 + 0, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 19, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 9, y2 + 00, 1, 1), BLOCKCOLOR,
			make_float4(x2 + 0, y2 + 00, 1, 1), BLOCKCOLOR);
	} 
	else 
	{
		b[index + 1] = make_quad4(
			make_float4(x2 + 0, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 1, 1, 1), HEADERCOLOR,
			make_float4(x2 + 3.9, y2 + 0, 1, 1), HEADERCOLOR,
			make_float4(x2 + 0, y2 + 0, 1, 1), HEADERCOLOR);
		// block or node
		fallocCtx *ctx = (fallocCtx *)((char *)hdr + sizeof(fallocBlockHeader));
		if (ctx->node.magic != FALLOCNODE_MAGIC)
		{
			b[index + 2] = make_quad4(
				make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK2COLOR,
				make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK2COLOR);
		}
		else 
		{
			float percent = .303; //ceilf((heap->blockSize / ctx->node.freeOffset) * 100) / 100;
			float split = 19 * percent;
			if (ctx->magic != FALLOCCTX_MAGIC)
			{
				b[index + 2] = make_quad4(
					make_float4(x2 + 0, y2 + split, 1, 1), BLOCK4COLOR,
					make_float4(x2 + 9, y2 + split, 1, 1), BLOCK4COLOR,
					make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK4COLOR,
					make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK4COLOR);
			}
			else
			{
				b[index + 2] = make_quad4(
					make_float4(x2 + 0, y2 + split, 1, 1), BLOCK5COLOR,
					make_float4(x2 + 9, y2 + split, 1, 1), BLOCK5COLOR,
					make_float4(x2 + 9, y2 + 00, 1, 1), BLOCK5COLOR,
					make_float4(x2 + 0, y2 + 00, 1, 1), BLOCK5COLOR);
			}
			b[index + 3] = make_quad4(
				make_float4(x2 + 0, y2 + 19, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 9, y2 + 19, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 9, y2 + split, 1, 1), BLOCK3COLOR,
				make_float4(x2 + 0, y2 + split, 1, 1), BLOCK3COLOR);
		}
	}
}

__device__ void *_fallocTestObj;
__device__ fallocCtx *_fallocTestCtx;
__global__ static void Keypress(fallocHeap *heap, unsigned char key)
{
	_fallocSetHeap(heap);
	//char *testString;
	//int *testInteger;
	switch (key)
	{
	case 'a':
		_fallocTestObj = fallocGetBlock(heap);
		break;
	case 'b':
		fallocFreeBlock(heap, _fallocTestObj);
		break;
	case 'x':
		_fallocTestCtx = fallocCreateCtx(heap);
		break;
	//case 'y':
	//	testString = (char *)falloc(_fallocTestCtx, 10);
	//	testInteger = falloc<int>(_fallocTestCtx);
	//	break;
	case 'z':
		fallocDisposeCtx(_fallocTestCtx);
		break;
	}
}

inline size_t GetFallocRenderQuads(size_t blocks)
{ 
	return 3 + (blocks * 4);
}

static void LaunchFallocRender(float4 *b, size_t blocks, fallocHeap *heap)
{
	cudaCheckErrors(cudaFallocSetHeap(heap), exit(0));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	RenderHeap<<<heapGrid, heapBlock>>>((quad4 *)b, heap, 0);
	//
	dim3 blockBlock(16, 16, 1);
	dim3 blockGrid((unsigned int)MAX(BLOCKPITCH / 16, 1), (unsigned int)MAX(blocks / BLOCKPITCH / 16, 1), 1);
	RenderBlock<<<blockGrid, blockBlock>>>((quad4 *)b, blocks, (unsigned int)blocks / BLOCKPITCH, heap, 3);
}

static void LaunchFallocKeypress(fallocHeap *heap, unsigned char key)
{
	cudaCheckErrors(cudaFallocSetHeap(heap), exit(0));
	dim3 heapBlock(1, 1, 1);
	dim3 heapGrid(1, 1, 1);
	Keypress<<<heapGrid, heapBlock>>>(heap, key);
}

// _vbo variables
static GLuint _fallocVbo;
static GLsizei _fallocVboSize;
static struct cudaGraphicsResource *_fallocVboResource;

static void FallocRunCuda(size_t blocks, fallocHeap *heap, struct cudaGraphicsResource **resource)
{
	// map OpenGL buffer object for writing from CUDA
	cudaCheckErrors(cudaGraphicsMapResources(1, resource, nullptr), exit(0));
	float4 *b;
	size_t size;
	cudaCheckErrors(cudaGraphicsResourceGetMappedPointer((void **)&b, &size, *resource), exit(0));
	//printf("CUDA mapped VBO: May access %ld bytes\n", size);
	LaunchFallocRender(b, blocks, heap);
	// unmap buffer object
	cudaCheckErrors(cudaGraphicsUnmapResources(1, resource, nullptr), exit(0));
}

static void FallocCreateVBO(size_t blocks, GLuint *vbo, struct cudaGraphicsResource **resource, unsigned int vbo_res_flags)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	// initialize buffer object
	_fallocVboSize = (GLsizei)GetFallocRenderQuads(blocks) * 4;
	unsigned int size = _fallocVboSize * 2 * sizeof(float4);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// register this buffer object with CUDA
	cudaCheckErrors(cudaGraphicsGLRegisterBuffer(resource, *vbo, vbo_res_flags), exit(0));
	SDK_CHECK_ERROR_GL();
}

static void FallocDeleteVBO(GLuint *vbo, struct cudaGraphicsResource *resource)
{
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(resource);
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

void FallocVisualRender::Dispose()
{
	if (_fallocVbo)
		FallocDeleteVBO(&_fallocVbo, _fallocVboResource);
}

void FallocVisualRender::Keyboard(unsigned char key)
{
	switch (key)
	{
	case 'a':
	case 'b':
	case 'x':
	case 'y':
	case 'z':
		LaunchFallocKeypress((fallocHeap *)_fallocHost.heap, key);
		break;
	}
}

void FallocVisualRender::Display()
{
	size_t blocks = _fallocHost.blocksLength / _fallocHost.blockSize;
	// run CUDA kernel to generate vertex positions
	FallocRunCuda(blocks, (fallocHeap *)_fallocHost.heap, &_fallocVboResource);

	//gluLookAt(0, 0, 200, 0, 0, 0, 0, 1, 0);
	//glScalef(.02, .02, .02);
	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(Visual::TranslateX, Visual::TranslateY, Visual::TranslateZ);
	glRotatef(Visual::RotateX, 1.0, 0.0, 0.0);
	glRotatef(Visual::RotateY, 0.0, 1.0, 0.0);

	// render from the _vbo
	glBindBuffer(GL_ARRAY_BUFFER, _fallocVbo);
	glVertexPointer(4, GL_FLOAT, sizeof(float4) * 2, 0);
	glColorPointer(4, GL_FLOAT, sizeof(float4) * 2, (GLvoid*)sizeof(float4));

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_QUADS, 0, _fallocVboSize);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void FallocVisualRender::Initialize()
{
	size_t blocks = _fallocHost.blocksLength / _fallocHost.blockSize;
	// create VBO
	FallocCreateVBO(blocks, &_fallocVbo, &_fallocVboResource, cudaGraphicsMapFlagsWriteDiscard);
	// run the cuda part
	FallocRunCuda(blocks, (fallocHeap *)_fallocHost.heap, &_fallocVboResource);
}

#undef MAX
#undef BLOCKPITCH
#undef HEADERPITCH
#undef BLOCKREFCOLOR
#undef BLOCKREF2COLOR
#undef HEADERCOLOR
#undef BLOCKCOLOR
#undef BLOCK2COLOR
#undef BLOCK3COLOR
#undef BLOCK4COLOR
#undef BLOCK5COLOR
#undef MARKERCOLOR

#else
void FallocVisualRender::Dispose() { }
void FallocVisualRender::Keyboard(unsigned char key) { }
void FallocVisualRender::Display() { }
void FallocVisualRender::Initialize() { }
#endif
#endif
#pragma endregion
