#include <System.cu\cuPrintf.cu>
#define CLIENTONLY
#include <System.cu\cuFalloc.cu>

__global__ void testLinkedList(fallocHeap* heap)
{
	cuPrintf("Here");

	//fallocInit(heap);

	//// create/free heap
	//void* obj = fallocGetChunk(heap);
	//fallocFreeChunk(heap, obj);

	////void* obj2 = fallocGetChunks(heap, 144*2);
	////fallocFreeChunks(heap, obj2);

	//// create/free alloc
	//fallocContext* ctx = fallocCreateCtx(heap);
	//char* testString = (char*)falloc(ctx, 10);
	//int* testInteger = falloc<int>(ctx);
	//fallocDisposeCtx(ctx);
	//
	//// create/free stack
	//fallocContext* stack = fallocCreateCtx(heap);
	//fallocPush<int>(ctx, 1);
	//fallocPush<int>(ctx, 2);
	//int b = fallocPop<int>(ctx);
	//int a = fallocPop<int>(ctx);
	//fallocDisposeCtx(ctx);
}

void _testLinkedList(fallocHeap* heap) { testLinkedList<<<1, 1>>>(heap); }