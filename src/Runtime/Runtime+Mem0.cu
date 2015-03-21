#include "Runtime.h"
#ifdef RUNTIME_ALLOC_ZERO

static void *__allocsystem_alloc(int bytes) { return 0; }
static void __allocsystem_free(void *prior) { return; }
static void *__allocsystem_realloc(void *prior, int bytes) { return 0; }
static int __allocsystem_size(void *prior ){ return 0; }
static int __allocsystem_roundup(int n) { return n; }
static int __allocsystem_init(void *notUsed1) { return 1; }
static void __allocsystem_shutdown(void *notUsed1) { return; }

#endif /* RUNTIME_ALLOC_ZERO */
