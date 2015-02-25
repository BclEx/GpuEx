#include <Runtime.h>
#include <RuntimeTypes.h>

#if defined(__GNUC__) && 0
#define likely(X)    __builtin_expect((X),1)
#define unlikely(X)  __builtin_expect((X),0)
#else
#define likely(X) !!(X)
#define unlikely(X) !!(X)
#endif

#include "ConvertEx.cu.h"
#include "RC.cu.h"
#include "VAlloc.cu.h"
#include "MutexEx.cu.h"
#include "SysEx.cu.h"
#include "Bitvec.cu.h"
#include "Hash.cu.h"
#include "StatusEx.cu.h"
#include "VSystem.cu.h"
#include "MathEx.cu.h"
#include "IO\IO.VFile.cu.h"
using namespace Core;
using namespace Core::IO;
