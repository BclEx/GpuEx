#ifndef __TESTTCL_CU_H__
#define __TESTTCL_CU_H__

#include <Core/Core.cu.h>

typedef struct Tcl_Interp Tcl_Interp;

typedef void (Tcl_FreeInternalRepProc)(struct Tcl_Obj *objPtr);
typedef void (Tcl_DupInternalRepProc)(struct Tcl_Obj *srcPtr, struct Tcl_Obj *dupPtr);
typedef void (Tcl_UpdateStringProc)(struct Tcl_Obj *objPtr);
typedef int (Tcl_SetFromAnyProc)(Tcl_Interp *interp, struct Tcl_Obj *objPtr);

// The following structure represents a type of object, which is a particular internal representation for an object plus a set of
// procedures that provide standard operations on objects of that type.
typedef struct Tcl_ObjType
{
	char *Name;			// Name of the type, e.g. "int".
	Tcl_FreeInternalRepProc *FreeIntRepProc; // Called to free any storage for the type's internal rep. NULL if the internal rep does not need freeing.
	Tcl_DupInternalRepProc *DupIntRepProc; // Called to create a new object as a copy of an existing object.
	Tcl_UpdateStringProc *UpdateStringProc; // Called to update the string rep from the type's internal representation.
	Tcl_SetFromAnyProc *SetFromAnyProc; // Called to convert the object's internal rep to this type. Frees the internal rep of the old type. Returns TCL_ERROR on failure.
} Tcl_ObjType;


//////////////////////
// TCLARG
#pragma region TCLARG

struct tcl_list0 { char *i; };
template <typename T1> struct tcl_list1 : tcl_list0 { T1 v1; };
template <typename T1, typename T2> struct tcl_list2 : tcl_list0 { T1 v1; T2 v2; };
template <typename T1, typename T2, typename T3> struct tcl_list3 : tcl_list0 { T1 v1; T2 v2; T3 v3; };
template <typename T1, typename T2, typename T3, typename T4> struct tcl_list4 : tcl_list0 { T1 v1; T2 v2; T3 v3; T4 v4; };
template <typename T1, typename T2, typename T3, typename T4, typename T5> struct tcl_list5 : tcl_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> struct tcl_list6 : tcl_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> struct tcl_list7 : tcl_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> struct tcl_list8 : tcl_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> struct tcl_list9 : tcl_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; };
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> struct tcl_listA : tcl_list0 { T1 v1; T2 v2; T3 v3; T4 v4; T5 v5; T6 v6; T7 v7; T8 v8; T9 v9; TA vA; };
#define tcl_list tcl_list0 
#define _INTSIZEOF(n) ((sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1))
#define tcl_arg(ap, t) (*(t *)((ap.i = (char *)_ROUNDT(t, (unsigned long long)(ap.i + _INTSIZEOF(t)))) - _INTSIZEOF(t)))
#define tcl_end(ap) (ap.i = nullptr);

__device__ __forceinline static void tcl_start(tcl_list &args)
{
	args.i = nullptr;
}
template <typename T1> __device__ __forceinline static void tcl_start(tcl_list1<T1> &args, T1 arg1)
{
	args.i = (char *)&args.v1; args.v1 = arg1;
}
template <typename T1, typename T2> __device__ __forceinline static void tcl_start(tcl_list2<T1,T2> &args, T1 arg1, T2 arg2)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2;
}
template <typename T1, typename T2, typename T3> __device__ __forceinline static void tcl_start(tcl_list3<T1,T2,T3> &args, T1 arg1, T2 arg2, T3 arg3)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3;
}
template <typename T1, typename T2, typename T3, typename T4> __device__ __forceinline static void tcl_start(tcl_list4<T1,T2,T3,T4> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ __forceinline static void tcl_start(tcl_list5<T1,T2,T3,T4,T5> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ __forceinline static void tcl_start(tcl_list6<T1,T2,T3,T4,T5,T6> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ __forceinline static void tcl_start(tcl_list7<T1,T2,T3,T4,T5,T6,T7> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ __forceinline static void tcl_start(tcl_list8<T1,T2,T3,T4,T5,T6,T7,T8> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ __forceinline static void tcl_start(tcl_list9<T1,T2,T3,T4,T5,T6,T7,T8,T9> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ __forceinline static void tcl_start(tcl_listA<T1,T2,T3,T4,T5,T6,T7,T8,T9,TA> &args, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA)
{
	args.i = (char *)&args.v1; args.v1 = arg1; args.v2 = arg2; args.v3 = arg3; args.v4 = arg4; args.v5 = arg5; args.v6 = arg6; args.v7 = arg7; args.v8 = arg8; args.v9 = arg9; args.vA = argA;
}

#pragma endregion

// One of the following structures exists for each object in the Tcl system. An object stores a value as either a string, some internal
// representation, or both.
typedef struct Tcl_Obj
{
	int RefCount;	// When 0 the object will be freed.
	char *Bytes;	// This points to the first byte of the object's string representation. The array must be followed by a null byte (i.e., at offset length) but may also contain
	// embedded null characters. The array's storage is allocated by ckalloc. NULL means the string rep is invalid and must be regenerated from the internal rep.
	// Clients should use Tcl_GetStringFromObj or Tcl_GetString to get a pointer to the byte array as a readonly value.
	int Length;		// The number of bytes at *bytes, not including the terminating null.
	Tcl_ObjType *TypePtr;	// Denotes the object's type. Always corresponds to the type of the object's internal rep. NULL indicates the object has no internal rep (has no type).
	union // The internal representation:
	{			
		long LongValue;			// an long integer value
		double DoubleValue;		// a double-precision floating value
		void *OtherValuePtr;	// another, type-specific value
		int64 WideValue;		// a long long value
		struct { void *Ptr1; void *Ptr2; } TwoPtrValue;// internal rep as two pointers
	} InternalRep;

	__device__ static Tcl_Obj *NewObj();
	__device__ static Tcl_Obj *NewListObj(int argsLength, Tcl_Obj **args);
	__device__ static Tcl_Obj *NewByteArrayObj(const void *value, int length);
	__device__ static Tcl_Obj *NewIntObj(int value);
	__device__ static Tcl_Obj *NewWideIntObj(int64 value);
	__device__ static Tcl_Obj *NewDoubleObj(double value);
	__device__ static Tcl_Obj *NewStringObj(const char *value, int length);
	template<typename Action> __device__ inline static Tcl_Obj *NewLambdaObj(Action action)
	{
		return nullptr;
	}
	__device__ static Tcl_Obj *GetVar2Ex(Tcl_Interp *interp, const char *name, void *a, void *b);
	__device__ static Tcl_Obj *DuplicateObj();

	__device__ void ObjSetVar2(Tcl_Interp *interp, Tcl_Obj *name, Tcl_Obj *value, bool a);
	__device__ RC ListObjAppendElement(Tcl_Interp *interp, Tcl_Obj *value);
	__device__ RC ListObjGetElements(Tcl_Interp *interp, int *argsLength, Tcl_Obj ***args);

	__device__ uint8 *GetByteArrayFromObj(int *length);
	__device__ void IncrRefCount();
	__device__ void DecrRefCount();
	__device__ RC GetIntFromObj(Tcl_Interp *interp, int *n);
	__device__ RC SetIntObj(int n);
	__device__ RC GetDoubleFromObj(Tcl_Interp *interp, double *r);
	__device__ RC GetWideIntFromObj(Tcl_Interp *interp, int64 *v);
	__device__ RC GetBooleanFromObj(Tcl_Interp *interp, bool *r);
	__device__ char *GetString();
	__device__ char *GetStringFromObj(int *length);
} Tcl_Obj;

typedef void *ClientData;

struct Tcl_Interp
{
	char *Result;
	Destructor_t FreeProc;

	__device__ RC VarEval(char *cmd1, char *cmd2 = nullptr, char *cmd3 = nullptr, char *cmd4 = nullptr);
	__device__ RC Eval(char *cmd);
	__device__ RC EvalObjEx(Tcl_Obj *objPtr, bool a);
	__device__ void AppendResult(const char *arg1, const char *arg2 = nullptr, const char *arg3 = nullptr, const char *arg4 = nullptr);
	__device__ void AppendResult(Tcl_Obj *arg1, Tcl_Obj *arg2 = nullptr, Tcl_Obj *arg3 = nullptr, Tcl_Obj *arg4 = nullptr);
	__device__ void SetObjResult(Tcl_Obj *value);
	__device__ Tcl_Obj *GetObjResult();
	__device__ void SetResult(char *string, Destructor_t destructor);
	__device__ char *GetStringResult();
	__device__ void ResetResult();
	__device__ void BackgroundError();
	__device__ void WrongNumArgs(int objc, array_t<Tcl_Obj *> objv, char *msg);

	__device__ void CreateObjCommand(const char *name, void *dummy1, ClientData cd, void *freeProc);
	__device__ void DeleteCommand(const char *name);
};

typedef void *Tcl_Channel;

#define TCL_EVAL_GLOBAL 0
#define TCL_EVAL_DIRECT 0
#define TCL_GLOBAL_ONLY

__device__ static inline array_t<const char *> Y(const char *arg1) { const char **v = new const char *[1]; v[0] = arg1; return array_t<const char *>(v, 1); }
__device__ static inline array_t<const char *> Y(const char *arg1, const char *arg2) { const char **v = new const char *[2]; v[0] = arg1; v[1] = arg2; return array_t<const char *>(v, 2); }
__device__ static inline array_t<const char *> Y(const char *arg1, const char *arg2, const char *arg3) { const char **v = new const char *[3]; v[0] = arg1; v[1] = arg2; v[2] = arg3; return array_t<const char *>(v, 3); }

#define PA(n) Tcl_Obj **v = new Tcl_Obj *[n]
#define P_(n) v[n-1] = arg##n
#define PC(n) v[n-1] = Tcl_Obj::NewStringObj(arg##n, _strlen30(arg##n))
#define PF(n) v[n-1] = Tcl_Obj::NewLambdaObj(arg##n)
#define PZ(n) return array_t<Tcl_Obj *>(v, n)

__device__ static inline array_t<Tcl_Obj *> Zc(const char *arg1) { PA(1);PC(1);PZ(1); }
__device__ static inline array_t<Tcl_Obj *> Z_(Tcl_Obj *arg1) { PA(1);P_(1);PZ(1); }
template<typename T1> __device__ static inline array_t<Tcl_Obj *> Zf(T1 arg1) { PA(1);PF(1);PZ(1); }
//
__device__ static inline array_t<Tcl_Obj *> Zcc(const char *arg1, const char *arg2) { PA(2);PC(1);PC(2);PZ(2); }
__device__ static inline array_t<Tcl_Obj *> Zc_(const char *arg1, Tcl_Obj *arg2) { PA(2);PC(1);P_(2);PZ(2); }
template<typename T2> __device__ static inline array_t<Tcl_Obj *> Zcf(const char *arg1, T2 arg2) { PA(2);PC(1);PF(2);PZ(2); }
//
template<typename T3> __device__ static inline array_t<Tcl_Obj *> Zc_f(const char *arg1, Tcl_Obj *arg2, T3 arg3) { PA(3);PC(1);P_(2);PF(3);PZ(3); }

#endif // __TESTTCL_CU_H__