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
	__device__ void WrongNumArgs(int objc, Tcl_Obj *const objv[], char *msg);

	__device__ void CreateObjCommand(const char *name, void *dummy1, ClientData cd, void *freeProc);
	__device__ void DeleteCommand(const char *name);
};

typedef void *Tcl_Channel;

#define TCL_EVAL_GLOBAL 0
#define TCL_EVAL_DIRECT 0
#define TCL_GLOBAL_ONLY