#include "Tclite.h"

__device__ Tcl_Obj *Tcl_Obj::NewObj()
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::NewListObj(int argsLength, Tcl_Obj **args)
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::NewByteArrayObj(const void *value, int length)
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::NewIntObj(int value)
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::NewWideIntObj(int64 value)
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::NewDoubleObj(double value)
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::NewStringObj(const char *value, int length)
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::GetVar2Ex(Tcl_Interp *interp, const char *name, void *a, void *b)
{
	return nullptr;
}
__device__ Tcl_Obj *Tcl_Obj::DuplicateObj()
{
	return nullptr;
}

__device__ void Tcl_Obj::ObjSetVar2(Tcl_Interp *interp, Tcl_Obj *name, Tcl_Obj *value, bool a)
{
}
__device__ RC Tcl_Obj::ListObjAppendElement(Tcl_Interp *interp, Tcl_Obj *value)
{
	return RC_OK;
}
__device__ RC Tcl_Obj::ListObjGetElements(Tcl_Interp *interp, int *argsLength, Tcl_Obj ***args)
{
	return RC_OK;
}

__device__ uint8 *Tcl_Obj::GetByteArrayFromObj(int *length)
{
	return nullptr;
}
__device__ void Tcl_Obj::IncrRefCount()
{
}
__device__ void Tcl_Obj::DecrRefCount()
{
}
__device__ RC Tcl_Obj::GetIntFromObj(Tcl_Interp *interp, int *n)
{
	return RC_OK;
}
__device__ RC Tcl_Obj::SetIntObj(int n)
{
	return RC_OK;
}
__device__ RC Tcl_Obj::GetDoubleFromObj(Tcl_Interp *interp, double *r)
{
	return RC_OK;
}
__device__ RC Tcl_Obj::GetWideIntFromObj(Tcl_Interp *interp, int64 *v)
{
	return RC_OK;
}
__device__ RC Tcl_Obj::GetBooleanFromObj(Tcl_Interp *interp, bool *r)
{
	return RC_OK;
}
__device__ char *Tcl_Obj::GetString()
{
	return nullptr;
}
__device__ char *Tcl_Obj::GetStringFromObj(int *length)
{
	return nullptr;
}

__device__ RC Tcl_Interp::VarEval(char *cmd1, char *cmd2, char *cmd3, char *cmd4)
{
	return RC_OK;
}
__device__ RC Tcl_Interp::Eval(char *cmd)
{
	return RC_OK;
}
__device__ RC Tcl_Interp::EvalObjEx(Tcl_Obj *objPtr, bool a)
{
	return RC_OK;
}
__device__ void Tcl_Interp::AppendResult(const char *arg1, const char *arg2, const char *arg3, const char *arg4)
{
}
__device__ void Tcl_Interp::AppendResult(Tcl_Obj *arg1, Tcl_Obj *arg2, Tcl_Obj *arg3, Tcl_Obj *arg4)
{
}
__device__ void Tcl_Interp::SetObjResult(Tcl_Obj *value)
{
}
__device__ Tcl_Obj *Tcl_Interp::GetObjResult()
{
	return nullptr;
}
__device__ void Tcl_Interp::SetResult(char *string, Destructor_t destructor)
{
}
__device__ char *Tcl_Interp::GetStringResult()
{
	return nullptr;
}
__device__ void Tcl_Interp::ResetResult()
{
}
__device__ void Tcl_Interp::BackgroundError()
{
}
__device__ void Tcl_Interp::WrongNumArgs(int objc, array_t<Tcl_Obj *> objv, char *msg)
{
}

__device__ void Tcl_Interp::CreateObjCommand(const char *name, void *dummy1, ClientData cd, void *freeProc)
{
}
__device__ void Tcl_Interp::DeleteCommand(const char *name)
{
}
