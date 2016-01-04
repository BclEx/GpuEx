#ifndef __JIMEX__H
#define __JIMEX__H
#include "Jim.h"
#ifdef __cplusplus
extern "C" {
#endif

	//JIM_EXPORT __device__ int Jim_GetEnum(Jim_Interp *interp, Jim_Obj *objPtr, const char * const *tablePtr, int *indexPtr, const char *name, int flags);

	//static int Jim_ObjSetVar2(Jim_Interp *interp, Jim_Obj *nameObjPtr, Jim_Obj *keyObjPtr, Jim_Obj *valObjPtr) { return Jim_SetDictKeysVector(interp, nameObjPtr, &keyObjPtr, 1, valObjPtr, 0); }
	__inline __device__ static int Jim_ObjSetVar2(Jim_Interp *interp, Jim_Obj *nameObjPtr, Jim_Obj *keyObjPtr, Jim_Obj *valObjPtr) { return Jim_SetDictKeysVector(interp, nameObjPtr, &keyObjPtr, 1, valObjPtr, 0); }

#define Jim_AppendResult(i, ...) Jim_AppendStrings(i, Jim_GetResult(i), __VA_ARGS__)
#define Jim_AppendElement(i, s) Jim_ListAppendElement(i, Jim_GetResult(i), Jim_NewStringObj(i, s, -1))
#define Jim_LinkVar
#define JIM_LINK_STRING 0
#define JIM_LINK_READ_ONLY 0
#define JIM_LINK_INT 0

#define Jim_ListObjGetElements (int)0

	typedef struct Jim_CmdInfo {
		Jim_CmdProc *objProc;
		ClientData objClientData;
	} Jim_CmdInfo;
	JIM_EXPORT __device__ int Jim_GetCommandInfo(Jim_Interp *interp, const char *name, Jim_CmdInfo *cmdInfo);

#ifdef __cplusplus
}
#endif
#endif // __JIMEX__H
