#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <Core+Test\TestCtx.cu.h>

#pragma region Preamble

#if __CUDACC__
#define DEVICE(name) \
class name##Class : public Tester { public: __device__ void Test(); }; \
	__global__ void name##Launcher(void *r) { _runtimeSetHeap(r); name##Class c; c.Initialize(); c.Test(); } \
	void name##_host(cudaDeviceHeap &r) { name##Launcher<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r); } \
	__device__ void name##Class::Test()
#else
#define DEVICE(name) \
class name##Class : public Tester { public: __device__ void Test(); }; \
	__global__ void name##Launcher(void *r) { _runtimeSetHeap(r); name##Class c; c.Initialize(); c.Test(); } \
	void name##_host(cudaDeviceHeap &r) { name##Launcher(r.heap); cudaDeviceHeapSynchronize(r); } \
	__device__ void name##Class::Test()
#endif

#pragma endregion

enum PLATFORM
{
	PLATFORM_WINDOWS = 1,
};

class Tester
{
public:
	PLATFORM _tcl_platform;
	Hash G;
	bool _do_not_use_codec;
	bool _SLAVE;
	void *_DB;
	char *_SETUP_SQL;
	TestCtx *db;
	TestCtx *db2;
	TestCtx *db3;

	__device__ void sqlite3(TestCtx *ctx, array_t<const char *> args);
	__device__ int GetFileRetries();
	__device__ int GetFileRetryDelay();
	__device__ char *GetPwd();
	__device__ void copy_file(char *from, char *to);
	__device__ void forcecopy(char *from, char *to);
	__device__ void do_copy_file(bool force, char *from, char *to);
	__device__ bool is_relative_file(char *file);
	__device__ bool test_pwd(array_t<char *> args);
	__device__ void delete_file(const char *args[], int argsLength);
	__device__ void forcedelete(const char *args[], int argsLength);
	__device__ void do_delete_file(bool force, const char *args[], int argsLength);
	__device__ void execpresql(TestCtx *handle, void *args);
	__device__ void do_not_use_codec();
	__device__ void Initialize();
	__device__ void reset_db();
	__device__ void finish_test();
	__device__ void finalize_testing();
	__device__ void show_memstats();
	__device__ void execsql(const char *sql, TestCtx *db = nullptr);
	template<typename Action> __device__ inline void do_test(const char *name, Action cmd, char *expected)
	{
		//:fix_testname name
		//:_memdbg_settitle(name);

		//	//#  if {[llength $argv]==0} { 
		//	//#    set go 1
		//	//#  } else {
		//	//#    set go 0
		//	//#    foreach pattern $argv {
		//	//#      if {[string match $pattern $name]} {
		//	//#        set go 1
		//	//#        break
		//	//#      }
		//	//#    }
		//	//#  }

		//  if {[info exists ::G(perm:prefix)]} {
		//    set name "$::G(perm:prefix)$name"
		//  }

		//incr_ntest();
		printf("%s...", name);

		//	//  if {![info exists ::G(match)] || [string match $::G(match) $name]} {
		//	//    if {[catch {uplevel #0 "$cmd;\n"} result]} {
		//	//      puts "\nError: $result"
		//	//      fail_test $name
		//	//    } else {
		//	//      if {[regexp {^~?/.*/$} $expected]} {
		//	//        if {[string index $expected 0]=="~"} {
		//	//          set re [string map {# {[-0-9.]+}} [string range $expected 2 end-1]]
		//	//          set ok [expr {![regexp $re $result]}]
		//	//        } else {
		//	//          set re [string map {# {[-0-9.]+}} [string range $expected 1 end-1]]
		//	//          set ok [regexp $re $result]
		//	//        }
		//	//      } else {
		//	//        set ok [expr {[string compare $result $expected]==0}]
		//	//      }
		//	//      if {!$ok} {
		//	//        # if {![info exists ::testprefix] || $::testprefix eq ""} {
		//	//        #   error "no test prefix"
		//	//        # }
		//	//        puts "\nExpected: \[$expected\]\n     Got: \[$result\]"
		//	//        fail_test $name
		//	//      } else {
		//	//        puts " Ok"
		//	//      }
		//	//    }
		//	//  } else {
		//	//    puts " Omitted"
		//	//    omit_test $name "pattern mismatch" 0
		//	//  }



		cmd();
	}
};
