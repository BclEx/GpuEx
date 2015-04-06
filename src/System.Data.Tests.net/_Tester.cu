#include "_Tester.cu.h"

#pragma region tester.tcl

__device__ void Tester::sqlite3(TestCtx *tctx, array_t<const char *> args)
{
	if (args.length >= 1 && ((char *)args[0])[0] != '-')
	{
		// This command is opening a new database connection.
		if (G.Find("perm:sqlite3_args", 17))
		{
			//set args [concat $args $::G(perm:sqlite3_args)]
		}
		//if {[sqlite_orig -has-codec] && ![info exists ::do_not_use_codec]} {
		//  lappend args -key {xyzzy}
		//}

		auto res = 0; //[uplevel 1 sqlite_orig $args]
		auto presql = (Tcl_Obj *)G.Find("perm:presql", 11);
		//if (presql)
		//	((TestCtx *)args[0])->DB_EVAL(db->Encode(presql));
		if (G.Find("perm:dbconfig", 13))
		{
			//set ::dbhandle [lindex $args 0]
			//uplevel #0 G["perm:dbconfig"]
		}
		res = 0;
	}
	else
	{
		// This command is not opening a new database connection. Pass the arguments through to the C implementation as the are.
		//uplevel 1 sqlite_orig $args
	}
}

__device__ int Tester::GetFileRetries()
{
	if (!G.Find("file-retries", 12))
	{
		// NOTE: Return the default number of retries for [file] operations.  A value of zero or less here means "disabled".
		return (_tcl_platform == PLATFORM_WINDOWS ? 50 : 0);
	}
	return (int)G.Find("file-retries", 12);
}

__device__ int Tester::GetFileRetryDelay()
{
	if (!G.Find("file-retry-delay", 16))
	{
		// NOTE: Return the default number of milliseconds to wait when retrying failed [file] operations.  A value of zero or less means "do not wait".
		return 100; // TODO: Good default?
	}
	return (int)G.Find("file-retry-delay", 16);
}

// Return the string representing the name of the current directory.  On Windows, the result is "normalized" to whatever our parent command shell
// is using to prevent case-mismatch issues.
__device__ char *Tester::GetPwd()
{
	if (_tcl_platform == PLATFORM_WINDOWS)
	{
		return "pwd";
	}
	return "pwd";
}

// Copy file $from into $to. This is used because some versions of TCL for windows (notably the 8.4.1 binary package shipped with the
// current mingw release) have a broken "file copy" command.
__device__ void Tester::copy_file(char *from, char *to) { do_copy_file(false, from, to); }
__device__ void Tester::forcecopy(char *from, char *to) { do_copy_file(true, from, to); }
__device__ void Tester::do_copy_file(bool force, char *from, char *to)
{
	auto retrys = GetFileRetries(); // Maximum number of retries.
	auto delays = GetFileRetryDelay(); // Delay in ms before retrying.

	// On windows, sometimes even a [file copy -force] can fail. The cause is usually "tag-alongs" - programs like anti-virus software, automatic backup
	// tools and various explorer extensions that keep a file open a little longer than we expect, causing the delete to fail.
	//
	// The solution is to wait a short amount of time before retrying the copy.
	if (retrys > 0)
	{
		for (auto i = 0; i < retrys; i++)
		{
			//  set rc [catch {
			//    if {$force} {
			//      file copy -force $from $to
			//    } else {
			//      file copy $from $to
			//    }
			//  } msg]
			//  if {$rc==0} break
			//  if {$nDelay > 0} { after $nDelay }
		}
		//if {$rc} { error $msg }
	}
	else
	{
		if (force)
		{
			//file copy -force $from $to
		}
		else
		{
			//file copy $from $to
		}
	}
}

// Check if a file name is relative
__device__ bool Tester::is_relative_file(char *file)
{
	return false; //[expr {[file pathtype $file] != "absolute"}]
}

// If the VFS supports using the current directory, returns [pwd]; otherwise, it returns only the provided suffix string (which is empty by default).
__device__ bool Tester::test_pwd(array_t<char *> args)
{
	//  if {[llength $args] > 0} {
	//    set suffix1 [lindex $args 0]
	//    if {[llength $args] > 1} {
	//      set suffix2 [lindex $args 1]
	//    } else {
	//      set suffix2 $suffix1
	//    }
	//  } else {
	//    set suffix1 ""; set suffix2 ""
	//  }
	//  ifcapable curdir {
	//    return "[get_pwd]$suffix1"
	//  } else {
	//    return $suffix2
	//  }
	return false;
}

// Delete a file or directory
__device__ void Tester::delete_file(const char *args[], int argsLength) { do_delete_file(false, args, argsLength); }
__device__ void Tester::forcedelete(const char *args[], int argsLength) { do_delete_file(true, args, argsLength); }
__device__ void Tester::do_delete_file(bool force, const char *args[], int argsLength)
{
	auto retrys = GetFileRetries();     // Maximum number of retries.
	auto delays = GetFileRetryDelay();  // Delay in ms before retrying.

	for (auto i = 0; i < argsLength; i++)
	{
		auto filename = args[i];
		// On windows, sometimes even a [file delete -force] can fail just after a file is closed. The cause is usually "tag-alongs" - programs like
		// anti-virus software, automatic backup tools and various explorer extensions that keep a file open a little longer than we expect, causing
		// the delete to fail.
		//
		// The solution is to wait a short amount of time before retrying the delete.
		//
		if (retrys > 0)
		{
			//      for {set i 0} {$i<$nRetry} {incr i} {
			//        set rc [catch {
			if (force)
			{
				//file delete -force $filename
			}
			else
			{
				//file delete $filename
			}
			//        } msg]
			//        if {$rc==0} break
			//        if {$nDelay > 0} { after $nDelay }
			//      }
			//      if {$rc} { error $msg }
		}
		else
		{
			if (force)
			{
				//file delete -force $filename
			}
			else
			{
				//file delete $filename
			}
		}
	}
}

__device__ void Tester::execpresql(TestCtx *handle, void *args)
{
	//  trace remove execution $handle enter [list execpresql $handle]
	auto presql = (Tcl_Obj *)G.Find("perm:presql", 11);
	//if (presql)
	//	handle->DB_EVAL(handle->Encode(presql));
}

// This command should be called after loading tester.tcl from within all test scripts that are incompatible with encryption codecs.
__device__ void Tester::do_not_use_codec()
{
	_do_not_use_codec = true;
	reset_db();
}

// The following block only runs the first time this file is sourced. It does not run in slave interpreters (since the ::cmdlinearg array is
// populated before the test script is run in slave interpreters).
//#
//if {[info exists cmdlinearg]==0} {
__device__ void Tester::Initialize()
{
	G.Init();
	db = db2 = db3 = nullptr;
	//  # Parse any options specified in the $argv array. This script accepts the 
	//  # following options: 
	//  #
	//  #   --pause
	//  #   --soft-heap-limit=NN
	//  #   --maxerror=NN
	//  #   --malloctrace=N
	//  #   --backtrace=N
	//  #   --binarylog=N
	//  #   --soak=N
	//  #   --file-retries=N
	//  #   --file-retry-delay=N
	//  #   --start=[$permutation:]$testfile
	//  #   --match=$pattern
	//  #
	//  set cmdlinearg(soft-heap-limit)    0
	//  set cmdlinearg(maxerror)        1000
	//  set cmdlinearg(malloctrace)        0
	//  set cmdlinearg(backtrace)         10
	//  set cmdlinearg(binarylog)          0
	//  set cmdlinearg(soak)               0
	//  set cmdlinearg(file-retries)       0
	//  set cmdlinearg(file-retry-delay)   0
	//  set cmdlinearg(start)             ""
	//  set cmdlinearg(match)             ""

	//  set leftover [list]
	//  foreach a $argv {
	//    switch -regexp -- $a {
	//      {^-+pause$} {
	//        # Wait for user input before continuing. This is to give the user an 
	//        # opportunity to connect profiling tools to the process.
	//        puts -nonewline "Press RETURN to begin..."
	//        flush stdout
	//        gets stdin
	//      }
	//      {^-+soft-heap-limit=.+$} {
	//        foreach {dummy cmdlinearg(soft-heap-limit)} [split $a =] break
	//      }
	//      {^-+maxerror=.+$} {
	//        foreach {dummy cmdlinearg(maxerror)} [split $a =] break
	//      }
	//      {^-+malloctrace=.+$} {
	//        foreach {dummy cmdlinearg(malloctrace)} [split $a =] break
	//        if {$cmdlinearg(malloctrace)} {
	//          sqlite3_memdebug_log start
	//        }
	//      }
	//      {^-+backtrace=.+$} {
	//        foreach {dummy cmdlinearg(backtrace)} [split $a =] break
	//        sqlite3_memdebug_backtrace $value
	//      }
	//      {^-+binarylog=.+$} {
	//        foreach {dummy cmdlinearg(binarylog)} [split $a =] break
	//      }
	//      {^-+soak=.+$} {
	//        foreach {dummy cmdlinearg(soak)} [split $a =] break
	//        set ::G(issoak) $cmdlinearg(soak)
	//      }
	//      {^-+file-retries=.+$} {
	//        foreach {dummy cmdlinearg(file-retries)} [split $a =] break
	//        set ::G(file-retries) $cmdlinearg(file-retries)
	//      }
	//      {^-+file-retry-delay=.+$} {
	//        foreach {dummy cmdlinearg(file-retry-delay)} [split $a =] break
	//        set ::G(file-retry-delay) $cmdlinearg(file-retry-delay)
	//      }
	//      {^-+start=.+$} {
	//        foreach {dummy cmdlinearg(start)} [split $a =] break

	//        set ::G(start:file) $cmdlinearg(start)
	//        if {[regexp {(.*):(.*)} $cmdlinearg(start) -> s.perm s.file]} {
	//          set ::G(start:permutation) ${s.perm}
	//          set ::G(start:file)        ${s.file}
	//        }
	//        if {$::G(start:file) == ""} {unset ::G(start:file)}
	//      }
	//      {^-+match=.+$} {
	//        foreach {dummy cmdlinearg(match)} [split $a =] break

	//        set ::G(match) $cmdlinearg(match)
	//        if {$::G(match) == ""} {unset ::G(match)}
	//      }
	//      default {
	//        lappend leftover $a
	//      }
	//    }
	//  }
	//  set argv $leftover

	//  Install the malloc layer used to inject OOM errors. And the 'automatic' extensions. This only needs to be done once for the process.
	//  sqlite3_shutdown 
	//  install_malloc_faultsim 1 
	Main::Initialize();
	//  autoinstall_test_functions

	//  If the --binarylog option was specified, create the logging VFS. This call installs the new VFS as the default for all SQLite connections.
	//  if {$cmdlinearg(binarylog)} {
	//    vfslog new binarylog {} vfslog.bin
	//  }

	//  # Set the backtrace depth, if malloc tracing is enabled.
	//  if {$cmdlinearg(malloctrace)}  {
	//    sqlite3_memdebug_backtrace $cmdlinearg(backtrace)
	//  }

	reset_db();
}

// Update the soft-heap-limit each time this script is run. In that way if an individual test file changes the soft-heap-limit, it
// will be reset at the start of the next test file.
//sqlite3_soft_heap_limit $cmdlinearg(soft-heap-limit)

// Create a test database
__device__ void Tester::reset_db()
{
	db->CLOSE(nullptr);
	const char *args[] = {"test.db", "test.db-journal", "test.db-wal"};
	forcedelete(args, 3);
	sqlite3(db, Y("./test.db"));
	//  set ::DB [sqlite3_connection_pointer db]
	if (_SETUP_SQL)
		db->EVAL(Zc(_SETUP_SQL));
}


//# Abort early if this script has been run before.
//#
//if {[info exists TC(count)]} return

//# Make sure memory statistics are enabled.
//#
//sqlite3_config_memstatus 1

//# Initialize the test counters and set up commands to access them.
//# Or, if this is a slave interpreter, set up aliases to write the
//# counters in the parent interpreter.
//#
//if {0==[info exists ::SLAVE]} {
//  set TC(errors)    0
//  set TC(count)     0
//  set TC(fail_list) [list]
//  set TC(omit_list) [list]

//  proc set_test_counter {counter args} {
//    if {[llength $args]} {
//      set ::TC($counter) [lindex $args 0]
//    }
//    set ::TC($counter)
//  }
//}

//# Record the fact that a sequence of tests were omitted.
//#
//proc omit_test {name reason {append 1}} {
//  set omitList [set_test_counter omit_list]
//  if {$append} {
//    lappend omitList [list $name $reason]
//  }
//  set_test_counter omit_list $omitList
//}

//# Record the fact that a test failed.
//#
//proc fail_test {name} {
//  set f [set_test_counter fail_list]
//  lappend f $name
//  set_test_counter fail_list $f
//  set_test_counter errors [expr [set_test_counter errors] + 1]

//  set nFail [set_test_counter errors]
//  if {$nFail>=$::cmdlinearg(maxerror)} {
//    puts "*** Giving up..."
//    finalize_testing
//  }
//}

//# Increment the number of tests run
//#
//proc incr_ntest {} {
//  set_test_counter count [expr [set_test_counter count] + 1]
//}


// Invoke the do_test procedure to run a single test 
//#


//proc catchcmd {db {cmd ""}} {
//  global CLI
//  set out [open cmds.txt w]
//  puts $out $cmd
//  close $out
//  set line "exec $CLI $db < cmds.txt"
//  set rc [catch { eval $line } msg]
//  list $rc $msg
//}

//proc filepath_normalize {p} {
//  # test cases should be written to assume "unix"-like file paths
//  if {$::tcl_platform(platform)!="unix"} {
//    # lreverse*2 as a hack to remove any unneeded {} after the string map
//    lreverse [lreverse [string map {\\ /} [regsub -nocase -all {[a-z]:[/\\]+} $p {/}]]]
//  } {
//    set p
//  }
//}
//proc do_filepath_test {name cmd expected} {
//  uplevel [list do_test $name [
//    subst -nocommands { filepath_normalize [ $cmd ] }
//  ] [filepath_normalize $expected]]
//}

//proc realnum_normalize {r} {
//  # different TCL versions display floating point values differently.
//  string map {1.#INF inf Inf inf .0e e} [regsub -all {(e[+-])0+} $r {\1}]
//}
//proc do_realnum_test {name cmd expected} {
//  uplevel [list do_test $name [
//    subst -nocommands { realnum_normalize [ $cmd ] }
//  ] [realnum_normalize $expected]]
//}

//proc fix_testname {varname} {
//  upvar $varname testname
//  if {[info exists ::testprefix] 
//   && [string is digit [string range $testname 0 0]]
//  } {
//    set testname "${::testprefix}-$testname"
//  }
//}

//proc do_execsql_test {testname sql {result {}}} {
//  fix_testname testname
//  uplevel do_test [list $testname] [list "execsql {$sql}"] [list [list {*}$result]]
//}
//proc do_catchsql_test {testname sql result} {
//  fix_testname testname
//  uplevel do_test [list $testname] [list "catchsql {$sql}"] [list $result]
//}
//proc do_eqp_test {name sql res} {
//  uplevel do_execsql_test $name [list "EXPLAIN QUERY PLAN $sql"] [list $res]
//}

//#-------------------------------------------------------------------------
//#   Usage: do_select_tests PREFIX ?SWITCHES? TESTLIST
//#
//# Where switches are:
//#
//#   -errorformat FMTSTRING
//#   -count
//#   -query SQL
//#   -tclquery TCL
//#   -repair TCL
//#
//proc do_select_tests {prefix args} {

//  set testlist [lindex $args end]
//  set switches [lrange $args 0 end-1]

//  set errfmt ""
//  set countonly 0
//  set tclquery ""
//  set repair ""

//  for {set i 0} {$i < [llength $switches]} {incr i} {
//    set s [lindex $switches $i]
//    set n [string length $s]
//    if {$n>=2 && [string equal -length $n $s "-query"]} {
//      set tclquery [list execsql [lindex $switches [incr i]]]
//    } elseif {$n>=2 && [string equal -length $n $s "-tclquery"]} {
//      set tclquery [lindex $switches [incr i]]
//    } elseif {$n>=2 && [string equal -length $n $s "-errorformat"]} {
//      set errfmt [lindex $switches [incr i]]
//    } elseif {$n>=2 && [string equal -length $n $s "-repair"]} {
//      set repair [lindex $switches [incr i]]
//    } elseif {$n>=2 && [string equal -length $n $s "-count"]} {
//      set countonly 1
//    } else {
//      error "unknown switch: $s"
//    }
//  }

//  if {$countonly && $errfmt!=""} {
//    error "Cannot use -count and -errorformat together"
//  }
//  set nTestlist [llength $testlist]
//  if {$nTestlist%3 || $nTestlist==0 } {
//    error "SELECT test list contains [llength $testlist] elements"
//  }

//  eval $repair
//  foreach {tn sql res} $testlist {
//    if {$tclquery != ""} {
//      execsql $sql
//      uplevel do_test ${prefix}.$tn [list $tclquery] [list [list {*}$res]]
//    } elseif {$countonly} {
//      set nRow 0
//      db eval $sql {incr nRow}
//      uplevel do_test ${prefix}.$tn [list [list set {} $nRow]] [list $res]
//    } elseif {$errfmt==""} {
//      uplevel do_execsql_test ${prefix}.${tn} [list $sql] [list [list {*}$res]]
//    } else {
//      set res [list 1 [string trim [format $errfmt {*}$res]]]
//      uplevel do_catchsql_test ${prefix}.${tn} [list $sql] [list $res]
//    }
//    eval $repair
//  }

//}

//proc delete_all_data {} {
//  db eval {SELECT tbl_name AS t FROM sqlite_master WHERE type = 'table'} {
//    db eval "DELETE FROM '[string map {' ''} $t]'"
//  }
//}

//# Run an SQL script.  
//# Return the number of microseconds per statement.
//#
//proc speed_trial {name numstmt units sql} {
//  puts -nonewline [format {%-21.21s } $name...]
//  flush stdout
//  set speed [time {sqlite3_exec_nr db $sql}]
//  set tm [lindex $speed 0]
//  if {$tm == 0} {
//    set rate [format %20s "many"]
//  } else {
//    set rate [format %20.5f [expr {1000000.0*$numstmt/$tm}]]
//  }
//  set u2 $units/s
//  puts [format {%12d uS %s %s} $tm $rate $u2]
//  global total_time
//  set total_time [expr {$total_time+$tm}]
//  lappend ::speed_trial_times $name $tm
//}
//proc speed_trial_tcl {name numstmt units script} {
//  puts -nonewline [format {%-21.21s } $name...]
//  flush stdout
//  set speed [time {eval $script}]
//  set tm [lindex $speed 0]
//  if {$tm == 0} {
//    set rate [format %20s "many"]
//  } else {
//    set rate [format %20.5f [expr {1000000.0*$numstmt/$tm}]]
//  }
//  set u2 $units/s
//  puts [format {%12d uS %s %s} $tm $rate $u2]
//  global total_time
//  set total_time [expr {$total_time+$tm}]
//  lappend ::speed_trial_times $name $tm
//}
//proc speed_trial_init {name} {
//  global total_time
//  set total_time 0
//  set ::speed_trial_times [list]
//  sqlite3 versdb :memory:
//  set vers [versdb one {SELECT sqlite_source_id()}]
//  versdb close
//  puts "SQLite $vers"
//}
//proc speed_trial_summary {name} {
//  global total_time
//  puts [format {%-21.21s %12d uS TOTAL} $name $total_time]

//  if { 0 } {
//    sqlite3 versdb :memory:
//    set vers [lindex [versdb one {SELECT sqlite_source_id()}] 0]
//    versdb close
//    puts "CREATE TABLE IF NOT EXISTS time(version, script, test, us);"
//    foreach {test us} $::speed_trial_times {
//      puts "INSERT INTO time VALUES('$vers', '$name', '$test', $us);"
//    }
//  }
//}

// Run this routine last
__device__ void Tester::finish_test()
{
	//if (db) { DB_Closedb->Close(); db = nullptr; }
	//if (db2) { Main::Close(db2); db2 = nullptr; }
	//if (db3) { Main::Close(db3); db3 = nullptr; }
	if (!_SLAVE) finalize_testing();
}
__device__ void Tester::finalize_testing()
{
	//  global sqlite_open_file_count

	//  set omitList [set_test_counter omit_list]

	//if (db) { Main::Close(db); db = nullptr; }
	//if (db2) { Main::Close(db2); db2 = nullptr; }
	//if (db3) { Main::Close(db3); db3 = nullptr; }

	//  vfs_unlink_test
	//  sqlite3 db {}
	//  # sqlite3_clear_tsd_memdebug
	//  db close
	//  sqlite3_reset_auto_extension

	//  sqlite3_soft_heap_limit 0
	//  set nTest [incr_ntest]
	//  set nErr [set_test_counter errors]

	//  puts "$nErr errors out of $nTest tests"
	//  if {$nErr>0} {
	//    puts "Failures on these tests: [set_test_counter fail_list]"
	//  }
	//  run_thread_tests 1
	//  if {[llength $omitList]>0} {
	//    puts "Omitted test cases:"
	//    set prec {}
	//    foreach {rec} [lsort $omitList] {
	//      if {$rec==$prec} continue
	//      set prec $rec
	//      puts [format {  %-12s %s} [lindex $rec 0] [lindex $rec 1]]
	//    }
	//  }
	//  if {$nErr>0 && ![working_64bit_int]} {
	//    puts "******************************************************************"
	//    puts "N.B.:  The version of TCL that you used to build this test harness"
	//    puts "is defective in that it does not support 64-bit integers.  Some or"
	//    puts "all of the test failures above might be a result from this defect"
	//    puts "in your TCL build."
	//    puts "******************************************************************"
	//  }
	//  if {$::cmdlinearg(binarylog)} {
	//    vfslog finalize binarylog
	//  }
	//  if {$sqlite_open_file_count} {
	//    puts "$sqlite_open_file_count files were left open"
	//    incr nErr
	//  }
	//  if {[lindex [sqlite3_status SQLITE_STATUS_MALLOC_COUNT 0] 1]>0 ||
	//              [sqlite3_memory_used]>0} {
	//    puts "Unfreed memory: [sqlite3_memory_used] bytes in\
	//         [lindex [sqlite3_status SQLITE_STATUS_MALLOC_COUNT 0] 1] allocations"
	//    incr nErr
	//    ifcapable memdebug||mem5||(mem3&&debug) {
	//      puts "Writing unfreed memory log to \"./memleak.txt\""
	//      sqlite3_memdebug_dump ./memleak.txt
	//    }
	//  } else {
	//    puts "All memory allocations freed - no leaks"
	//    ifcapable memdebug||mem5 {
	//      sqlite3_memdebug_dump ./memusage.txt
	//    }
	//  }
	show_memstats();
	//  puts "Maximum memory usage: [sqlite3_memory_highwater 1] bytes"
	//  puts "Current memory usage: [sqlite3_memory_highwater] bytes"
	//  if {[info commands sqlite3_memdebug_malloc_count] ne ""} {
	//    puts "Number of malloc()  : [sqlite3_memdebug_malloc_count] calls"
	//  }
	//  if {$::cmdlinearg(malloctrace)} {
	//    puts "Writing mallocs.sql..."
	//    memdebug_log_sql
	//    sqlite3_memdebug_log stop
	//    sqlite3_memdebug_log clear

	//    if {[sqlite3_memory_used]>0} {
	//      puts "Writing leaks.sql..."
	//      sqlite3_memdebug_log sync
	//      memdebug_log_sql leaks.sql
	//    }
	//  }
	//  foreach f [glob -nocomplain test.db-*-journal] {
	//    forcedelete $f
	//  }
	//  foreach f [glob -nocomplain test.db-mj*] {
	//    forcedelete $f
	//  }
	//  exit [expr {$nErr>0}]
}

// Display memory statistics for analysis and debugging purposes.
__device__ void Tester::show_memstats()
{
	int x1, x2;
	bool x0 = _status(STATUS_MEMORY_USED, &x1, &x2, false);
	int y1, y2;
	bool y0 = _status(STATUS_MALLOC_SIZE, &y1, &y2, false);
	char val[100];
	__snprintf(val, sizeof(val), "now %10d  max %10d  max-size %10d", x1, x2, y2);
	printf("Memory used:          %s", val);
	x0 = _status(STATUS_MALLOC_COUNT, &x1, &x2, false);
	__snprintf(val, sizeof(val), "now %10d  max %10d", x1, x2);
	printf("Allocation count:     %s", val);
	x0 = _status(STATUS_PAGECACHE_USED, &x1, &x2, false);
	y0 = _status(STATUS_PAGECACHE_SIZE, &y1, &y2, false);
	__snprintf(val, sizeof(val), "now %10d  max %10d  max-size %10d", x1, x2, y2);
	printf("Page-cache used:      %s", val);
	x0 = _status(STATUS_PAGECACHE_OVERFLOW, &x1, &x2, false);
	__snprintf(val, sizeof(val), "now %10d  max %10d", x1, x2);
	printf("Page-cache overflow:  %s", val);
	x0 = _status(STATUS_SCRATCH_USED, &x1, &x2, false);	
	__snprintf(val, sizeof(val), "now %10d  max %10d", x1, x2);
	printf("Scratch memory used:  %s", val);
	x0 = _status(STATUS_SCRATCH_OVERFLOW, &x1, &x2, false);
	y0 = _status(STATUS_SCRATCH_SIZE, &y1, &y2, false);
	__snprintf(val, sizeof(val), "now %10d  max %10d  max-size %10d", x1, x2, y2);
	printf("Scratch overflow:     %s", val);
#if yytrackmaxstackdepth
	x0 = _status(STATUS_PARSER_STACK, &x1, &x2, false);
	__snprintf(val, sizeof(val), "               max %10d", x2);
	printf("Parser stack depth:    %s", val);
#endif
}

// A procedure to execute SQL
__device__ void Tester::execsql(const char *sql, TestCtx *db)
{
	printf("SQL = %s\n", sql);
	//db->DB_EVAL(db->Encode(sql));
}

// Execute SQL and catch exceptions.
//proc catchsql {sql {db db}} {
//  # puts "SQL = $sql"
//  set r [catch [list uplevel [list $db eval $sql]] msg]
//  lappend r $msg
//  return $r
//}

//# Do an VDBE code dump on the SQL given
//#
//proc explain {sql {db db}} {
//  puts ""
//  puts "addr  opcode        p1      p2      p3      p4               p5  #"
//  puts "----  ------------  ------  ------  ------  ---------------  --  -"
//  $db eval "explain $sql" {} {
//    puts [format {%-4d  %-12.12s  %-6d  %-6d  %-6d  % -17s %s  %s} \
//      $addr $opcode $p1 $p2 $p3 $p4 $p5 $comment
//    ]
//  }
//}

//# Show the VDBE program for an SQL statement but omit the Trace
//# opcode at the beginning.  This procedure can be used to prove
//# that different SQL statements generate exactly the same VDBE code.
//#
//proc explain_no_trace {sql} {
//  set tr [db eval "EXPLAIN $sql"]
//  return [lrange $tr 7 end]
//}

//# Another procedure to execute SQL.  This one includes the field
//# names in the returned list.
//#
//proc execsql2 {sql} {
//  set result {}
//  db eval $sql data {
//    foreach f $data(*) {
//      lappend result $f $data($f)
//    }
//  }
//  return $result
//}

//# Use the non-callback API to execute multiple SQL statements
//#
//proc stepsql {dbptr sql} {
//  set sql [string trim $sql]
//  set r 0
//  while {[string length $sql]>0} {
//    if {[catch {sqlite3_prepare $dbptr $sql -1 sqltail} vm]} {
//      return [list 1 $vm]
//    }
//    set sql [string trim $sqltail]
//#    while {[sqlite_step $vm N VAL COL]=="SQLITE_ROW"} {
//#      foreach v $VAL {lappend r $v}
//#    }
//    while {[sqlite3_step $vm]=="SQLITE_ROW"} {
//      for {set i 0} {$i<[sqlite3_data_count $vm]} {incr i} {
//        lappend r [sqlite3_column_text $vm $i]
//      }
//    }
//    if {[catch {sqlite3_finalize $vm} errmsg]} {
//      return [list 1 $errmsg]
//    }
//  }
//  return $r
//}

//# Do an integrity check of the entire database
//#
//proc integrity_check {name {db db}} {
//  ifcapable integrityck {
//    do_test $name [list execsql {PRAGMA integrity_check} $db] {ok}
//  }
//}

//# Check the extended error code
//#
//proc verify_ex_errcode {name expected {db db}} {
//  do_test $name [list sqlite3_extended_errcode $db] $expected
//}


//# Return true if the SQL statement passed as the second argument uses a
//# statement transaction.
//#
//proc sql_uses_stmt {db sql} {
//  set stmt [sqlite3_prepare $db $sql -1 dummy]
//  set uses [uses_stmt_journal $stmt]
//  sqlite3_finalize $stmt
//  return $uses
//}

//proc fix_ifcapable_expr {expr} {
//  set ret ""
//  set state 0
//  for {set i 0} {$i < [string length $expr]} {incr i} {
//    set char [string range $expr $i $i]
//    set newstate [expr {[string is alnum $char] || $char eq "_"}]
//    if {$newstate && !$state} {
//      append ret {$::sqlite_options(}
//    }
//    if {!$newstate && $state} {
//      append ret )
//    }
//    append ret $char
//    set state $newstate
//  }
//  if {$state} {append ret )}
//  return $ret
//}

//# Returns non-zero if the capabilities are present; zero otherwise.
//#
//proc capable {expr} {
//  set e [fix_ifcapable_expr $expr]; return [expr ($e)]
//}

//# Evaluate a boolean expression of capabilities.  If true, execute the
//# code.  Omit the code if false.
//#
//proc ifcapable {expr code {else ""} {elsecode ""}} {
//  #regsub -all {[a-z_0-9]+} $expr {$::sqlite_options(&)} e2
//  set e2 [fix_ifcapable_expr $expr]
//  if ($e2) {
//    set c [catch {uplevel 1 $code} r]
//  } else {
//    set c [catch {uplevel 1 $elsecode} r]
//  }
//  return -code $c $r
//}

//# This proc execs a seperate process that crashes midway through executing
//# the SQL script $sql on database test.db.
//#
//# The crash occurs during a sync() of file $crashfile. When the crash
//# occurs a random subset of all unsynced writes made by the process are
//# written into the files on disk. Argument $crashdelay indicates the
//# number of file syncs to wait before crashing.
//#
//# The return value is a list of two elements. The first element is a
//# boolean, indicating whether or not the process actually crashed or
//# reported some other error. The second element in the returned list is the
//# error message. This is "child process exited abnormally" if the crash
//# occurred.
//#
//#   crashsql -delay CRASHDELAY -file CRASHFILE ?-blocksize BLOCKSIZE? $sql
//#
//proc crashsql {args} {

//  set blocksize ""
//  set crashdelay 1
//  set prngseed 0
//  set tclbody {}
//  set crashfile ""
//  set dc ""
//  set sql [lindex $args end]

//  for {set ii 0} {$ii < [llength $args]-1} {incr ii 2} {
//    set z [lindex $args $ii]
//    set n [string length $z]
//    set z2 [lindex $args [expr $ii+1]]

//    if     {$n>1 && [string first $z -delay]==0}     {set crashdelay $z2} \
//    elseif {$n>1 && [string first $z -seed]==0}      {set prngseed $z2} \
//    elseif {$n>1 && [string first $z -file]==0}      {set crashfile $z2}  \
//    elseif {$n>1 && [string first $z -tclbody]==0}   {set tclbody $z2}  \
//    elseif {$n>1 && [string first $z -blocksize]==0} {set blocksize "-s $z2" } \
//    elseif {$n>1 && [string first $z -characteristics]==0} {set dc "-c {$z2}" } \
//    else   { error "Unrecognized option: $z" }
//  }

//  if {$crashfile eq ""} {
//    error "Compulsory option -file missing"
//  }

//  # $crashfile gets compared to the native filename in 
//  # cfSync(), which can be different then what TCL uses by
//  # default, so here we force it to the "nativename" format.
//  set cfile [string map {\\ \\\\} [file nativename [file join [get_pwd] $crashfile]]]

//  set f [open crash.tcl w]
//  puts $f "sqlite3_crash_enable 1"
//  puts $f "sqlite3_crashparams $blocksize $dc $crashdelay $cfile"
//  puts $f "sqlite3_test_control_pending_byte $::sqlite_pending_byte"
//  puts $f "sqlite3 db test.db -vfs crash"

//  # This block sets the cache size of the main database to 10
//  # pages. This is done in case the build is configured to omit
//  # "PRAGMA cache_size".
//  puts $f {db eval {SELECT * FROM sqlite_master;}}
//  puts $f {set bt [btree_from_db db]}
//  puts $f {btree_set_cache_size $bt 10}
//  if {$prngseed} {
//    set seed [expr {$prngseed%10007+1}]
//    # puts seed=$seed
//    puts $f "db eval {SELECT randomblob($seed)}"
//  }

//  if {[string length $tclbody]>0} {
//    puts $f $tclbody
//  }
//  if {[string length $sql]>0} {
//    puts $f "db eval {"
//    puts $f   "$sql"
//    puts $f "}"
//  }
//  close $f
//  set r [catch {
//    exec [info nameofexec] crash.tcl >@stdout
//  } msg]

//  # Windows/ActiveState TCL returns a slightly different
//  # error message.  We map that to the expected message
//  # so that we don't have to change all of the test
//  # cases.
//  if {$::tcl_platform(platform)=="windows"} {
//    if {$msg=="child killed: unknown signal"} {
//      set msg "child process exited abnormally"
//    }
//  }

//  lappend r $msg
//}

//proc run_ioerr_prep {} {
//  set ::sqlite_io_error_pending 0
//  catch {db close}
//  catch {db2 close}
//  catch {forcedelete test.db}
//  catch {forcedelete test.db-journal}
//  catch {forcedelete test2.db}
//  catch {forcedelete test2.db-journal}
//  set ::DB [sqlite3 db test.db; sqlite3_connection_pointer db]
//  sqlite3_extended_result_codes $::DB $::ioerropts(-erc)
//  if {[info exists ::ioerropts(-tclprep)]} {
//    eval $::ioerropts(-tclprep)
//  }
//  if {[info exists ::ioerropts(-sqlprep)]} {
//    execsql $::ioerropts(-sqlprep)
//  }
//  expr 0
//}

//# Usage: do_ioerr_test <test number> <options...>
//#
//# This proc is used to implement test cases that check that IO errors
//# are correctly handled. The first argument, <test number>, is an integer 
//# used to name the tests executed by this proc. Options are as follows:
//#
//#     -tclprep          TCL script to run to prepare test.
//#     -sqlprep          SQL script to run to prepare test.
//#     -tclbody          TCL script to run with IO error simulation.
//#     -sqlbody          TCL script to run with IO error simulation.
//#     -exclude          List of 'N' values not to test.
//#     -erc              Use extended result codes
//#     -persist          Make simulated I/O errors persistent
//#     -start            Value of 'N' to begin with (default 1)
//#
//#     -cksum            Boolean. If true, test that the database does
//#                       not change during the execution of the test case.
//#
//proc do_ioerr_test {testname args} {

//  set ::ioerropts(-start) 1
//  set ::ioerropts(-cksum) 0
//  set ::ioerropts(-erc) 0
//  set ::ioerropts(-count) 100000000
//  set ::ioerropts(-persist) 1
//  set ::ioerropts(-ckrefcount) 0
//  set ::ioerropts(-restoreprng) 1
//  array set ::ioerropts $args

//  # TEMPORARY: For 3.5.9, disable testing of extended result codes. There are
//  # a couple of obscure IO errors that do not return them.
//  set ::ioerropts(-erc) 0

//  # Create a single TCL script from the TCL and SQL specified
//  # as the body of the test.
//  set ::ioerrorbody {}
//  if {[info exists ::ioerropts(-tclbody)]} {
//    append ::ioerrorbody "$::ioerropts(-tclbody)\n"
//  }
//  if {[info exists ::ioerropts(-sqlbody)]} {
//    append ::ioerrorbody "db eval {$::ioerropts(-sqlbody)}"
//  }

//  save_prng_state
//  if {$::ioerropts(-cksum)} {
//    run_ioerr_prep
//    eval $::ioerrorbody
//    set ::goodcksum [cksum]
//  }

//  set ::go 1
//  #reset_prng_state
//  for {set n $::ioerropts(-start)} {$::go} {incr n} {
//    set ::TN $n
//    incr ::ioerropts(-count) -1
//    if {$::ioerropts(-count)<0} break

//    # Skip this IO error if it was specified with the "-exclude" option.
//    if {[info exists ::ioerropts(-exclude)]} {
//      if {[lsearch $::ioerropts(-exclude) $n]!=-1} continue
//    }
//    if {$::ioerropts(-restoreprng)} {
//      restore_prng_state
//    }

//    # Delete the files test.db and test2.db, then execute the TCL and 
//    # SQL (in that order) to prepare for the test case.
//    do_test $testname.$n.1 {
//      run_ioerr_prep
//    } {0}

//    # Read the 'checksum' of the database.
//    if {$::ioerropts(-cksum)} {
//      set ::checksum [cksum]
//    }

//    # Set the Nth IO error to fail.
//    do_test $testname.$n.2 [subst {
//      set ::sqlite_io_error_persist $::ioerropts(-persist)
//      set ::sqlite_io_error_pending $n
//    }] $n

//    # Execute the TCL script created for the body of this test. If
//    # at least N IO operations performed by SQLite as a result of 
//    # the script, the Nth will fail.
//    do_test $testname.$n.3 {
//      set ::sqlite_io_error_hit 0
//      set ::sqlite_io_error_hardhit 0
//      set r [catch $::ioerrorbody msg]
//      set ::errseen $r
//      set rc [sqlite3_errcode $::DB]
//      if {$::ioerropts(-erc)} {
//        # If we are in extended result code mode, make sure all of the
//        # IOERRs we get back really do have their extended code values.
//        # If an extended result code is returned, the sqlite3_errcode
//        # TCLcommand will return a string of the form:  SQLITE_IOERR+nnnn
//        # where nnnn is a number
//        if {[regexp {^SQLITE_IOERR} $rc] && ![regexp {IOERR\+\d} $rc]} {
//          return $rc
//        }
//      } else {
//        # If we are not in extended result code mode, make sure no
//        # extended error codes are returned.
//        if {[regexp {\+\d} $rc]} {
//          return $rc
//        }
//      }
//      # The test repeats as long as $::go is non-zero.  $::go starts out
//      # as 1.  When a test runs to completion without hitting an I/O
//      # error, that means there is no point in continuing with this test
//      # case so set $::go to zero.
//      #
//      if {$::sqlite_io_error_pending>0} {
//        set ::go 0
//        set q 0
//        set ::sqlite_io_error_pending 0
//      } else {
//        set q 1
//      }

//      set s [expr $::sqlite_io_error_hit==0]
//      if {$::sqlite_io_error_hit>$::sqlite_io_error_hardhit && $r==0} {
//        set r 1
//      }
//      set ::sqlite_io_error_hit 0

//      # One of two things must have happened. either
//      #   1.  We never hit the IO error and the SQL returned OK
//      #   2.  An IO error was hit and the SQL failed
//      #
//      #puts "s=$s r=$r q=$q"
//      expr { ($s && !$r && !$q) || (!$s && $r && $q) }
//    } {1}

//    set ::sqlite_io_error_hit 0
//    set ::sqlite_io_error_pending 0

//    # Check that no page references were leaked. There should be 
//    # a single reference if there is still an active transaction, 
//    # or zero otherwise.
//    #
//    # UPDATE: If the IO error occurs after a 'BEGIN' but before any
//    # locks are established on database files (i.e. if the error 
//    # occurs while attempting to detect a hot-journal file), then
//    # there may 0 page references and an active transaction according
//    # to [sqlite3_get_autocommit].
//    #
//    if {$::go && $::sqlite_io_error_hardhit && $::ioerropts(-ckrefcount)} {
//      do_test $testname.$n.4 {
//        set bt [btree_from_db db]
//        db_enter db
//        array set stats [btree_pager_stats $bt]
//        db_leave db
//        set nRef $stats(ref)
//        expr {$nRef == 0 || ([sqlite3_get_autocommit db]==0 && $nRef == 1)}
//      } {1}
//    }

//    # If there is an open database handle and no open transaction, 
//    # and the pager is not running in exclusive-locking mode,
//    # check that the pager is in "unlocked" state. Theoretically,
//    # if a call to xUnlock() failed due to an IO error the underlying
//    # file may still be locked.
//    #
//    ifcapable pragma {
//      if { [info commands db] ne ""
//        && $::ioerropts(-ckrefcount)
//        && [db one {pragma locking_mode}] eq "normal"
//        && [sqlite3_get_autocommit db]
//      } {
//        do_test $testname.$n.5 {
//          set bt [btree_from_db db]
//          db_enter db
//          array set stats [btree_pager_stats $bt]
//          db_leave db
//          set stats(state)
//        } 0
//      }
//    }

//    # If an IO error occurred, then the checksum of the database should
//    # be the same as before the script that caused the IO error was run.
//    #
//    if {$::go && $::sqlite_io_error_hardhit && $::ioerropts(-cksum)} {
//      do_test $testname.$n.6 {
//        catch {db close}
//        catch {db2 close}
//        set ::DB [sqlite3 db test.db; sqlite3_connection_pointer db]
//        set nowcksum [cksum]
//        set res [expr {$nowcksum==$::checksum || $nowcksum==$::goodcksum}]
//        if {$res==0} {
//          puts "now=$nowcksum"
//          puts "the=$::checksum"
//          puts "fwd=$::goodcksum"
//        }
//        set res
//      } 1
//    }

//    set ::sqlite_io_error_hardhit 0
//    set ::sqlite_io_error_pending 0
//    if {[info exists ::ioerropts(-cleanup)]} {
//      catch $::ioerropts(-cleanup)
//    }
//  }
//  set ::sqlite_io_error_pending 0
//  set ::sqlite_io_error_persist 0
//  unset ::ioerropts
//}

//# Return a checksum based on the contents of the main database associated
//# with connection $db
//#
//proc cksum {{db db}} {
//  set txt [$db eval {
//      SELECT name, type, sql FROM sqlite_master order by name
//  }]\n
//  foreach tbl [$db eval {
//      SELECT name FROM sqlite_master WHERE type='table' order by name
//  }] {
//    append txt [$db eval "SELECT * FROM $tbl"]\n
//  }
//  foreach prag {default_synchronous default_cache_size} {
//    append txt $prag-[$db eval "PRAGMA $prag"]\n
//  }
//  set cksum [string length $txt]-[md5 $txt]
//  # puts $cksum-[file size test.db]
//  return $cksum
//}

//# Generate a checksum based on the contents of the main and temp tables
//# database $db. If the checksum of two databases is the same, and the
//# integrity-check passes for both, the two databases are identical.
//#
//proc allcksum {{db db}} {
//  set ret [list]
//  ifcapable tempdb {
//    set sql {
//      SELECT name FROM sqlite_master WHERE type = 'table' UNION
//      SELECT name FROM sqlite_temp_master WHERE type = 'table' UNION
//      SELECT 'sqlite_master' UNION
//      SELECT 'sqlite_temp_master' ORDER BY 1
//    }
//  } else {
//    set sql {
//      SELECT name FROM sqlite_master WHERE type = 'table' UNION
//      SELECT 'sqlite_master' ORDER BY 1
//    }
//  }
//  set tbllist [$db eval $sql]
//  set txt {}
//  foreach tbl $tbllist {
//    append txt [$db eval "SELECT * FROM $tbl"]
//  }
//  foreach prag {default_cache_size} {
//    append txt $prag-[$db eval "PRAGMA $prag"]\n
//  }
//  # puts txt=$txt
//  return [md5 $txt]
//}

//# Generate a checksum based on the contents of a single database with
//# a database connection.  The name of the database is $dbname.  
//# Examples of $dbname are "temp" or "main".
//#
//proc dbcksum {db dbname} {
//  if {$dbname=="temp"} {
//    set master sqlite_temp_master
//  } else {
//    set master $dbname.sqlite_master
//  }
//  set alltab [$db eval "SELECT name FROM $master WHERE type='table'"]
//  set txt [$db eval "SELECT * FROM $master"]\n
//  foreach tab $alltab {
//    append txt [$db eval "SELECT * FROM $dbname.$tab"]\n
//  }
//  return [md5 $txt]
//}

//proc memdebug_log_sql {{filename mallocs.sql}} {

//  set data [sqlite3_memdebug_log dump]
//  set nFrame [expr [llength [lindex $data 0]]-2]
//  if {$nFrame < 0} { return "" }

//  set database temp

//  set tbl "CREATE TABLE ${database}.malloc(zTest, nCall, nByte, lStack);"

//  set sql ""
//  foreach e $data {
//    set nCall [lindex $e 0]
//    set nByte [lindex $e 1]
//    set lStack [lrange $e 2 end]
//    append sql "INSERT INTO ${database}.malloc VALUES"
//    append sql "('test', $nCall, $nByte, '$lStack');\n"
//    foreach f $lStack {
//      set frames($f) 1
//    }
//  }

//  set tbl2 "CREATE TABLE ${database}.frame(frame INTEGER PRIMARY KEY, line);\n"
//  set tbl3 "CREATE TABLE ${database}.file(name PRIMARY KEY, content);\n"

//  foreach f [array names frames] {
//    set addr [format %x $f]
//    set cmd "addr2line -e [info nameofexec] $addr"
//    set line [eval exec $cmd]
//    append sql "INSERT INTO ${database}.frame VALUES($f, '$line');\n"

//    set file [lindex [split $line :] 0]
//    set files($file) 1
//  }

//  foreach f [array names files] {
//    set contents ""
//    catch {
//      set fd [open $f]
//      set contents [read $fd]
//      close $fd
//    }
//    set contents [string map {' ''} $contents]
//    append sql "INSERT INTO ${database}.file VALUES('$f', '$contents');\n"
//  }

//  set fd [open $filename w]
//  puts $fd "BEGIN; ${tbl}${tbl2}${tbl3}${sql} ; COMMIT;"
//  close $fd
//}

//# Drop all tables in database [db]
//proc drop_all_tables {{db db}} {
//  ifcapable trigger&&foreignkey {
//    set pk [$db one "PRAGMA foreign_keys"]
//    $db eval "PRAGMA foreign_keys = OFF"
//  }
//  foreach {idx name file} [db eval {PRAGMA database_list}] {
//    if {$idx==1} {
//      set master sqlite_temp_master
//    } else {
//      set master $name.sqlite_master
//    }
//    foreach {t type} [$db eval "
//      SELECT name, type FROM $master
//      WHERE type IN('table', 'view') AND name NOT LIKE 'sqliteX_%' ESCAPE 'X'
//    "] {
//      $db eval "DROP $type \"$t\""
//    }
//  }
//  ifcapable trigger&&foreignkey {
//    $db eval "PRAGMA foreign_keys = $pk"
//  }
//}

//#-------------------------------------------------------------------------
//# If a test script is executed with global variable $::G(perm:name) set to
//# "wal", then the tests are run in WAL mode. Otherwise, they should be run 
//# in rollback mode. The following Tcl procs are used to make this less 
//# intrusive:
//#
//#   wal_set_journal_mode ?DB?
//#
//#     If running a WAL test, execute "PRAGMA journal_mode = wal" using
//#     connection handle DB. Otherwise, this command is a no-op.
//#
//#   wal_check_journal_mode TESTNAME ?DB?
//#
//#     If running a WAL test, execute a tests case that fails if the main
//#     database for connection handle DB is not currently a WAL database.
//#     Otherwise (if not running a WAL permutation) this is a no-op.
//#
//#   wal_is_wal_mode
//#   
//#     Returns true if this test should be run in WAL mode. False otherwise.
//# 
//proc wal_is_wal_mode {} {
//  expr {[permutation] eq "wal"}
//}
//proc wal_set_journal_mode {{db db}} {
//  if { [wal_is_wal_mode] } {
//    $db eval "PRAGMA journal_mode = WAL"
//  }
//}
//proc wal_check_journal_mode {testname {db db}} {
//  if { [wal_is_wal_mode] } {
//    $db eval { SELECT * FROM sqlite_master }
//    do_test $testname [list $db eval "PRAGMA main.journal_mode"] {wal}
//  }
//}

//proc permutation {} {
//  set perm ""
//  catch {set perm $::G(perm:name)}
//  set perm
//}
//proc presql {} {
//  set presql ""
//  catch {set presql $::G(perm:presql)}
//  set presql
//}

//#-------------------------------------------------------------------------
//#
//proc slave_test_script {script} {

//  # Create the interpreter used to run the test script.
//  interp create tinterp

//  # Populate some global variables that tester.tcl expects to see.
//  foreach {var value} [list              \
//    ::argv0 $::argv0                     \
//    ::argv  {}                           \
//    ::SLAVE 1                            \
//  ] {
//    interp eval tinterp [list set $var $value]
//  }

//  # The alias used to access the global test counters.
//  tinterp alias set_test_counter set_test_counter

//  # Set up the ::cmdlinearg array in the slave.
//  interp eval tinterp [list array set ::cmdlinearg [array get ::cmdlinearg]]

//  # Set up the ::G array in the slave.
//  interp eval tinterp [list array set ::G [array get ::G]]

//  # Load the various test interfaces implemented in C.
//  load_testfixture_extensions tinterp

//  # Run the test script.
//  interp eval tinterp $script

//  # Check if the interpreter call [run_thread_tests]
//  if { [interp eval tinterp {info exists ::run_thread_tests_called}] } {
//    set ::run_thread_tests_called 1
//  }

//  # Delete the interpreter used to run the test script.
//  interp delete tinterp
//}

//proc slave_test_file {zFile} {
//  set tail [file tail $zFile]

//  if {[info exists ::G(start:permutation)]} {
//    if {[permutation] != $::G(start:permutation)} return
//    unset ::G(start:permutation)
//  }
//  if {[info exists ::G(start:file)]} {
//    if {$tail != $::G(start:file) && $tail!="$::G(start:file).test"} return
//    unset ::G(start:file)
//  }

//  # Remember the value of the shared-cache setting. So that it is possible
//  # to check afterwards that it was not modified by the test script.
//  #
//  ifcapable shared_cache { set scs [sqlite3_enable_shared_cache] }

//  # Run the test script in a slave interpreter.
//  #
//  unset -nocomplain ::run_thread_tests_called
//  reset_prng_state
//  set ::sqlite_open_file_count 0
//  set time [time { slave_test_script [list source $zFile] }]
//  set ms [expr [lindex $time 0] / 1000]

//  # Test that all files opened by the test script were closed. Omit this
//  # if the test script has "thread" in its name. The open file counter
//  # is not thread-safe.
//  #
//  if {[info exists ::run_thread_tests_called]==0} {
//    do_test ${tail}-closeallfiles { expr {$::sqlite_open_file_count>0} } {0}
//  }
//  set ::sqlite_open_file_count 0

//  # Test that the global "shared-cache" setting was not altered by 
//  # the test script.
//  #
//  ifcapable shared_cache { 
//    set res [expr {[sqlite3_enable_shared_cache] == $scs}]
//    do_test ${tail}-sharedcachesetting [list set {} $res] 1
//  }

//  # Add some info to the output.
//  #
//  puts "Time: $tail $ms ms"
//  show_memstats
//}

//# Open a new connection on database test.db and execute the SQL script
//# supplied as an argument. Before returning, close the new conection and
//# restore the 4 byte fields starting at header offsets 28, 92 and 96
//# to the values they held before the SQL was executed. This simulates
//# a write by a pre-3.7.0 client.
//#
//proc sql36231 {sql} {
//  set B [hexio_read test.db 92 8]
//  set A [hexio_read test.db 28 4]
//  sqlite3 db36231 test.db
//  catch { db36231 func a_string a_string }
//  execsql $sql db36231
//  db36231 close
//  hexio_write test.db 28 $A
//  hexio_write test.db 92 $B
//  return ""
//}

//proc db_save {} {
//  foreach f [glob -nocomplain sv_test.db*] { forcedelete $f }
//  foreach f [glob -nocomplain test.db*] {
//    set f2 "sv_$f"
//    forcecopy $f $f2
//  }
//}
//proc db_save_and_close {} {
//  db_save
//  catch { db close }
//  return ""
//}
//proc db_restore {} {
//  foreach f [glob -nocomplain test.db*] { forcedelete $f }
//  foreach f2 [glob -nocomplain sv_test.db*] {
//    set f [string range $f2 3 end]
//    forcecopy $f2 $f
//  }
//}
//proc db_restore_and_reopen {{dbfile test.db}} {
//  catch { db close }
//  db_restore
//  sqlite3 db $dbfile
//}
//proc db_delete_and_reopen {{file test.db}} {
//  catch { db close }
//  foreach f [glob -nocomplain test.db*] { forcedelete $f }
//  sqlite3 db $file
//}

#pragma endregion

// This file contains code used by several different test scripts. The code in this file allows testfixture to control another process (or processes) to test locking.
#pragma region lock_common.tcl

//protected void do_multiclient_test(string varname, string script)
//{
//
//	//foreach code [list {
//	//  if {[info exists ::G(valgrind)]} { db close ; continue }
//	//  set ::code2_chan [launch_testfixture]
//	//  set ::code3_chan [launch_testfixture]
//	//  proc code2 {tcl} { testfixture $::code2_chan $tcl }
//	//  proc code3 {tcl} { testfixture $::code3_chan $tcl }
//	//  set tn 1
//	//} {
//	//  proc code2 {tcl} { uplevel #0 $tcl }
//	//  proc code3 {tcl} { uplevel #0 $tcl }
//	//  set tn 2
//	//}] {
//	//  faultsim_delete_and_reopen
//
//	//  proc code1 {tcl} { uplevel #0 $tcl }
//
//	//  # Open connections [db2] and [db3]. Depending on which iteration this
//	//  # is, the connections may be created in this interpreter, or in 
//	//  # interpreters running in other OS processes. As such, the [db2] and [db3]
//	//  # commands should only be accessed within [code2] and [code3] blocks,
//	//  # respectively.
//	//  #
//	//  eval $code
//	//  code2 { sqlite3 db2 test.db }
//	//  code3 { sqlite3 db3 test.db }
//
//
//	//  uplevel set $varname $tn
//	//  uplevel $script
//
//	//  catch { code2 { db2 close } }
//	//  catch { code3 { db3 close } }
//	//  catch { close $::code2_chan }
//	//  catch { close $::code3_chan }
//	//  catch { db close }
//	//}
//}
//
//// Shorthand commands. Execute SQL using database connection [db2] or [db3]. Return the results.
//protected void sql1(string sql)
//{
//	//db eval $sql
//}
//protected void sql2(string sql)
//{
//	//code2 [list db2 eval $sql]
//}
//protected void sql3(string sql)
//{
//	//code3 [list db3 eval $sql]
//}
//
//protected void csql1(string sql)
//{
//	//list [catch { sql1 $sql } msg] $msg
//}
//protected void csql2(string sql)
//{
//	//list [catch { sql2 $sql } msg] $msg
//}
//protected void csql3(string sql)
//{
//	//list [catch { sql3 $sql } msg] $msg
//}
//
//// Launch another testfixture process to be controlled by this one. A channel name is returned that may be passed as the first argument to proc
//// 'testfixture' to execute a command. The child testfixture process is shut down by closing the channel.
//protected void launch_testfixture(string prg = "")
//{
//	//write_main_loop
//	//if {$prg eq ""} { set prg [info nameofexec] }
//	//if {$prg eq ""} { set prg testfixture }
//	//if {[file tail $prg]==$prg} { set prg [file join . $prg] }
//	//set chan [open "|$prg tf_main.tcl" r+]
//	//fconfigure $chan -buffering line
//	//set rc [catch { 
//	//  testfixture $chan "sqlite3_test_control_pending_byte $::sqlite_pending_byte"
//	//}]
//	//if {$rc} {
//	//  testfixture $chan "set ::sqlite_pending_byte $::sqlite_pending_byte"
//	//}
//	//return $chan
//}
//
//// Execute a command in a child testfixture process, connected by two-way channel $chan. Return the result of the command, or an error message.
//protected void testfixture(object chan, object cmd)
//{
//	//puts $chan $cmd
//	//puts $chan OVER
//	//set r ""
//	//while { 1 } {
//	//  set line [gets $chan]
//	//  if { $line == "OVER" } { 
//	//    set res [lindex $r 1]
//	//    if { [lindex $r 0] } { error $res }
//	//    return $res
//	//  }
//	//  if {[eof $chan]} {
//	//    return "ERROR: Child process hung up"
//	//  }
//	//  append r $line
//	//}
//}
//
//protected void testfixture_nb_cb(string varname, string chan)
//{
//	//if {[eof $chan]} {
//	//  append ::tfnb($chan) "ERROR: Child process hung up"
//	//  set line "OVER"
//	//} else {
//	//  set line [gets $chan]
//	//}
//
//	//if { $line == "OVER" } {
//	//  set $varname [lindex $::tfnb($chan) 1]
//	//  unset ::tfnb($chan)
//	//  close $chan
//	//} else {
//	//  append ::tfnb($chan) $line
//	//}
//}
//
//protected void testfixture_nb(string varname, string cmd)
//{
//	//set chan [launch_testfixture]
//	//set ::tfnb($chan) ""
//	//fconfigure $chan -blocking 0 -buffering none
//	//puts $chan $cmd
//	//puts $chan OVER
//	//fileevent $chan readable [list testfixture_nb_cb $varname $chan]
//	//return ""
//}
//
//// Write the main loop for the child testfixture processes into file tf_main.tcl. The parent (this script) interacts with the child processes
//// via a two way pipe. The parent writes a script to the stdin of the child process, followed by the word "OVER" on a line of its own. The child
//// process evaluates the script and writes the results to stdout, followed by an "OVER" of its own.
////set main_loop_written 0
//protected void write_main_loop()
//{
//	//if {$::main_loop_written} return
//	//set wrapper ""
//	//if {[sqlite3 -has-codec] && [info exists ::do_not_use_codec]==0} {
//	//  set wrapper "
//	//    rename sqlite3 sqlite_orig
//	//    proc sqlite3 {args} {[info body sqlite3]}
//	//  "
//	//}
//
//	//set fd [open tf_main.tcl w]
//	//puts $fd [string map [list %WRAPPER% $wrapper] {
//	//  %WRAPPER%
//	//  set script ""
//	//  while {![eof stdin]} {
//	//    flush stdout
//	//    set line [gets stdin]
//	//    if { $line == "OVER" } {
//	//      set rc [catch {eval $script} result]
//	//      puts [list $rc $result]
//	//      puts OVER
//	//      flush stdout
//	//      set script ""
//	//    } else {
//	//      append script $line
//	//      append script "\n"
//	//    }
//	//  }
//	//}]
//	//close $fd
//	//set main_loop_written 1
//}

#pragma endregion