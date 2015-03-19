# 2010 June 15
#
# The author disclaims copyright to this source code.  In place of
# a legal notice, here is a blessing:
#
#    May you do good and not evil.
#    May you find forgiveness for yourself and forgive others.
#    May you share freely, never taking more than you give.
#
#***********************************************************************
#

set testdir [file dirname $argv0]
source $testdir/tester.tcl
source $testdir/lock_common.tcl
source $testdir/malloc_common.tcl

if {[permutation] == "inmemory_journal"} {
  finish_test
  return
}

if {$::tcl_platform(platform)=="windows"} {
  finish_test
  return
}

set a_string_counter 1
proc a_string {n} {
  global a_string_counter
  incr a_string_counter
  string range [string repeat "${a_string_counter}." $n] 1 $n
}
db func a_string a_string

#-------------------------------------------------------------------------
# Test fault-injection while rolling back a hot-journal file.
#
do_test pagerfault-1-pre1 {
  execsql {
    PRAGMA journal_mode = DELETE;
    PRAGMA cache_size = 10;
    CREATE TABLE t1(a UNIQUE, b UNIQUE);
    INSERT INTO t1 VALUES(a_string(200), a_string(300));
    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
    BEGIN;
      INSERT INTO t1 SELECT a_string(201), a_string(301) FROM t1;
      INSERT INTO t1 SELECT a_string(202), a_string(302) FROM t1;
      INSERT INTO t1 SELECT a_string(203), a_string(303) FROM t1;
      INSERT INTO t1 SELECT a_string(204), a_string(304) FROM t1;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-1 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { SELECT count(*) FROM t1 }
} -test {
  faultsim_test_result {0 4} 
  faultsim_integrity_check
  if {[db one { SELECT count(*) FROM t1 }] != 4} {
    error "Database content appears incorrect"
  }
}

#-------------------------------------------------------------------------
# Test fault-injection while rolling back a hot-journal file with a 
# page-size different from the current value stored on page 1 of the
# database file.
#
do_test pagerfault-2-pre1 {
  testvfs tv -default 1
  tv filter xSync
  tv script xSyncCb
  proc xSyncCb {filename args} {
    if {[string match *journal filename]==0} faultsim_save
  }
  faultsim_delete_and_reopen
  execsql {
    PRAGMA page_size = 4096;
    BEGIN;
      CREATE TABLE abc(a, b, c);
      INSERT INTO abc VALUES('o', 't', 't'); 
      INSERT INTO abc VALUES('f', 'f', 's'); 
      INSERT INTO abc SELECT * FROM abc; -- 4
      INSERT INTO abc SELECT * FROM abc; -- 8
      INSERT INTO abc SELECT * FROM abc; -- 16
      INSERT INTO abc SELECT * FROM abc; -- 32
      INSERT INTO abc SELECT * FROM abc; -- 64
      INSERT INTO abc SELECT * FROM abc; -- 128
      INSERT INTO abc SELECT * FROM abc; -- 256
    COMMIT;
    PRAGMA page_size = 1024;
    VACUUM;
  }
  db close
  tv delete
} {}
do_faultsim_test pagerfault-2 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { SELECT * FROM abc }
} -test {
  set answer [split [string repeat "ottffs" 128] ""]
  faultsim_test_result [list 0 $answer]
  faultsim_integrity_check
  set res [db eval { SELECT * FROM abc }]
  if {$res != $answer} { error "Database content appears incorrect ($res)" }
} 

#-------------------------------------------------------------------------
# Test fault-injection while rolling back hot-journals that were created
# as part of a multi-file transaction.
#
do_test pagerfault-3-pre1 {
  testvfs tstvfs -default 1
  tstvfs filter xDelete
  tstvfs script xDeleteCallback

  proc xDeleteCallback {method file args} {
    set file [file tail $file]
    if { [string match *mj* $file] } { faultsim_save }
  }

  faultsim_delete_and_reopen
  db func a_string a_string

  execsql {
    ATTACH 'test.db2' AS aux;
    PRAGMA journal_mode = DELETE;
    PRAGMA main.cache_size = 10;
    PRAGMA aux.cache_size = 10;

    CREATE TABLE t1(a UNIQUE, b UNIQUE);
    CREATE TABLE aux.t2(a UNIQUE, b UNIQUE);
    INSERT INTO t1 VALUES(a_string(200), a_string(300));
    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
    INSERT INTO t2 SELECT * FROM t1;

    BEGIN;
      INSERT INTO t1 SELECT a_string(201), a_string(301) FROM t1;
      INSERT INTO t1 SELECT a_string(202), a_string(302) FROM t1;
      INSERT INTO t1 SELECT a_string(203), a_string(303) FROM t1;
      INSERT INTO t1 SELECT a_string(204), a_string(304) FROM t1;
      REPLACE INTO t2 SELECT * FROM t1;
    COMMIT;
  }

  db close
  tstvfs delete
} {}
do_faultsim_test pagerfault-3 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { 
    ATTACH 'test.db2' AS aux;
    SELECT count(*) FROM t2;
    SELECT count(*) FROM t1;
  }
} -test {
  faultsim_test_result {0 {4 4}} {1 {unable to open database: test.db2}}
  faultsim_integrity_check
  catchsql { ATTACH 'test.db2' AS aux }
  if {[db one { SELECT count(*) FROM t1 }] != 4
   || [db one { SELECT count(*) FROM t2 }] != 4
  } {
    error "Database content appears incorrect"
  }
}

#-------------------------------------------------------------------------
# Test fault-injection as part of a vanilla, no-transaction, INSERT
# statement.
#
do_faultsim_test pagerfault-4 -prep {
  faultsim_delete_and_reopen
} -body {
  execsql { 
    CREATE TABLE x(y);
    INSERT INTO x VALUES('z');
    SELECT * FROM x;
  }
} -test {
  faultsim_test_result {0 z}
  faultsim_integrity_check
}

#-------------------------------------------------------------------------
# Test fault-injection as part of a commit when using journal_mode=PERSIST.
# Three different cases:
#
#    pagerfault-5.1: With no journal_size_limit configured.
#    pagerfault-5.2: With a journal_size_limit configured.
#    pagerfault-5.4: Multi-file transaction. One connection has a 
#                    journal_size_limit of 0, the other has no limit.
#
do_test pagerfault-5-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string
  execsql {
    CREATE TABLE t1(a UNIQUE, b UNIQUE);
    INSERT INTO t1 VALUES(a_string(200), a_string(300));
    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-5.1 -prep {
  faultsim_restore_and_reopen
  db func a_string a_string
  execsql { PRAGMA journal_mode = PERSIST }
} -body {
  execsql { INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1 }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}
do_faultsim_test pagerfault-5.2 -prep {
  faultsim_restore_and_reopen
  db func a_string a_string
  execsql { 
    PRAGMA journal_mode = PERSIST;
    PRAGMA journal_size_limit = 2048;
  }
} -body {
  execsql { INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1 }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}
do_faultsim_test pagerfault-5.3 -faults oom-transient -prep {
  faultsim_restore_and_reopen
  db func a_string a_string
  forcedelete test2.db test2.db-journal test2.db-wal
  execsql { 
    PRAGMA journal_mode = PERSIST;
    ATTACH 'test2.db' AS aux;
    PRAGMA aux.journal_mode = PERSIST;
    PRAGMA aux.journal_size_limit = 0;
  }
} -body {
  execsql {
    BEGIN;
      INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
      CREATE TABLE aux.t2 AS SELECT * FROM t1;
    COMMIT;
  }
} -test {
  faultsim_test_result {0 {}}

  catchsql { COMMIT }
  catchsql { ROLLBACK }

  faultsim_integrity_check
  set res ""
  set rc [catch { set res [db one { PRAGMA aux.integrity_check }] }]
  if {$rc!=0 || $res != "ok"} {error "integrity-check problem:$rc $res"}
}

#-------------------------------------------------------------------------
# Test fault-injection as part of a commit when using 
# journal_mode=TRUNCATE.
#
do_test pagerfault-6-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string
  execsql {
    CREATE TABLE t1(a UNIQUE, b UNIQUE);
    INSERT INTO t1 VALUES(a_string(200), a_string(300));
  }
  faultsim_save_and_close
} {}

do_faultsim_test pagerfault-6.1 -prep {
  faultsim_restore_and_reopen
  db func a_string a_string
  execsql { PRAGMA journal_mode = TRUNCATE }
} -body {
  execsql { INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1 }
  execsql { INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1 }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}

# The unix vfs xAccess() method considers a file zero bytes in size to
# "not exist". This proc overrides that behaviour so that a zero length
# file is considered to exist.
#
proc xAccess {method filename op args} {
  if {$op != "SQLITE_ACCESS_EXISTS"} { return "" }
  return [file exists $filename]
}
do_faultsim_test pagerfault-6.2 -faults cantopen-* -prep {
  shmfault filter xAccess
  shmfault script xAccess

  faultsim_restore_and_reopen
  db func a_string a_string
  execsql { PRAGMA journal_mode = TRUNCATE }
} -body {
  execsql { INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1 }
  execsql { INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1 }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}

# The following was an attempt to get a bitvec malloc to fail. Didn't work.
#
# do_test pagerfault-6-pre1 {
#   faultsim_delete_and_reopen
#   execsql {
#     CREATE TABLE t1(x, y, UNIQUE(x, y));
#     INSERT INTO t1 VALUES(1, randomblob(1501));
#     INSERT INTO t1 VALUES(2, randomblob(1502));
#     INSERT INTO t1 VALUES(3, randomblob(1503));
#     INSERT INTO t1 VALUES(4, randomblob(1504));
#     INSERT INTO t1 
#       SELECT x, randomblob(1500+oid+(SELECT max(oid) FROM t1)) FROM t1;
#     INSERT INTO t1 
#       SELECT x, randomblob(1500+oid+(SELECT max(oid) FROM t1)) FROM t1;
#     INSERT INTO t1 
#       SELECT x, randomblob(1500+oid+(SELECT max(oid) FROM t1)) FROM t1;
#     INSERT INTO t1 
#       SELECT x, randomblob(1500+oid+(SELECT max(oid) FROM t1)) FROM t1;
#   }
#   faultsim_save_and_close
# } {}
# do_faultsim_test pagerfault-6 -prep {
#   faultsim_restore_and_reopen
# } -body {
#   execsql { 
#     BEGIN;
#       UPDATE t1 SET x=x+4 WHERE x=1;
#       SAVEPOINT one;
#         UPDATE t1 SET x=x+4 WHERE x=2;
#         SAVEPOINT three;
#           UPDATE t1 SET x=x+4 WHERE x=3;
#           SAVEPOINT four;
#             UPDATE t1 SET x=x+4 WHERE x=4;
#         RELEASE three;
#     COMMIT;
#     SELECT DISTINCT x FROM t1;
#   }
# } -test {
#   faultsim_test_result {0 {5 6 7 8}}
#   faultsim_integrity_check
# }
#

# This is designed to provoke a special case in the pager code:
#
# If an error (specifically, a FULL or IOERR error) occurs while writing a
# dirty page to the file-system in order to free up memory, the pager enters
# the "error state". An IO error causes SQLite to roll back the current
# transaction (exiting the error state). A FULL error, however, may only
# rollback the current statement.
#
# This block tests that nothing goes wrong if a FULL error occurs while
# writing a dirty page out to free memory from within a statement that has
# opened a statement transaction.
#
do_test pagerfault-7-pre1 {
  faultsim_delete_and_reopen
  execsql {
    CREATE TABLE t2(a INTEGER PRIMARY KEY, b);
    BEGIN;
      INSERT INTO t2 VALUES(NULL, randomblob(1500));
      INSERT INTO t2 VALUES(NULL, randomblob(1500));
      INSERT INTO t2 SELECT NULL, randomblob(1500) FROM t2;    --  4
      INSERT INTO t2 SELECT NULL, randomblob(1500) FROM t2;    --  8
      INSERT INTO t2 SELECT NULL, randomblob(1500) FROM t2;    -- 16
      INSERT INTO t2 SELECT NULL, randomblob(1500) FROM t2;    -- 32
      INSERT INTO t2 SELECT NULL, randomblob(1500) FROM t2;    -- 64
    COMMIT;
    CREATE TABLE t1(a PRIMARY KEY, b);
    INSERT INTO t1 SELECT * FROM t2;
    DROP TABLE t2;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-7 -prep {
  faultsim_restore_and_reopen
  execsql { 
    PRAGMA cache_size = 10;
    BEGIN;
      UPDATE t1 SET b = randomblob(1500);
  }
} -body {
  execsql { UPDATE t1 SET a = 65, b = randomblob(1500) WHERE (a+1)>200 }
  execsql COMMIT
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}

do_test pagerfault-8-pre1 {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA auto_vacuum = 1;
    CREATE TABLE t1(a INTEGER PRIMARY KEY, b);
    BEGIN;
      INSERT INTO t1 VALUES(NULL, randomblob(1500));
      INSERT INTO t1 VALUES(NULL, randomblob(1500));
      INSERT INTO t1 SELECT NULL, randomblob(1500) FROM t1;    --  4
      INSERT INTO t1 SELECT NULL, randomblob(1500) FROM t1;    --  8
      INSERT INTO t1 SELECT NULL, randomblob(1500) FROM t1;    -- 16
      INSERT INTO t1 SELECT NULL, randomblob(1500) FROM t1;    -- 32
      INSERT INTO t1 SELECT NULL, randomblob(1500) FROM t1;    -- 64
    COMMIT;
  }
  faultsim_save_and_close
  set filesize [file size test.db]
  set {} {}
} {}
do_test pagerfault-8-pre2 {
  faultsim_restore_and_reopen
  execsql { DELETE FROM t1 WHERE a>32 }
  expr {[file size test.db] < $filesize}
} {1}
do_faultsim_test pagerfault-8 -prep {
  faultsim_restore_and_reopen
  execsql { 
    BEGIN;
    DELETE FROM t1 WHERE a>32;
  }
} -body {
  execsql COMMIT
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}

#-------------------------------------------------------------------------
# This test case is specially designed so that during a savepoint 
# rollback, a new cache entry must be allocated (see comments surrounding
# the call to sqlite3PagerAcquire() from within pager_playback_one_page()
# for details). Test the effects of injecting an OOM at this point.
#
do_test pagerfault-9-pre1 {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA auto_vacuum = incremental;
    CREATE TABLE t1(x);
    CREATE TABLE t2(y);
    CREATE TABLE t3(z);

    INSERT INTO t1 VALUES(randomblob(900));
    INSERT INTO t1 VALUES(randomblob(900));
    DELETE FROM t1;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-9.1 -prep {
  faultsim_restore_and_reopen
  execsql { 
    BEGIN;
      INSERT INTO t1 VALUES(randomblob(900));
      INSERT INTO t1 VALUES(randomblob(900));
      DROP TABLE t3;
      DROP TABLE t2;
      SAVEPOINT abc;
        PRAGMA incremental_vacuum;
  }
} -body {
  execsql {
    ROLLBACK TO abc;
    COMMIT;
    PRAGMA freelist_count
  }
} -test {
  faultsim_test_result {0 2}
  faultsim_integrity_check

  set sl [db one { SELECT COALESCE(sum(length(x)), 'null') FROM t1 }]
  if {$sl!="null" && $sl!=1800} { 
    error "Content looks no good... ($sl)" 
  }
}

#-------------------------------------------------------------------------
# Test fault injection with a temporary database file.
#
foreach v {a b} {
  do_faultsim_test pagerfault-10$v -prep {
    sqlite3 db ""
    db func a_string a_string;
    execsql {
      PRAGMA cache_size = 10;
      BEGIN;
        CREATE TABLE xx(a, b, UNIQUE(a, b));
        INSERT INTO xx VALUES(a_string(200), a_string(200));
        INSERT INTO xx SELECT a_string(200), a_string(200) FROM xx;
        INSERT INTO xx SELECT a_string(200), a_string(200) FROM xx;
        INSERT INTO xx SELECT a_string(200), a_string(200) FROM xx;
        INSERT INTO xx SELECT a_string(200), a_string(200) FROM xx;
      COMMIT;
    }
  } -body {
    execsql { UPDATE xx SET a = a_string(300) }
  } -test {
    faultsim_test_result {0 {}}
    if {$::v == "b"} { execsql { PRAGMA journal_mode = TRUNCATE } }
    faultsim_integrity_check
    faultsim_integrity_check
  }
}

#-------------------------------------------------------------------------
# Test fault injection with transaction savepoints (savepoints created
# when a SAVEPOINT command is executed outside of any other savepoint
# or transaction context).
#
do_test pagerfault-9-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string;
  execsql {
    PRAGMA auto_vacuum = on;
    CREATE TABLE t1(x UNIQUE);
    CREATE TABLE t2(y UNIQUE);
    CREATE TABLE t3(z UNIQUE);
    BEGIN;
      INSERT INTO t1 VALUES(a_string(202));
      INSERT INTO t2 VALUES(a_string(203));
      INSERT INTO t3 VALUES(a_string(204));
      INSERT INTO t1 SELECT a_string(202) FROM t1;
      INSERT INTO t1 SELECT a_string(203) FROM t1;
      INSERT INTO t1 SELECT a_string(204) FROM t1;
      INSERT INTO t1 SELECT a_string(205) FROM t1;
      INSERT INTO t2 SELECT a_string(length(x)) FROM t1;
      INSERT INTO t3 SELECT a_string(length(x)) FROM t1;
    COMMIT;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-11 -prep {
  faultsim_restore_and_reopen
  execsql { PRAGMA cache_size = 10 }
} -body {
  execsql {
    SAVEPOINT trans;
      UPDATE t2 SET y = y||'2';
      INSERT INTO t3 SELECT * FROM t2;
      DELETE FROM t1;
    ROLLBACK TO trans;
    UPDATE t1 SET x = x||'3';
    INSERT INTO t2 SELECT * FROM t1;
    DELETE FROM t3;
    RELEASE trans;
  }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}


#-------------------------------------------------------------------------
# Test fault injection when writing to a database file that resides on
# a file-system with a sector-size larger than the database page-size.
#
do_test pagerfault-12-pre1 {
  testvfs ss_layer -default 1
  ss_layer sectorsize 4096
  faultsim_delete_and_reopen
  db func a_string a_string;

  execsql {
    PRAGMA page_size = 1024;
    PRAGMA journal_mode = PERSIST;
    PRAGMA cache_size = 10;
    BEGIN;
      CREATE TABLE t1(x, y UNIQUE);
      INSERT INTO t1 VALUES(a_string(333), a_string(444));
      INSERT INTO t1 SELECT a_string(333+rowid), a_string(444+rowid) FROM t1;
      INSERT INTO t1 SELECT a_string(333+rowid), a_string(444+rowid) FROM t1;
      INSERT INTO t1 SELECT a_string(333+rowid), a_string(444+rowid) FROM t1;
      INSERT INTO t1 SELECT a_string(333+rowid), a_string(444+rowid) FROM t1;
      INSERT INTO t1 SELECT a_string(44), a_string(55) FROM t1 LIMIT 13;
    COMMIT;
  }
  faultsim_save_and_close
} {}

do_faultsim_test pagerfault-12a -prep {
  faultsim_restore_and_reopen
  execsql { PRAGMA cache_size = 10 }
  db func a_string a_string;
} -body {
  execsql {
    UPDATE t1 SET x = a_string(length(x)), y = a_string(length(y));
  }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}

do_test pagerfault-12-pre2 {
  faultsim_restore_and_reopen
  execsql {
    CREATE TABLE t2 AS SELECT * FROM t1 LIMIT 10;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-12b -prep {
  faultsim_restore_and_reopen
  db func a_string a_string;
  execsql { SELECT * FROM t1 }
} -body {
  set sql(1) { UPDATE t2 SET x = a_string(280) }
  set sql(2) { UPDATE t1 SET x = a_string(280) WHERE rowid = 5 }

  db eval { SELECT rowid FROM t1 LIMIT 2 } { db eval $sql($rowid) }

} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}

catch { db close }
ss_layer delete


#-------------------------------------------------------------------------
# Test fault injection when SQLite opens a database where the size of the
# database file is zero bytes but the accompanying journal file is larger
# than that. In this scenario SQLite should delete the journal file 
# without rolling it back, even if it is in all other respects a valid
# hot-journal file.
#
do_test pagerfault-13-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string;
  execsql {
    PRAGMA journal_mode = PERSIST;
    BEGIN;
      CREATE TABLE t1(x, y UNIQUE);
      INSERT INTO t1 VALUES(a_string(333), a_string(444));
    COMMIT;
  }
  db close
  forcedelete test.db
  faultsim_save
} {}
do_faultsim_test pagerfault-13 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { CREATE TABLE xx(a, b) }
} -test {
  faultsim_test_result {0 {}}
}

#---------------------------------------------------------------------------
# Test fault injection into a small backup operation.
#
do_test pagerfault-14-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string;
  execsql {
    PRAGMA journal_mode = PERSIST;
    ATTACH 'test.db2' AS two;
    BEGIN;
      CREATE TABLE t1(x, y UNIQUE);
      CREATE TABLE two.t2(x, y UNIQUE);
      INSERT INTO t1 VALUES(a_string(333), a_string(444));
      INSERT INTO t2 VALUES(a_string(333), a_string(444));
    COMMIT;
  }
  faultsim_save_and_close
} {}

do_faultsim_test pagerfault-14a -prep {
  faultsim_restore_and_reopen
} -body {
  if {[catch {db backup test.db2} msg]} { error [regsub {.*: } $msg {}] }
} -test {
  faultsim_test_result {0 {}} {1 {}} {1 {SQL logic error or missing database}}
}

# If TEMP_STORE is 2 or greater, then the database [db2] will be created
# as an in-memory database. This test will not work in that case, as it
# is not possible to change the page-size of an in-memory database. Even
# using the backup API.
#
if {$TEMP_STORE<2} {
  do_faultsim_test pagerfault-14b -prep {
    catch { db2 close }
    faultsim_restore_and_reopen
    sqlite3 db2 ""
    db2 eval { PRAGMA page_size = 4096; CREATE TABLE xx(a) }
  } -body {
    sqlite3_backup B db2 main db main
    B step 200
    set rc [B finish]
    if {[string match SQLITE_IOERR_* $rc]} {set rc SQLITE_IOERR}
    if {$rc != "SQLITE_OK"} { error [sqlite3_test_errstr $rc] }
    set {} {}
  } -test {
    faultsim_test_result {0 {}} {1 {sqlite3_backup_init() failed}}
  }
}

do_faultsim_test pagerfault-14c -prep {
  catch { db2 close }
  faultsim_restore_and_reopen
  sqlite3 db2 test.db2
  db2 eval { 
    PRAGMA synchronous = off; 
    PRAGMA page_size = 4096; 
    CREATE TABLE xx(a);
  }
} -body {
  sqlite3_backup B db2 main db main
  B step 200
  set rc [B finish]
  if {[string match SQLITE_IOERR_* $rc]} {set rc SQLITE_IOERR}
  if {$rc != "SQLITE_OK"} { error [sqlite3_test_errstr $rc] }
  set {} {}
} -test {
  faultsim_test_result {0 {}} {1 {sqlite3_backup_init() failed}}
}

do_test pagerfault-15-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string;
  execsql {
    BEGIN;
      CREATE TABLE t1(x, y UNIQUE);
      INSERT INTO t1 VALUES(a_string(11), a_string(22));
      INSERT INTO t1 VALUES(a_string(11), a_string(22));
    COMMIT;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-15 -prep {
  faultsim_restore_and_reopen
  db func a_string a_string;
} -body {
  db eval { SELECT * FROM t1 LIMIT 1 } {
    execsql {
      BEGIN; INSERT INTO t1 VALUES(a_string(333), a_string(555)); COMMIT;
      BEGIN; INSERT INTO t1 VALUES(a_string(333), a_string(555)); COMMIT;
    }
  }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}


do_test pagerfault-16-pre1 {
  faultsim_delete_and_reopen
  execsql { CREATE TABLE t1(x, y UNIQUE) }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-16 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql {
    PRAGMA locking_mode = exclusive;
    PRAGMA journal_mode = wal;
    INSERT INTO t1 VALUES(1, 2);
    INSERT INTO t1 VALUES(3, 4);
    PRAGMA journal_mode = delete;
    INSERT INTO t1 VALUES(4, 5);
    PRAGMA journal_mode = wal;
    INSERT INTO t1 VALUES(6, 7);
    PRAGMA journal_mode = persist;
    INSERT INTO t1 VALUES(8, 9);
  }
} -test {
  faultsim_test_result {0 {exclusive wal delete wal persist}}
  faultsim_integrity_check
}


#-------------------------------------------------------------------------
# Test fault injection while changing into and out of WAL mode.
#
do_test pagerfault-17-pre1 {
  faultsim_delete_and_reopen
  execsql {
    CREATE TABLE t1(a PRIMARY KEY, b);
    INSERT INTO t1 VALUES(1862, 'Botha');
    INSERT INTO t1 VALUES(1870, 'Smuts');
    INSERT INTO t1 VALUES(1866, 'Hertzog');
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-17a -prep {
  faultsim_restore_and_reopen
} -body {
  execsql {
    PRAGMA journal_mode = wal;
    PRAGMA journal_mode = delete;
  }
} -test {
  faultsim_test_result {0 {wal delete}}
  faultsim_integrity_check
}
do_faultsim_test pagerfault-17b -prep {
  faultsim_restore_and_reopen
  execsql { PRAGMA synchronous = OFF }
} -body {
  execsql {
    PRAGMA journal_mode = wal;
    INSERT INTO t1 VALUES(22, 'Clarke');
    PRAGMA journal_mode = delete;
  }
} -test {
  faultsim_test_result {0 {wal delete}}
  faultsim_integrity_check
}
do_faultsim_test pagerfault-17c -prep {
  faultsim_restore_and_reopen
  execsql { 
    PRAGMA locking_mode = exclusive;
    PRAGMA journal_mode = wal;
  }
} -body {
  execsql { PRAGMA journal_mode = delete }
} -test {
  faultsim_test_result {0 delete}
  faultsim_integrity_check
}
do_faultsim_test pagerfault-17d -prep {
  catch { db2 close }
  faultsim_restore_and_reopen
  sqlite3 db2 test.db
  execsql { PRAGMA journal_mode = delete }
  execsql { PRAGMA journal_mode = wal }
  execsql { INSERT INTO t1 VALUES(99, 'Bradman') } db2
} -body {
  execsql { PRAGMA journal_mode = delete }
} -test {
  faultsim_test_result {1 {database is locked}}
  faultsim_integrity_check
}
do_faultsim_test pagerfault-17e -prep {
  catch { db2 close }
  faultsim_restore_and_reopen
  sqlite3 db2 test.db
  execsql { PRAGMA journal_mode = delete }
  execsql { PRAGMA journal_mode = wal }
  set ::chan [launch_testfixture]
  testfixture $::chan {
    sqlite3 db test.db
    db eval { INSERT INTO t1 VALUES(101, 'Latham') }
  }
  catch { testfixture $::chan sqlite_abort }
  catch { close $::chan }
} -body {
  execsql { PRAGMA journal_mode = delete }
} -test {
  faultsim_test_result {0 delete}
  faultsim_integrity_check
}

#-------------------------------------------------------------------------
# Test fault-injection when changing from journal_mode=persist to 
# journal_mode=delete (this involves deleting the journal file).
#
do_test pagerfault-18-pre1 {
  faultsim_delete_and_reopen
  execsql {
    CREATE TABLE qq(x);
    INSERT INTO qq VALUES('Herbert');
    INSERT INTO qq VALUES('Macalister');
    INSERT INTO qq VALUES('Mackenzie');
    INSERT INTO qq VALUES('Lilley');
    INSERT INTO qq VALUES('Palmer');
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-18 -prep {
  faultsim_restore_and_reopen
  execsql {
    PRAGMA journal_mode = PERSIST;
    INSERT INTO qq VALUES('Beatty');
  }
} -body {
  execsql { PRAGMA journal_mode = delete }
} -test {
  faultsim_test_result {0 delete}
  faultsim_integrity_check
}

do_faultsim_test pagerfault-19a -prep {
  sqlite3 db :memory:
  db func a_string a_string
  execsql {
    PRAGMA auto_vacuum = FULL;
    BEGIN;
      CREATE TABLE t1(a, b);
      INSERT INTO t1 VALUES(a_string(5000), a_string(6000));
    COMMIT;
  }
} -body {
  execsql { 
    CREATE TABLE t2(a, b);
    INSERT INTO t2 SELECT * FROM t1; 
    DELETE FROM t1;
  }
} -test {
  faultsim_test_result {0 {}}
}

do_test pagerfault-19-pre1 {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA auto_vacuum = FULL;
    CREATE TABLE t1(x); INSERT INTO t1 VALUES(1);
    CREATE TABLE t2(x); INSERT INTO t2 VALUES(2);
    CREATE TABLE t3(x); INSERT INTO t3 VALUES(3);
    CREATE TABLE t4(x); INSERT INTO t4 VALUES(4);
    CREATE TABLE t5(x); INSERT INTO t5 VALUES(5);
    CREATE TABLE t6(x); INSERT INTO t6 VALUES(6);
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-19b -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { 
    BEGIN;
      UPDATE t4 SET x = x+1;
      UPDATE t6 SET x = x+1;
      SAVEPOINT one;
        UPDATE t3 SET x = x+1;
        SAVEPOINT two;
          DROP TABLE t2;
      ROLLBACK TO one;
    COMMIT;
    SELECT * FROM t3;
    SELECT * FROM t4;
    SELECT * FROM t6;
  }
} -test {
  faultsim_test_result {0 {3 5 7}}
}

#-------------------------------------------------------------------------
# This tests fault-injection in a special case in the auto-vacuum code.
#
do_test pagerfault-20-pre1 {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA cache_size = 10;
    PRAGMA auto_vacuum = FULL;
    CREATE TABLE t0(a, b);
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-20 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { 
    BEGIN;
      CREATE TABLE t1(a, b);
      CREATE TABLE t2(a, b);
      DROP TABLE t1;
    COMMIT;
  }
} -test {
  faultsim_test_result {0 {}}
}

do_test pagerfault-21-pre1 {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA cache_size = 10;
    CREATE TABLE t0(a PRIMARY KEY, b);
    INSERT INTO t0 VALUES(1, 2);
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-21 -prep {
  faultsim_restore_and_reopen
} -body {
  db eval { SELECT * FROM t0 LIMIT 1 } {
    db eval { INSERT INTO t0 SELECT a+1, b FROM t0 }
    db eval { INSERT INTO t0 SELECT a+2, b FROM t0 }
  }
} -test {
  faultsim_test_result {0 {}}
}


#-------------------------------------------------------------------------
# Test fault-injection and rollback when the nReserve header value 
# is non-zero.
#
do_test pagerfault-21-pre1 {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA page_size = 1024;
    PRAGMA journal_mode = WAL;
    PRAGMA journal_mode = DELETE;
  }
  db close
  hexio_write test.db 20    10
  hexio_write test.db 105 03F0
  sqlite3 db test.db
  db func a_string a_string
  execsql {
    CREATE TABLE t0(a PRIMARY KEY, b UNIQUE);
    INSERT INTO t0 VALUES(a_string(222), a_string(333));
    INSERT INTO t0 VALUES(a_string(223), a_string(334));
    INSERT INTO t0 VALUES(a_string(224), a_string(335));
    INSERT INTO t0 VALUES(a_string(225), a_string(336));
  }
  faultsim_save_and_close
} {}

do_faultsim_test pagerfault-21 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { INSERT INTO t0 SELECT a||'x', b||'x' FROM t0 }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}
ifcapable crashtest {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA page_size = 1024;
    PRAGMA journal_mode = WAL;
    PRAGMA journal_mode = DELETE;
  }
  db close
  hexio_write test.db 20    10
  hexio_write test.db 105 03F0

  sqlite3 db test.db
  db func a_string a_string
  execsql {
    CREATE TABLE t0(a PRIMARY KEY, b UNIQUE);
    INSERT INTO t0 VALUES(a_string(222), a_string(333));
    INSERT INTO t0 VALUES(a_string(223), a_string(334));
  }
  faultsim_save_and_close

  for {set iTest 1} {$iTest<50} {incr iTest} {
    do_test pagerfault-21.crash.$iTest.1 {
      crashsql -delay 1 -file test.db -seed $iTest {
        BEGIN;
          CREATE TABLE t1(a PRIMARY KEY, b UNIQUE);
          INSERT INTO t1 SELECT a, b FROM t0;
        COMMIT;
      }
    } {1 {child process exited abnormally}}
    do_test pagerfault-22.$iTest.2 {
      sqlite3 db test.db
      execsql { PRAGMA integrity_check }
    } {ok}
    db close
  }
}


#-------------------------------------------------------------------------
# When a 3.7.0 client opens a write-transaction on a database file that
# has been appended to or truncated by a pre-370 client, it updates
# the db-size in the file header immediately. This test case provokes
# errors during that operation.
#
do_test pagerfault-22-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string
  execsql {
    PRAGMA page_size = 1024;
    PRAGMA auto_vacuum = 0;
    CREATE TABLE t1(a);
    CREATE INDEX i1 ON t1(a);
    INSERT INTO t1 VALUES(a_string(3000));
    CREATE TABLE t2(a);
    INSERT INTO t2 VALUES(1);
  }
  db close
  sql36231 { INSERT INTO t1 VALUES(a_string(3000)) }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-22 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { INSERT INTO t2 VALUES(2) }
  execsql { SELECT * FROM t2 }
} -test {
  faultsim_test_result {0 {1 2}}
  faultsim_integrity_check
}

#-------------------------------------------------------------------------
# Provoke an OOM error during a commit of multi-file transaction. One of
# the databases written during the transaction is an in-memory database.
# This test causes rollback of the in-memory database after CommitPhaseOne()
# has successfully returned. i.e. the series of calls for the aborted commit 
# is:
#
#   PagerCommitPhaseOne(<in-memory-db>)   ->   SQLITE_OK
#   PagerCommitPhaseOne(<file-db>)        ->   SQLITE_IOERR
#   PagerRollback(<in-memory-db>)
#   PagerRollback(<file-db>)
#
do_faultsim_test pagerfault-23 -prep {
  sqlite3 db :memory:
  foreach f [glob -nocomplain test.db*] { forcedelete $f }
  db eval { 
    ATTACH 'test.db2' AS aux;
    CREATE TABLE t1(a, b);
    CREATE TABLE aux.t2(a, b);
  }
} -body {
  execsql { 
    BEGIN;
      INSERT INTO t1 VALUES(1,2);
      INSERT INTO t2 VALUES(3,4); 
    COMMIT;
  }
} -test {
  faultsim_test_result {0 {}}
  faultsim_integrity_check
}

do_faultsim_test pagerfault-24 -prep {
  faultsim_delete_and_reopen
  db eval { PRAGMA temp_store = file }
  execsql { CREATE TABLE x(a, b) }
} -body {
  execsql { CREATE TEMP TABLE t1(a, b) }
} -test {
  faultsim_test_result {0 {}} \
    {1 {unable to open a temporary database file for storing temporary tables}}
  set ic [db eval { PRAGMA temp.integrity_check }]
  if {$ic != "ok"} { error "Integrity check: $ic" }
}

proc lockrows {n} {
  if {$n==0} { return "" }
  db eval { SELECT * FROM t1 WHERE oid = $n } { 
    return [lockrows [expr {$n-1}]]
  }
}


do_test pagerfault-25-pre1 {
  faultsim_delete_and_reopen
  db func a_string a_string
  execsql {
    PRAGMA page_size = 1024;
    PRAGMA auto_vacuum = 0;
    CREATE TABLE t1(a);
    INSERT INTO t1 VALUES(a_string(500));
    INSERT INTO t1 SELECT a_string(500) FROM t1;
    INSERT INTO t1 SELECT a_string(500) FROM t1;
    INSERT INTO t1 SELECT a_string(500) FROM t1;
    INSERT INTO t1 SELECT a_string(500) FROM t1;
    INSERT INTO t1 SELECT a_string(500) FROM t1;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-25 -prep {
  faultsim_restore_and_reopen
  db func a_string a_string
  set ::channel [db incrblob -readonly t1 a 1]
  execsql { 
    PRAGMA cache_size = 10;
    BEGIN;
      INSERT INTO t1 VALUES(a_string(3000));
      INSERT INTO t1 VALUES(a_string(3000));
  }
} -body {
  lockrows 30
} -test {
  catch { lockrows 30 }
  catch { db eval COMMIT }
  close $::channel
  faultsim_test_result {0 {}} 
}

do_faultsim_test pagerfault-26 -prep {
  faultsim_delete_and_reopen
  execsql {
    PRAGMA page_size = 1024;
    PRAGMA journal_mode = truncate;
    PRAGMA auto_vacuum = full;
    PRAGMA locking_mode=exclusive;
    CREATE TABLE t1(a, b);
    INSERT INTO t1 VALUES(1, 2);
    PRAGMA page_size = 4096;
  }
} -body {
  execsql {
    VACUUM;
  }
} -test {
  faultsim_test_result {0 {}}

  set contents [db eval {SELECT * FROM t1}]
  if {$contents != "1 2"} { error "Bad database contents ($contents)" }

  set sz [file size test.db]
  if {$testrc!=0 && $sz!=1024*3 && $sz!=4096*3} { 
    error "Expected file size to be 3072 or 12288 bytes - actual size $sz bytes"
  }
  if {$testrc==0 && $sz!=4096*3} { 
    error "Expected file size to be 12288 bytes - actual size $sz bytes"
  }
} 

do_test pagerfault-27-pre {
  faultsim_delete_and_reopen
  db func a_string a_string
  execsql {
    PRAGMA page_size = 1024;
    CREATE TABLE t1(a, b);
    CREATE TABLE t2(a UNIQUE, b UNIQUE);
    INSERT INTO t2 VALUES( a_string(800), a_string(800) );
    INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    INSERT INTO t1 VALUES (a_string(20000), a_string(20000));
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-27 -faults ioerr-persistent -prep {
  faultsim_restore_and_reopen
  db func a_string a_string
  execsql { 
    PRAGMA cache_size = 10;
    BEGIN EXCLUSIVE;
  }
  set ::channel [db incrblob t1 a 1]
} -body {
  puts $::channel [string repeat abc 6000]
  flush $::channel
} -test {
  catchsql { UPDATE t2 SET a = a_string(800), b = a_string(800) }
  catch { close $::channel }
  catchsql { ROLLBACK }
  faultsim_integrity_check
}


#-------------------------------------------------------------------------
#
do_test pagerfault-28-pre {
  faultsim_delete_and_reopen
  db func a_string a_string
  execsql {
    PRAGMA page_size = 512;

    PRAGMA journal_mode = wal;
    PRAGMA wal_autocheckpoint = 0;
    PRAGMA cache_size = 100000;

    BEGIN;
      CREATE TABLE t2(a UNIQUE, b UNIQUE);
      INSERT INTO t2 VALUES( a_string(800), a_string(800) );
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    COMMIT;
    CREATE TABLE t1(a PRIMARY KEY, b);
  }
  expr {[file size test.db-shm] >= 96*1024}
} {1}
faultsim_save_and_close

do_faultsim_test pagerfault-28a -faults oom* -prep {
  faultsim_restore_and_reopen
  execsql { PRAGMA mmap_size=0 }

  sqlite3 db2 test.db
  db2 eval { SELECT count(*) FROM t2 }

  db func a_string a_string
  execsql { 
    BEGIN;
      INSERT INTO t1 VALUES(a_string(2000), a_string(2000));
      INSERT INTO t1 VALUES(a_string(2000), a_string(2000));
  }
  set ::STMT [sqlite3_prepare db "SELECT * FROM t1 ORDER BY a" -1 DUMMY]
  sqlite3_step $::STMT
} -body {
  execsql { ROLLBACK }
} -test {
  db2 close
  sqlite3_finalize $::STMT
  catchsql { ROLLBACK }
  faultsim_integrity_check
}

faultsim_restore_and_reopen
sqlite3 db2 test.db
db2 eval {SELECT count(*) FROM t2}
db close

do_faultsim_test pagerfault-28b -faults oom* -prep {
  sqlite3 db test.db
} -body {
  execsql { SELECT count(*) FROM t2 }
} -test {
  faultsim_test_result {0 2048}
  db close
}

db2 close

#-------------------------------------------------------------------------
# Try this:
#
#    1) Put the pager in ERROR state (error during rollback)
#
#    2) Next time the connection is used inject errors into all xWrite() and
#       xUnlock() calls. This causes the hot-journal rollback to fail and
#       the pager to declare its locking state UNKNOWN.
#
#    3) Same again.
#
#    4a) Stop injecting errors. Allow the rollback to succeed. Check that
#        the database is Ok. Or, 
#
#    4b) Close and reopen the db. Check that the db is Ok.
#
proc custom_injectinstall {} {
  testvfs custom -default true
  custom filter {xWrite xUnlock}
}
proc custom_injectuninstall {} {
  catch {db  close}
  catch {db2 close}
  custom delete
}
proc custom_injectstart {iFail} {
  custom ioerr $iFail 1
}
proc custom_injectstop {} {
  custom ioerr
}
set ::FAULTSIM(custom)          [list      \
  -injectinstall   custom_injectinstall    \
  -injectstart     custom_injectstart      \
  -injectstop      custom_injectstop       \
  -injecterrlist   {{1 {disk I/O error}}}  \
  -injectuninstall custom_injectuninstall  \
]

do_test pagerfault-29-pre {
  faultsim_delete_and_reopen
  db func a_string a_string
  execsql {
    PRAGMA page_size = 1024;
    PRAGMA cache_size = 5;

    BEGIN;
      CREATE TABLE t2(a UNIQUE, b UNIQUE);
      INSERT INTO t2 VALUES( a_string(800), a_string(800) );
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
      INSERT INTO t2 SELECT a_string(800), a_string(800) FROM t2;
    COMMIT;
  }
  expr {[file size test.db] >= 50*1024}
} {1}
faultsim_save_and_close
foreach {tn tt} {
  29 { catchsql ROLLBACK }
  30 { db close ; sqlite3 db test.db }
} {
  do_faultsim_test pagerfault-$tn -faults custom -prep {
    faultsim_restore_and_reopen
      db func a_string a_string
      execsql {
        PRAGMA cache_size = 5;
        BEGIN;
        UPDATE t2 SET a = a_string(799);
      }
  } -body {
    catchsql ROLLBACK
    catchsql ROLLBACK
    catchsql ROLLBACK
  } -test {
    eval $::tt
    if {"ok" != [db one {PRAGMA integrity_check}]} {
      error "integrity check failed"
    }
  }
}

do_test pagerfault-31-pre {
  sqlite3_shutdown
  sqlite3_config_uri 1
} {SQLITE_OK}
do_faultsim_test pagerfault-31 -faults oom* -body {
  sqlite3 db {file:one?mode=memory&cache=shared}
  db eval {
    CREATE TABLE t1(x);
    INSERT INTO t1 VALUES(1);
    SELECT * FROM t1;
  }
} -test {
  faultsim_test_result {0 1} {1 {}}
  catch { db close }
}
sqlite3_shutdown
sqlite3_config_uri 0

do_test pagerfault-32-pre {
  reset_db
  execsql {
    CREATE TABLE t1(x);
    INSERT INTO t1 VALUES('one');
  }
} {}
faultsim_save_and_close

do_faultsim_test pagerfault-32 -prep {
  faultsim_restore_and_reopen
  db eval { SELECT * FROM t1; }
} -body {
  execsql { SELECT * FROM t1; }
} -test {
  faultsim_test_result {0 one}
}
sqlite3_shutdown
sqlite3_config_uri 0

do_faultsim_test pagerfault-33a -prep {
  sqlite3 db :memory:
  execsql {
    CREATE TABLE t1(a, b);
    INSERT INTO t1 VALUES(1, 2);
  }
} -body {
  execsql { VACUUM }
} -test {
  faultsim_test_result {0 {}}
} 
do_faultsim_test pagerfault-33b -prep {
  sqlite3 db ""
  execsql {
    CREATE TABLE t1(a, b);
    INSERT INTO t1 VALUES(1, 2);
  }
} -body {
  execsql { VACUUM }
} -test {
  faultsim_test_result {0 {}}
} 

do_test pagerfault-34-pre {
  reset_db
  execsql {
    CREATE TABLE t1(x PRIMARY KEY);
  }
} {}
faultsim_save_and_close
do_faultsim_test pagerfault-34 -prep {
  faultsim_restore_and_reopen
  execsql {
    BEGIN;
      INSERT INTO t1 VALUES( randomblob(4000) );
      DELETE FROM t1;
  }
} -body {
  execsql COMMIT
} -test {
  faultsim_test_result {0 {}}
} 

do_test pagerfault-35-pre {
  faultsim_delete_and_reopen
  execsql {
    CREATE TABLE t1(x PRIMARY KEY, y);
    INSERT INTO t1 VALUES(randomblob(200), randomblob(200));
    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
  }
  faultsim_save_and_close
} {}
testvfs tv -default 1
tv sectorsize 8192;
tv devchar [list]
do_faultsim_test pagerfault-35 -prep {
  faultsim_restore_and_reopen
} -body {
  execsql { UPDATE t1 SET x=randomblob(200) }
} -test {
  faultsim_test_result {0 {}}
}
catch {db close}
tv delete

sqlite3_shutdown
sqlite3_config_uri 1
do_test pagerfault-36-pre {
  faultsim_delete_and_reopen
  execsql {
    CREATE TABLE t1(x PRIMARY KEY, y);
    INSERT INTO t1 VALUES(randomblob(200), randomblob(200));
    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
  }
  faultsim_save_and_close
} {}
do_faultsim_test pagerfault-36 -prep {
  faultsim_restore
  sqlite3 db file:test.db?cache=shared
  sqlite3 db2 file:test.db?cache=shared
  db2 eval {
    BEGIN;
    SELECT count(*) FROM sqlite_master;
  }
  db eval {
    PRAGMA cache_size = 1;
    BEGIN;
      UPDATE t1 SET x = randomblob(200);
  }
} -body {
  execsql ROLLBACK db
} -test {
  catch { db eval {UPDATE t1 SET x = randomblob(200)} }
  faultsim_test_result {0 {}}
  catch { db close }
  catch { db2 close }
}

sqlite3_shutdown
sqlite3_config_uri 0

finish_test

