//source $testdir/lock_common.tcl
//source $testdir/malloc_common.tcl
//source $testdir/wal_common.tcl
//set testprefix pager1

//// Do not use a codec for tests in this file, as the database file is manipulated directly using tcl scripts (using the [hexio_write] command).
// do_not_use_codec
using Xunit;
public class Pager1 : TesterBase.MultiClientTesterBase
{
    // pager1-1.*: Test inter-process locking (clients in multiple processes).
    //
    // pager1-2.*: Test intra-process locking (multiple clients in this process).
    //
    // pager1-3.*: Savepoint related tests.
    //
    // pager1-4.*: Hot-journal related tests.
    //
    // pager1-5.*: Cases related to multi-file commits.
    //
    // pager1-6.*: Cases related to "PRAGMA max_page_count"
    //
    // pager1-7.*: Cases specific to "PRAGMA journal_mode=TRUNCATE"
    //
    // pager1-8.*: Cases using temporary and in-memory databases.
    //
    // pager1-9.*: Tests related to the backup API.
    //
    // pager1-10.*: Test that the assumed file-system sector-size is limited to 64KB.
    //
    // pager1-12.*: Tests involving "PRAGMA page_size"
    //
    // pager1-13.*: Cases specific to "PRAGMA journal_mode=PERSIST"
    //
    // pager1-14.*: Cases specific to "PRAGMA journal_mode=OFF"
    //
    // pager1-15.*: Varying sqlite3_vfs.szOsFile
    //
    // pager1-16.*: Varying sqlite3_vfs.mxPathname
    //
    // pager1-17.*: Tests related to "PRAGMA omit_readlock" (The omit_readlock pragma has been removed and so have these tests.)
    //
    // pager1-18.*: Test that the pager layer responds correctly if the b-tree requests an invalid page number (due to db corruption).

    //proc recursive_select {id table {script {}}} {
    //  set cnt 0
    //  db eval "SELECT rowid, * FROM $table WHERE rowid = ($id-1)" {
    //    recursive_select $rowid $table $script
    //    incr cnt
    //  }
    //  if {$cnt==0} { eval $script }
    //}

    //set a_string_counter 1
    //proc a_string {n} {
    //  global a_string_counter
    //  incr a_string_counter
    //  string range [string repeat "${a_string_counter}." $n] 1 $n
    //}
    //db func a_string a_string

    int _busys;
    //proc busy {n} {
    //  lappend ::nbusy $n
    //  if {$n>5} { sql2 COMMIT }
    //  return 0
    //}

    [Fact]
    public void T1()
    {
        do_multiclient_test("tn", () =>
        {

            // Create and populate a database table using connection [db]. Check that connections [db2] and [db3] can see the schema and content.
            do_test("pager1-$tn.1", () =>
            {
                sql1(@"
                    CREATE TABLE t1(a PRIMARY KEY, b);
                    CREATE INDEX i1 ON t1(b);
                    INSERT INTO t1 VALUES(1, 'one'); INSERT INTO t1 VALUES(2, 'two');
                ");
            }, null);
            do_test("pager1-$tn.2", () => { sql2(@"SELECT * FROM t1"); }, "1 one 2 two");
            do_test("pager1-$tn.3", () => { sql3(@"SELECT * FROM t1"); }, "1 one 2 two");

            // Open a transaction and add a row using [db]. This puts [db] in RESERVED state. Check that connections [db2] and [db3] can still
            // read the database content as it was before the transaction was opened. [db] should see the inserted row.
            do_test("pager1-$tn.4", () =>
            {
                sql1(@"
                    BEGIN;
                    INSERT INTO t1 VALUES(3, 'three');
                ");
            }, null);
            do_test("pager1-$tn.5", () => { sql2(@"SELECT * FROM t1"); }, "1 one 2 two");
            do_test("pager1-$tn.7", () => { sql1(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");

            // [db] still has an open write transaction. Check that this prevents other connections (specifically [db2]) from writing to the database.
            //
            // Even if [db2] opens a transaction first, it may not write to the database. After the attempt to write the db within a transaction, 
            // [db2] is left with an open transaction, but not a read-lock on the main database. So it does not prevent [db] from committing.
            do_test("pager1-$tn.8", () =>
            {
                csql2(@"UPDATE t1 SET a = a + 10");
            }, "1 {database is locked}");
            do_test("pager1-$tn.9", () =>
            {
                csql2(@"
                    BEGIN;
                    UPDATE t1 SET a = a + 10;
                ");
            }, "1 {database is locked}");

            // Have [db] commit its transactions. Check the other connections can now see the new database content.
            do_test("pager1-$tn.10", () => { sql1(@"COMMIT"); }, null);
            do_test("pager1-$tn.11", () => { sql1(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");
            do_test("pager1-$tn.12", () => { sql2(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");
            do_test("pager1-$tn.13", () => { sql3(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");

            // Check that, as noted above, [db2] really did keep an open transaction after the attempt to write the database failed.
            do_test("pager1-$tn.14", () =>
            {
                csql2(@"BEGIN");
            }, "1 {cannot start a transaction within a transaction}");
            do_test("pager1-$tn.15", () => { sql2(@"ROLLBACK"); }, null);

            // Have [db2] open a transaction and take a read-lock on the database. Check that this prevents [db] from writing to the database (outside
            // of any transaction). After this fails, check that [db3] can read the db (showing that [db] did not take a PENDING lock etc.)
            do_test("pager1-$tn.15", () =>
            {
                sql2(@"BEGIN; SELECT * FROM t1;");
            }, "1 one 2 two 3 three");
            do_test("pager1-$tn.16", () =>
            {
                csql1(@"UPDATE t1 SET a = a + 10");
            }, "1 {database is locked}");
            do_test("pager1-$tn.17", () => { sql3(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");

            // This time, have [db] open a transaction before writing the database. This works - [db] gets a RESERVED lock which does not conflict with
            // the SHARED lock [db2] is holding.
            do_test("pager1-$tn.18", () =>
            {
                sql1(@"
                    BEGIN;  
                    UPDATE t1 SET a = a + 10; 
                ");
            }, null);
            do_test("pager1-$tn-19", () =>
            {
                sql1(@"PRAGMA lock_status");
            }, "main reserved temp closed");
            do_test("pager1-$tn-20", () =>
            {
                sql2(@"PRAGMA lock_status");
            }, "main shared temp closed");

            // Check that all connections can still read the database. Only [db] sees the updated content (as the transaction has not been committed yet).
            do_test("pager1-$tn.21", () => { sql1(@"SELECT * FROM t1"); }, "11 one 12 two 13 three");
            do_test("pager1-$tn.22", () => { sql2(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");
            do_test("pager1-$tn.23", () => { sql3(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");

            // Because [db2] still has the SHARED lock, [db] is unable to commit the transaction. If it tries, an error is returned and the connection 
            // upgrades to a PENDING lock.
            //
            // Once this happens, [db] can read the database and see the new content, [db2] (still holding SHARED) can still read the old content, but [db3]
            // (not holding any lock) is prevented by [db]'s PENDING from reading the database.
            do_test("pager1-$tn.24", () => { csql1(@"COMMIT"); }, "1 {database is locked}");
            do_test("pager1-$tn-25", () =>
            {
                sql1(@"PRAGMA lock_status");
            }, "main pending temp closed");
            do_test("pager1-$tn.26", () => { sql1(@"SELECT * FROM t1"); }, "11 one 12 two 13 three");
            do_test("pager1-$tn.27", () => { sql2(@"SELECT * FROM t1"); }, "1 one 2 two 3 three");
            do_test("pager1-$tn.28", () => { csql3(@"SELECT * FROM t1"); }, "1 {database is locked}");

            // Have [db2] commit its read transaction, releasing the SHARED lock it is holding. Now, neither [db2] nor [db3] may read the database (as [db]
            // is still holding a PENDING).
            //
            do_test("pager1-$tn.29", () => { sql2(@"COMMIT"); }, null);
            do_test("pager1-$tn.30", () => { csql2(@"SELECT * FROM t1"); }, "1 {database is locked}");
            do_test("pager1-$tn.31", () => { csql3(@"SELECT * FROM t1"); }, "1 {database is locked}");

            // [db] is now able to commit the transaction. Once the transaction is committed, all three connections can read the new content.
            do_test("pager1-$tn.25", () => { sql1(@"UPDATE t1 SET a = a+10"); }, null);
            do_test("pager1-$tn.26", () => { sql1(@"COMMIT"); }, null);
            do_test("pager1-$tn.27", () => { sql1(@"SELECT * FROM t1"); }, "21 one 22 two 23 three");
            do_test("pager1-$tn.27", () => { sql2(@"SELECT * FROM t1"); }, "21 one 22 two 23 three");
            do_test("pager1-$tn.28", () => { sql3(@"SELECT * FROM t1"); }, "21 one 22 two 23 three");

            // Install a busy-handler for connection [db].
            //_busys = [list]
            //db busy busy

            do_test("pager1-$tn.29", () =>
            {
                sql1(@"BEGIN ; INSERT INTO t1 VALUES('x', 'y')");
            }, null);
            do_test("pager1-$tn.30", () =>
            {
                sql2(@"BEGIN ; SELECT * FROM t1");
            }, "21 one 22 two 23 three");
            do_test("pager1-$tn.31", () => { sql1(@"COMMIT"); }, null);
            //do_test("pager1-$tn.32", () => { _busys = 0}, "0 1 2 3 4 5 6");

        });

        //#-------------------------------------------------------------------------
        //# Savepoint related test cases.
        //#
        //# pager1-3.1.2.*: Force a savepoint rollback to cause the database file
        //#                 to grow.
        //#
        //# pager1-3.1.3.*: Use a journal created in synchronous=off mode as part
        //#                 of a savepoint rollback.
        //# 
        //do_test pager1-3.1.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    CREATE TABLE t1(a PRIMARY KEY, b);
        //    CREATE TABLE counter(
        //      i CHECK (i<5), 
        //      u CHECK (u<10)
        //    );
        //    INSERT INTO counter VALUES(0, 0);
        //    CREATE TRIGGER tr1 AFTER INSERT ON t1 BEGIN
        //      UPDATE counter SET i = i+1;
        //    END;
        //    CREATE TRIGGER tr2 AFTER UPDATE ON t1 BEGIN
        //      UPDATE counter SET u = u+1;
        //    END;
        //  }
        //  execsql { SELECT * FROM counter }
        //} {0 0}

        //do_execsql_test pager1-3.1.2 {
        //  PRAGMA cache_size = 10;
        //  BEGIN;
        //    INSERT INTO t1 VALUES(1, randomblob(1500));
        //    INSERT INTO t1 VALUES(2, randomblob(1500));
        //    INSERT INTO t1 VALUES(3, randomblob(1500));
        //    SELECT * FROM counter;
        //} {3 0}
        //do_catchsql_test pager1-3.1.3 {
        //    INSERT INTO t1 SELECT a+3, randomblob(1500) FROM t1
        //} {1 {constraint failed}}
        //do_execsql_test pager1-3.4 { SELECT * FROM counter } {3 0}
        //do_execsql_test pager1-3.5 { SELECT a FROM t1 } {1 2 3}
        //do_execsql_test pager1-3.6 { COMMIT } {}

        //foreach {tn sql tcl} {
        //  7  { PRAGMA synchronous = NORMAL ; PRAGMA temp_store = 0 } {
        //    testvfs tv -default 1
        //    tv devchar safe_append
        //  }
        //  8  { PRAGMA synchronous = NORMAL ; PRAGMA temp_store = 2 } {
        //    testvfs tv -default 1
        //    tv devchar sequential
        //  }
        //  9  { PRAGMA synchronous = FULL } { }
        //  10 { PRAGMA synchronous = NORMAL } { }
        //  11 { PRAGMA synchronous = OFF } { }
        //  12 { PRAGMA synchronous = FULL ; PRAGMA fullfsync = 1 } { }
        //  13 { PRAGMA synchronous = FULL } {
        //    testvfs tv -default 1
        //    tv devchar sequential
        //  }
        //  14 { PRAGMA locking_mode = EXCLUSIVE } {
        //  }
        //} {
        //  do_test pager1-3.$tn.1 {
        //    eval $tcl
        //    faultsim_delete_and_reopen
        //    db func a_string a_string
        //    execsql $sql
        //    execsql {
        //      PRAGMA auto_vacuum = 2;
        //      PRAGMA cache_size = 10;
        //      CREATE TABLE z(x INTEGER PRIMARY KEY, y);
        //      BEGIN;
        //        INSERT INTO z VALUES(NULL, a_string(800));
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     --   2
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     --   4
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     --   8
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     --  16
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     --  32
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     --  64
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     -- 128
        //        INSERT INTO z SELECT NULL, a_string(800) FROM z;     -- 256
        //      COMMIT;
        //    }
        //    execsql { PRAGMA auto_vacuum }
        //  } {2}
        //  do_execsql_test pager1-3.$tn.2 {
        //    BEGIN;
        //      INSERT INTO z VALUES(NULL, a_string(800));
        //      INSERT INTO z VALUES(NULL, a_string(800));
        //      SAVEPOINT one;
        //        UPDATE z SET y = NULL WHERE x>256;
        //        PRAGMA incremental_vacuum;
        //        SELECT count(*) FROM z WHERE x < 100;
        //      ROLLBACK TO one;
        //    COMMIT;
        //  } {99}

        //  do_execsql_test pager1-3.$tn.3 {
        //    BEGIN;
        //      SAVEPOINT one;
        //        UPDATE z SET y = y||x;
        //      ROLLBACK TO one;
        //    COMMIT;
        //    SELECT count(*) FROM z;
        //  } {258}

        //  do_execsql_test pager1-3.$tn.4 {
        //    SAVEPOINT one;
        //      UPDATE z SET y = y||x;
        //    ROLLBACK TO one;
        //  } {}
        //  do_execsql_test pager1-3.$tn.5 {
        //    SELECT count(*) FROM z;
        //    RELEASE one;
        //    PRAGMA integrity_check;
        //  } {258 ok}

        //  do_execsql_test pager1-3.$tn.6 {
        //    SAVEPOINT one;
        //    RELEASE one;
        //  } {}

        //  db close
        //  catch { tv delete }
        //}

        //#-------------------------------------------------------------------------
        //# Hot journal rollback related test cases.
        //#
        //# pager1.4.1.*: Test that the pager module deletes very small invalid
        //#               journal files.
        //#
        //# pager1.4.2.*: Test that if the master journal pointer at the end of a
        //#               hot-journal file appears to be corrupt (checksum does not
        //#               compute) the associated journal is rolled back (and no
        //#               xAccess() call to check for the presence of any master 
        //#               journal file is made).
        //#
        //# pager1.4.3.*: Test that the contents of a hot-journal are ignored if the
        //#               page-size or sector-size in the journal header appear to
        //#               be invalid (too large, too small or not a power of 2).
        //#
        //# pager1.4.4.*: Test hot-journal rollback of journal file with a master
        //#               journal pointer generated in various "PRAGMA synchronous"
        //#               modes.
        //#
        //# pager1.4.5.*: Test that hot-journal rollback stops if it encounters a
        //#               journal-record for which the checksum fails.
        //#
        //# pager1.4.6.*: Test that when rolling back a hot-journal that contains a
        //#               master journal pointer, the master journal file is deleted
        //#               after all the hot-journals that refer to it are deleted.
        //#
        //# pager1.4.7.*: Test that if a hot-journal file exists but a client can
        //#               open it for reading only, the database cannot be accessed and
        //#               SQLITE_CANTOPEN is returned.
        //# 
        //do_test pager1.4.1.1 {
        //  faultsim_delete_and_reopen
        //  execsql { 
        //    CREATE TABLE x(y, z);
        //    INSERT INTO x VALUES(1, 2);
        //  }
        //  set fd [open test.db-journal w]
        //  puts -nonewline $fd "helloworld"
        //  close $fd
        //  file exists test.db-journal
        //} {1}
        //do_test pager1.4.1.2 { execsql { SELECT * FROM x } } {1 2}
        //do_test pager1.4.1.3 { file exists test.db-journal } {0}

        //# Set up a [testvfs] to snapshot the file-system just before SQLite
        //# deletes the master-journal to commit a multi-file transaction.
        //#
        //# In subsequent test cases, invoking [faultsim_restore_and_reopen] sets
        //# up the file system to contain two databases, two hot-journal files and
        //# a master-journal.
        //#
        //do_test pager1.4.2.1 {
        //  testvfs tstvfs -default 1
        //  tstvfs filter xDelete
        //  tstvfs script xDeleteCallback
        //  proc xDeleteCallback {method file args} {
        //    set file [file tail $file]
        //    if { [string match *mj* $file] } { faultsim_save }
        //  }
        //  faultsim_delete_and_reopen
        //  db func a_string a_string
        //  execsql {
        //    ATTACH 'test.db2' AS aux;
        //    PRAGMA journal_mode = DELETE;
        //    PRAGMA main.cache_size = 10;
        //    PRAGMA aux.cache_size = 10;
        //    CREATE TABLE t1(a UNIQUE, b UNIQUE);
        //    CREATE TABLE aux.t2(a UNIQUE, b UNIQUE);
        //    INSERT INTO t1 VALUES(a_string(200), a_string(300));
        //    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
        //    INSERT INTO t1 SELECT a_string(200), a_string(300) FROM t1;
        //    INSERT INTO t2 SELECT * FROM t1;
        //    BEGIN;
        //      INSERT INTO t1 SELECT a_string(201), a_string(301) FROM t1;
        //      INSERT INTO t1 SELECT a_string(202), a_string(302) FROM t1;
        //      INSERT INTO t1 SELECT a_string(203), a_string(303) FROM t1;
        //      INSERT INTO t1 SELECT a_string(204), a_string(304) FROM t1;
        //      REPLACE INTO t2 SELECT * FROM t1;
        //    COMMIT;
        //  }
        //  db close
        //  tstvfs delete
        //} {}

        //if {$::tcl_platform(platform)!="windows"} {
        //do_test pager1.4.2.2 {
        //  faultsim_restore_and_reopen
        //  execsql {
        //    SELECT count(*) FROM t1;
        //    PRAGMA integrity_check;
        //  }
        //} {4 ok}
        //do_test pager1.4.2.3 {
        //  faultsim_restore_and_reopen
        //  foreach f [glob test.db-mj*] { forcedelete $f }
        //  execsql {
        //    SELECT count(*) FROM t1;
        //    PRAGMA integrity_check;
        //  }
        //} {64 ok}
        //do_test pager1.4.2.4 {
        //  faultsim_restore_and_reopen
        //  hexio_write test.db-journal [expr [file size test.db-journal]-30] 123456
        //  execsql {
        //    SELECT count(*) FROM t1;
        //    PRAGMA integrity_check;
        //  }
        //} {4 ok}
        //do_test pager1.4.2.5 {
        //  faultsim_restore_and_reopen
        //  hexio_write test.db-journal [expr [file size test.db-journal]-30] 123456
        //  foreach f [glob test.db-mj*] { forcedelete $f }
        //  execsql {
        //    SELECT count(*) FROM t1;
        //    PRAGMA integrity_check;
        //  }
        //} {4 ok}
        //}

        //do_test pager1.4.3.1 {
        //  testvfs tstvfs -default 1
        //  tstvfs filter xSync
        //  tstvfs script xSyncCallback
        //  proc xSyncCallback {method file args} {
        //    set file [file tail $file]
        //    if { 0==[string match *journal $file] } { faultsim_save }
        //  }
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA journal_mode = DELETE;
        //    CREATE TABLE t1(a, b);
        //    INSERT INTO t1 VALUES(1, 2);
        //    INSERT INTO t1 VALUES(3, 4);
        //  }
        //  db close
        //  tstvfs delete
        //} {}

        //foreach {tn ofst value result} {
        //          2   20    31       {1 2 3 4}
        //          3   20    32       {1 2 3 4}
        //          4   20    33       {1 2 3 4}
        //          5   20    65536    {1 2 3 4}
        //          6   20    131072   {1 2 3 4}

        //          7   24    511      {1 2 3 4}
        //          8   24    513      {1 2 3 4}
        //          9   24    131072   {1 2 3 4}

        //         10   32    65536    {1 2}
        //} {
        //  do_test pager1.4.3.$tn {
        //    faultsim_restore_and_reopen
        //    hexio_write test.db-journal $ofst [format %.8x $value]
        //    execsql { SELECT * FROM t1 }
        //  } $result
        //}
        //db close

        //# Set up a VFS that snapshots the file-system just before a master journal
        //# file is deleted to commit a multi-file transaction. Specifically, the
        //# file-system is saved just before the xDelete() call to remove the 
        //# master journal file from the file-system.
        //#
        //set pwd [get_pwd]
        //testvfs tv -default 1
        //tv script copy_on_mj_delete
        //set ::mj_filename_length 0
        //proc copy_on_mj_delete {method filename args} {
        //  if {[string match *mj* [file tail $filename]]} { 
        //    #
        //    # NOTE: Is the file name relative?  If so, add the length of the current
        //    #       directory.
        //    #
        //    if {[is_relative_file $filename]} {
        //      set ::mj_filename_length \
        //        [expr {[string length $filename] + [string length $::pwd]}]
        //    } else {
        //      set ::mj_filename_length [string length $filename]
        //    }
        //    faultsim_save 
        //  }
        //  return SQLITE_OK
        //}

        //foreach {tn1 tcl} {
        //  1 { set prefix "test.db" }
        //  2 { 
        //    # This test depends on the underlying VFS being able to open paths
        //    # 512 bytes in length. The idea is to create a hot-journal file that
        //    # contains a master-journal pointer so large that it could contain
        //    # a valid page record (if the file page-size is 512 bytes). So as to
        //    # make sure SQLite doesn't get confused by this.
        //    #
        //    set nPadding [expr 511 - $::mj_filename_length]
        //    if {$tcl_platform(platform)=="windows"} {
        //      # TBD need to figure out how to do this correctly for Windows!!!
        //      set nPadding [expr 255 - $::mj_filename_length]
        //    }

        //    # We cannot just create a really long database file name to open, as
        //    # Linux limits a single component of a path to 255 bytes by default
        //    # (and presumably other systems have limits too). So create a directory
        //    # hierarchy to work in.
        //    #
        //    set dirname "d123456789012345678901234567890/"
        //    set nDir [expr $nPadding / 32]
        //    if { $nDir } {
        //      set p [string repeat $dirname $nDir]
        //      file mkdir $p
        //      cd $p
        //    }

        //    set padding [string repeat x [expr $nPadding %32]]
        //    set prefix "test.db${padding}"
        //  }
        //} {
        //  eval $tcl
        //  foreach {tn2 sql} {
        //    o { 
        //      PRAGMA main.synchronous=OFF;
        //      PRAGMA aux.synchronous=OFF;
        //      PRAGMA journal_mode = DELETE;
        //    }
        //    o512 { 
        //      PRAGMA main.synchronous=OFF;
        //      PRAGMA aux.synchronous=OFF;
        //      PRAGMA main.page_size = 512;
        //      PRAGMA aux.page_size = 512;
        //      PRAGMA journal_mode = DELETE;
        //    }
        //    n { 
        //      PRAGMA main.synchronous=NORMAL;
        //      PRAGMA aux.synchronous=NORMAL;
        //      PRAGMA journal_mode = DELETE;
        //    }
        //    f { 
        //      PRAGMA main.synchronous=FULL;
        //      PRAGMA aux.synchronous=FULL;
        //      PRAGMA journal_mode = DELETE;
        //    }
        //  } {

        //    set tn "${tn1}.${tn2}"

        //    # Set up a connection to have two databases, test.db (main) and 
        //    # test.db2 (aux). Then run a multi-file transaction on them. The
        //    # VFS will snapshot the file-system just before the master-journal
        //    # file is deleted to commit the transaction.
        //    #
        //    tv filter xDelete
        //    do_test pager1-4.4.$tn.1 {
        //      faultsim_delete_and_reopen $prefix
        //      execsql "
        //        ATTACH '${prefix}2' AS aux;
        //        $sql
        //        CREATE TABLE a(x);
        //        CREATE TABLE aux.b(x);
        //        INSERT INTO a VALUES('double-you');
        //        INSERT INTO a VALUES('why');
        //        INSERT INTO a VALUES('zed');
        //        INSERT INTO b VALUES('won');
        //        INSERT INTO b VALUES('too');
        //        INSERT INTO b VALUES('free');
        //      "
        //      execsql {
        //        BEGIN;
        //          INSERT INTO a SELECT * FROM b WHERE rowid<=3;
        //          INSERT INTO b SELECT * FROM a WHERE rowid<=3;
        //        COMMIT;
        //      }
        //    } {}
        //    tv filter {}

        //    # Check that the transaction was committed successfully.
        //    #
        //    do_execsql_test pager1-4.4.$tn.2 {
        //      SELECT * FROM a
        //    } {double-you why zed won too free}
        //    do_execsql_test pager1-4.4.$tn.3 {
        //      SELECT * FROM b
        //    } {won too free double-you why zed}

        //    # Restore the file-system and reopen the databases. Check that it now
        //    # appears that the transaction was not committed (because the file-system
        //    # was restored to the state where it had not been).
        //    #
        //    do_test pager1-4.4.$tn.4 {
        //      faultsim_restore_and_reopen $prefix
        //      execsql "ATTACH '${prefix}2' AS aux"
        //    } {}
        //    do_execsql_test pager1-4.4.$tn.5 {SELECT * FROM a} {double-you why zed}
        //    do_execsql_test pager1-4.4.$tn.6 {SELECT * FROM b} {won too free}

        //    # Restore the file-system again. This time, before reopening the databases,
        //    # delete the master-journal file from the file-system. It now appears that
        //    # the transaction was committed (no master-journal file == no rollback).
        //    #
        //    do_test pager1-4.4.$tn.7 {
        //      faultsim_restore_and_reopen $prefix
        //      foreach f [glob ${prefix}-mj*] { forcedelete $f }
        //      execsql "ATTACH '${prefix}2' AS aux"
        //    } {}
        //    do_execsql_test pager1-4.4.$tn.8 {
        //      SELECT * FROM a
        //    } {double-you why zed won too free}
        //    do_execsql_test pager1-4.4.$tn.9 {
        //      SELECT * FROM b
        //    } {won too free double-you why zed}
        //  }

        //  cd $pwd
        //}
        //db close
        //tv delete
        //forcedelete $dirname


        //# Set up a VFS to make a copy of the file-system just before deleting a
        //# journal file to commit a transaction. The transaction modifies exactly
        //# two database pages (and page 1 - the change counter).
        //#
        //testvfs tv -default 1
        //tv sectorsize 512
        //tv script copy_on_journal_delete
        //tv filter xDelete
        //proc copy_on_journal_delete {method filename args} {
        //  if {[string match *journal $filename]} faultsim_save 
        //  return SQLITE_OK
        //}
        //faultsim_delete_and_reopen
        //do_execsql_test pager1.4.5.1 {
        //  PRAGMA journal_mode = DELETE;
        //  PRAGMA page_size = 1024;
        //  CREATE TABLE t1(a, b);
        //  CREATE TABLE t2(a, b);
        //  INSERT INTO t1 VALUES('I', 'II');
        //  INSERT INTO t2 VALUES('III', 'IV');
        //  BEGIN;
        //    INSERT INTO t1 VALUES(1, 2);
        //    INSERT INTO t2 VALUES(3, 4);
        //  COMMIT;
        //} {delete}
        //tv filter {}

        //# Check the transaction was committed:
        //#
        //do_execsql_test pager1.4.5.2 {
        //  SELECT * FROM t1;
        //  SELECT * FROM t2;
        //} {I II 1 2 III IV 3 4}

        //# Now try four tests:
        //#
        //#  pager1-4.5.3: Restore the file-system. Check that the whole transaction 
        //#                is rolled back.
        //#
        //#  pager1-4.5.4: Restore the file-system. Corrupt the first record in the
        //#                journal. Check the transaction is not rolled back.
        //#
        //#  pager1-4.5.5: Restore the file-system. Corrupt the second record in the
        //#                journal. Check that the first record in the transaction is 
        //#                played back, but not the second.
        //#
        //#  pager1-4.5.6: Restore the file-system. Try to open the database with a
        //#                readonly connection. This should fail, as a read-only
        //#                connection cannot roll back the database file.
        //#
        //faultsim_restore_and_reopen
        //do_execsql_test pager1.4.5.3 {
        //  SELECT * FROM t1;
        //  SELECT * FROM t2;
        //} {I II III IV}
        //faultsim_restore_and_reopen
        //hexio_write test.db-journal [expr 512+4+1024 - 202] 0123456789ABCDEF
        //do_execsql_test pager1.4.5.4 {
        //  SELECT * FROM t1;
        //  SELECT * FROM t2;
        //} {I II 1 2 III IV 3 4}
        //faultsim_restore_and_reopen
        //hexio_write test.db-journal [expr 512+4+1024+4+4+1024 - 202] 0123456789ABCDEF
        //do_execsql_test pager1.4.5.5 {
        //  SELECT * FROM t1;
        //  SELECT * FROM t2;
        //} {I II III IV 3 4}

        //faultsim_restore_and_reopen
        //db close
        //sqlite3 db test.db -readonly 1
        //do_catchsql_test pager1.4.5.6 {
        //  SELECT * FROM t1;
        //  SELECT * FROM t2;
        //} {1 {attempt to write a readonly database}}
        //db close

        //# Snapshot the file-system just before multi-file commit. Save the name
        //# of the master journal file in $::mj_filename.
        //#
        //tv script copy_on_mj_delete
        //tv filter xDelete
        //proc copy_on_mj_delete {method filename args} {
        //  if {[string match *mj* [file tail $filename]]} { 
        //    set ::mj_filename $filename
        //    faultsim_save 
        //  }
        //  return SQLITE_OK
        //}
        //do_test pager1.4.6.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA journal_mode = DELETE;
        //    ATTACH 'test.db2' AS two;
        //    CREATE TABLE t1(a, b);
        //    CREATE TABLE two.t2(a, b);
        //    INSERT INTO t1 VALUES(1, 't1.1');
        //    INSERT INTO t2 VALUES(1, 't2.1');
        //    BEGIN;
        //      UPDATE t1 SET b = 't1.2';
        //      UPDATE t2 SET b = 't2.2';
        //    COMMIT;
        //  }
        //  tv filter {}
        //  db close
        //} {}

        //faultsim_restore_and_reopen
        //do_execsql_test pager1.4.6.2 { SELECT * FROM t1 }           {1 t1.1}
        //do_test         pager1.4.6.3 { file exists $::mj_filename } {1}
        //do_execsql_test pager1.4.6.4 {
        //  ATTACH 'test.db2' AS two;
        //  SELECT * FROM t2;
        //} {1 t2.1}
        //do_test pager1.4.6.5 { file exists $::mj_filename } {0}

        //faultsim_restore_and_reopen
        //db close
        //do_test pager1.4.6.8 {
        //  set ::mj_filename1 $::mj_filename
        //  tv filter xDelete
        //  sqlite3 db test.db2
        //  execsql {
        //    PRAGMA journal_mode = DELETE;
        //    ATTACH 'test.db3' AS three;
        //    CREATE TABLE three.t3(a, b);
        //    INSERT INTO t3 VALUES(1, 't3.1');
        //    BEGIN;
        //      UPDATE t2 SET b = 't2.3';
        //      UPDATE t3 SET b = 't3.3';
        //    COMMIT;
        //  }
        //  expr {$::mj_filename1 != $::mj_filename}
        //} {1}
        //faultsim_restore_and_reopen
        //tv filter {}

        //# The file-system now contains:
        //#
        //#   * three databases
        //#   * three hot-journal files
        //#   * two master-journal files.
        //#
        //# The hot-journals associated with test.db2 and test.db3 point to
        //# master journal $::mj_filename. The hot-journal file associated with
        //# test.db points to master journal $::mj_filename1. So reading from
        //# test.db should delete $::mj_filename1.
        //#
        //do_test pager1.4.6.9 {
        //  lsort [glob test.db*]
        //} [lsort [list                                           \
        //  test.db test.db2 test.db3                              \
        //  test.db-journal test.db2-journal test.db3-journal      \
        //  [file tail $::mj_filename] [file tail $::mj_filename1]
        //]]

        //# The master-journal $::mj_filename1 contains pointers to test.db and 
        //# test.db2. However the hot-journal associated with test.db2 points to
        //# a different master-journal. Therefore, reading from test.db only should
        //# be enough to cause SQLite to delete $::mj_filename1.
        //#
        //do_test         pager1.4.6.10 { file exists $::mj_filename  } {1}
        //do_test         pager1.4.6.11 { file exists $::mj_filename1 } {1}
        //do_execsql_test pager1.4.6.12 { SELECT * FROM t1 } {1 t1.1}
        //do_test         pager1.4.6.13 { file exists $::mj_filename  } {1}
        //do_test         pager1.4.6.14 { file exists $::mj_filename1 } {0}

        //do_execsql_test pager1.4.6.12 {
        //  ATTACH 'test.db2' AS two;
        //  SELECT * FROM t2;
        //} {1 t2.1}
        //do_test         pager1.4.6.13 { file exists $::mj_filename }  {1}
        //do_execsql_test pager1.4.6.14 {
        //  ATTACH 'test.db3' AS three;
        //  SELECT * FROM t3;
        //} {1 t3.1}
        //do_test         pager1.4.6.15 { file exists $::mj_filename }  {0}

        //db close
        //tv delete

        //testvfs tv -default 1
        //tv sectorsize 512
        //tv script copy_on_journal_delete
        //tv filter xDelete
        //proc copy_on_journal_delete {method filename args} {
        //  if {[string match *journal $filename]} faultsim_save 
        //  return SQLITE_OK
        //}
        //faultsim_delete_and_reopen
        //do_execsql_test pager1.4.7.1 {
        //  PRAGMA journal_mode = DELETE;
        //  CREATE TABLE t1(x PRIMARY KEY, y);
        //  CREATE INDEX i1 ON t1(y);
        //  INSERT INTO t1 VALUES('I',   'one');
        //  INSERT INTO t1 VALUES('II',  'four');
        //  INSERT INTO t1 VALUES('III', 'nine');
        //  BEGIN;
        //    INSERT INTO t1 VALUES('IV', 'sixteen');
        //    INSERT INTO t1 VALUES('V' , 'twentyfive');
        //  COMMIT;
        //} {delete}
        //tv filter {}
        //db close
        //tv delete 
        //catch {
        //  test_syscall install fchmod
        //  test_syscall fault 1 1
        //}
        //do_test pager1.4.7.2 {
        //  faultsim_restore_and_reopen
        //  catch {file attributes test.db-journal -permissions r--------}
        //  catch {file attributes test.db-journal -readonly 1}
        //  catchsql { SELECT * FROM t1 }
        //} {1 {unable to open database file}}
        //catch {
        //  test_syscall reset
        //  test_syscall fault 0 0
        //}
        //do_test pager1.4.7.3 {
        //  db close
        //  catch {file attributes test.db-journal -permissions rw-rw-rw-}
        //  catch {file attributes test.db-journal -readonly 0}
        //  delete_file test.db-journal
        //  file exists test.db-journal
        //} {0}
        //do_test pager1.4.8.1 {
        //  catch {file attributes test.db -permissions r--------}
        //  catch {file attributes test.db -readonly 1}
        //  sqlite3 db test.db
        //  db eval { SELECT * FROM t1 }
        //  sqlite3_db_readonly db main
        //} {1}
        //do_test pager1.4.8.2 {
        //  sqlite3_db_readonly db xyz
        //} {-1}
        //do_test pager1.4.8.3 {
        //  db close
        //  catch {file attributes test.db -readonly 0}
        //  catch {file attributes test.db -permissions rw-rw-rw-} msg
        //  sqlite3 db test.db
        //  db eval { SELECT * FROM t1 }
        //  sqlite3_db_readonly db main
        //} {0}

        //#-------------------------------------------------------------------------
        //# The following tests deal with multi-file commits.
        //#
        //# pager1-5.1.*: The case where a multi-file cannot be committed because
        //#               another connection is holding a SHARED lock on one of the
        //#               files. After the SHARED lock is removed, the COMMIT succeeds.
        //#
        //# pager1-5.2.*: Multi-file commits with journal_mode=memory.
        //#
        //# pager1-5.3.*: Multi-file commits with journal_mode=memory.
        //#
        //# pager1-5.4.*: Check that with synchronous=normal, the master-journal file
        //#               name is added to a journal file immediately after the last
        //#               journal record. But with synchronous=full, extra unused space
        //#               is allocated between the last journal record and the 
        //#               master-journal file name so that the master-journal file
        //#               name does not lie on the same sector as the last journal file
        //#               record.
        //#
        //# pager1-5.5.*: Check that in journal_mode=PERSIST mode, a journal file is
        //#               truncated to zero bytes when a multi-file transaction is 
        //#               committed (instead of the first couple of bytes being zeroed).
        //#
        //#
        //do_test pager1-5.1.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    ATTACH 'test.db2' AS aux;
        //    CREATE TABLE t1(a, b);
        //    CREATE TABLE aux.t2(a, b);
        //    INSERT INTO t1 VALUES(17, 'Lenin');
        //    INSERT INTO t1 VALUES(22, 'Stalin');
        //    INSERT INTO t1 VALUES(53, 'Khrushchev');
        //  }
        //} {}
        //do_test pager1-5.1.2 {
        //  execsql {
        //    BEGIN;
        //      INSERT INTO t1 VALUES(64, 'Brezhnev');
        //      INSERT INTO t2 SELECT * FROM t1;
        //  }
        //  sqlite3 db2 test.db2
        //  execsql {
        //    BEGIN;
        //      SELECT * FROM t2;
        //  } db2
        //} {}
        //do_test pager1-5.1.3 {
        //  catchsql COMMIT
        //} {1 {database is locked}}
        //do_test pager1-5.1.4 {
        //  execsql COMMIT db2
        //  execsql COMMIT
        //  execsql { SELECT * FROM t2 } db2
        //} {17 Lenin 22 Stalin 53 Khrushchev 64 Brezhnev}
        //do_test pager1-5.1.5 {
        //  db2 close
        //} {}

        //do_test pager1-5.2.1 {
        //  execsql {
        //    PRAGMA journal_mode = memory;
        //    BEGIN;
        //      INSERT INTO t1 VALUES(84, 'Andropov');
        //      INSERT INTO t2 VALUES(84, 'Andropov');
        //    COMMIT;
        //  }
        //} {memory}
        //do_test pager1-5.3.1 {
        //  execsql {
        //    PRAGMA journal_mode = off;
        //    BEGIN;
        //      INSERT INTO t1 VALUES(85, 'Gorbachev');
        //      INSERT INTO t2 VALUES(85, 'Gorbachev');
        //    COMMIT;
        //  }
        //} {off}

        //do_test pager1-5.4.1 {
        //  db close
        //  testvfs tv
        //  sqlite3 db test.db -vfs tv
        //  execsql { ATTACH 'test.db2' AS aux }

        //  tv filter xDelete
        //  tv script max_journal_size
        //  tv sectorsize 512
        //  set ::max_journal 0
        //  proc max_journal_size {method args} {
        //    set sz 0
        //    catch { set sz [file size test.db-journal] }
        //    if {$sz > $::max_journal} {
        //      set ::max_journal $sz
        //    }
        //    return SQLITE_OK
        //  }
        //  execsql {
        //    PRAGMA journal_mode = DELETE;
        //    PRAGMA synchronous = NORMAL;
        //    BEGIN;
        //      INSERT INTO t1 VALUES(85, 'Gorbachev');
        //      INSERT INTO t2 VALUES(85, 'Gorbachev');
        //    COMMIT;
        //  }

        //  # The size of the journal file is now:
        //  # 
        //  #   1) 512 byte header +
        //  #   2) 2 * (1024+8) byte records +
        //  #   3) 20+N bytes of master-journal pointer, where N is the size of 
        //  #      the master-journal name encoded as utf-8 with no nul term.
        //  #
        //  set mj_pointer [expr {
        //    20 + [string length "test.db-mjXXXXXX9XX"]
        //  }]
        //  #
        //  #   NOTE: For item 3 above, if the current SQLite VFS lacks the concept of a
        //  #         current directory, the length of the current directory name plus 1
        //  #         character for the directory separator character are NOT counted as
        //  #         part of the total size; otherwise, they are.
        //  #
        //  ifcapable curdir {
        //    set mj_pointer [expr {$mj_pointer + [string length [get_pwd]] + 1}]
        //  }
        //  expr {$::max_journal==(512+2*(1024+8)+$mj_pointer)}
        //} 1
        //do_test pager1-5.4.2 {
        //  set ::max_journal 0
        //  execsql {
        //    PRAGMA synchronous = full;
        //    BEGIN;
        //      DELETE FROM t1 WHERE b = 'Lenin';
        //      DELETE FROM t2 WHERE b = 'Lenin';
        //    COMMIT;
        //  }

        //  # In synchronous=full mode, the master-journal pointer is not written
        //  # directly after the last record in the journal file. Instead, it is
        //  # written starting at the next (in this case 512 byte) sector boundary.
        //  #
        //  set mj_pointer [expr {
        //    20 + [string length "test.db-mjXXXXXX9XX"]
        //  }]
        //  #
        //  #   NOTE: If the current SQLite VFS lacks the concept of a current directory,
        //  #         the length of the current directory name plus 1 character for the
        //  #         directory separator character are NOT counted as part of the total
        //  #         size; otherwise, they are.
        //  #
        //  ifcapable curdir {
        //    set mj_pointer [expr {$mj_pointer + [string length [get_pwd]] + 1}]
        //  }
        //  expr {$::max_journal==(((512+2*(1024+8)+511)/512)*512 + $mj_pointer)}
        //} 1
        //db close
        //tv delete

        //do_test pager1-5.5.1 {
        //  sqlite3 db test.db
        //  execsql { 
        //    ATTACH 'test.db2' AS aux;
        //    PRAGMA journal_mode = PERSIST;
        //    CREATE TABLE t3(a, b);
        //    INSERT INTO t3 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    UPDATE t3 SET b = randomblob(1500);
        //  }
        //  expr [file size test.db-journal] > 15000
        //} {1}
        //do_test pager1-5.5.2 {
        //  execsql {
        //    PRAGMA synchronous = full;
        //    BEGIN;
        //      DELETE FROM t1 WHERE b = 'Stalin';
        //      DELETE FROM t2 WHERE b = 'Stalin';
        //    COMMIT;
        //  }
        //  file size test.db-journal
        //} {0}


        //#-------------------------------------------------------------------------
        //# The following tests work with "PRAGMA max_page_count"
        //#
        //do_test pager1-6.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA auto_vacuum = none;
        //    PRAGMA max_page_count = 10;
        //    CREATE TABLE t2(a, b);
        //    CREATE TABLE t3(a, b);
        //    CREATE TABLE t4(a, b);
        //    CREATE TABLE t5(a, b);
        //    CREATE TABLE t6(a, b);
        //    CREATE TABLE t7(a, b);
        //    CREATE TABLE t8(a, b);
        //    CREATE TABLE t9(a, b);
        //    CREATE TABLE t10(a, b);
        //  }
        //} {10}
        //do_catchsql_test pager1-6.2 {
        //  CREATE TABLE t11(a, b)
        //} {1 {database or disk is full}}
        //do_execsql_test pager1-6.4 { PRAGMA max_page_count      } {10}
        //do_execsql_test pager1-6.5 { PRAGMA max_page_count = 15 } {15}
        //do_execsql_test pager1-6.6 { CREATE TABLE t11(a, b)     } {}
        //do_execsql_test pager1-6.7 {
        //  BEGIN;
        //    INSERT INTO t11 VALUES(1, 2);
        //    PRAGMA max_page_count = 13;
        //} {13}
        //do_execsql_test pager1-6.8 {
        //    INSERT INTO t11 VALUES(3, 4);
        //    PRAGMA max_page_count = 10;
        //} {11}
        //do_execsql_test pager1-6.9 { COMMIT } {}

        //do_execsql_test pager1-6.10 { PRAGMA max_page_count = 10 } {11}
        //do_execsql_test pager1-6.11 { SELECT * FROM t11 }          {1 2 3 4}
        //do_execsql_test pager1-6.12 { PRAGMA max_page_count }      {11}


        //#-------------------------------------------------------------------------
        //# The following tests work with "PRAGMA journal_mode=TRUNCATE" and
        //# "PRAGMA locking_mode=EXCLUSIVE".
        //#
        //# Each test is specified with 5 variables. As follows:
        //#
        //#   $tn:  Test Number. Used as part of the [do_test] test names.
        //#   $sql: SQL to execute.
        //#   $res: Expected result of executing $sql.
        //#   $js:  The expected size of the journal file, in bytes, after executing
        //#         the SQL script. Or -1 if the journal is not expected to exist.
        //#   $ws:  The expected size of the WAL file, in bytes, after executing
        //#         the SQL script. Or -1 if the WAL is not expected to exist.
        //#
        //ifcapable wal {
        //  faultsim_delete_and_reopen
        //  foreach {tn sql res js ws} [subst {

        //    1  {
        //      CREATE TABLE t1(a, b);
        //      PRAGMA auto_vacuum=OFF;
        //      PRAGMA synchronous=NORMAL;
        //      PRAGMA page_size=1024;
        //      PRAGMA locking_mode=EXCLUSIVE;
        //      PRAGMA journal_mode=TRUNCATE;
        //      INSERT INTO t1 VALUES(1, 2);
        //    } {exclusive truncate} 0 -1

        //    2  {
        //      BEGIN IMMEDIATE;
        //        SELECT * FROM t1;
        //      COMMIT;
        //    } {1 2} 0 -1

        //    3  {
        //      BEGIN;
        //        SELECT * FROM t1;
        //      COMMIT;
        //    } {1 2} 0 -1

        //    4  { PRAGMA journal_mode = WAL }    wal       -1 -1
        //    5  { INSERT INTO t1 VALUES(3, 4) }  {}        -1 [wal_file_size 1 1024]
        //    6  { PRAGMA locking_mode = NORMAL } exclusive -1 [wal_file_size 1 1024]
        //    7  { INSERT INTO t1 VALUES(5, 6); } {}        -1 [wal_file_size 2 1024]

        //    8  { PRAGMA journal_mode = TRUNCATE } truncate          0 -1
        //    9  { INSERT INTO t1 VALUES(7, 8) }    {}                0 -1
        //    10 { SELECT * FROM t1 }               {1 2 3 4 5 6 7 8} 0 -1

        //  }] {
        //    do_execsql_test pager1-7.1.$tn.1 $sql $res
        //    catch { set J -1 ; set J [file size test.db-journal] }
        //    catch { set W -1 ; set W [file size test.db-wal] }
        //    do_test pager1-7.1.$tn.2 { list $J $W } [list $js $ws]
        //  }
        //}

        //do_test pager1-7.2.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA locking_mode = EXCLUSIVE;
        //    CREATE TABLE t1(a, b);
        //    BEGIN;
        //      PRAGMA journal_mode = delete;
        //      PRAGMA journal_mode = truncate;
        //  }
        //} {exclusive delete truncate}
        //do_test pager1-7.2.2 {
        //  execsql { INSERT INTO t1 VALUES(1, 2) }
        //  execsql { PRAGMA journal_mode = persist }
        //} {truncate}
        //do_test pager1-7.2.3 {
        //  execsql { COMMIT }
        //  execsql {
        //    PRAGMA journal_mode = persist;
        //    PRAGMA journal_size_limit;
        //  }
        //} {persist -1}

        //#-------------------------------------------------------------------------
        //# The following tests, pager1-8.*, test that the special filenames 
        //# ":memory:" and "" open temporary databases.
        //#
        //foreach {tn filename} {
        //  1 :memory:
        //  2 ""
        //} {
        //  do_test pager1-8.$tn.1 {
        //    faultsim_delete_and_reopen
        //    db close
        //    sqlite3 db $filename
        //    execsql {
        //      PRAGMA auto_vacuum = 1;
        //      CREATE TABLE x1(x);
        //      INSERT INTO x1 VALUES('Charles');
        //      INSERT INTO x1 VALUES('James');
        //      INSERT INTO x1 VALUES('Mary');
        //      SELECT * FROM x1;
        //    }
        //  } {Charles James Mary}

        //  do_test pager1-8.$tn.2 {
        //    sqlite3 db2 $filename
        //    catchsql { SELECT * FROM x1 } db2
        //  } {1 {no such table: x1}}

        //  do_execsql_test pager1-8.$tn.3 {
        //    BEGIN;
        //      INSERT INTO x1 VALUES('William');
        //      INSERT INTO x1 VALUES('Anne');
        //    ROLLBACK;
        //  } {}
        //}

        //#-------------------------------------------------------------------------
        //# The next block of tests - pager1-9.* - deal with interactions between
        //# the pager and the backup API. Test cases:
        //#
        //#   pager1-9.1.*: Test that a backup completes successfully even if the
        //#                 source db is written to during the backup op.
        //#
        //#   pager1-9.2.*: Test that a backup completes successfully even if the
        //#                 source db is written to and then rolled back during a 
        //#                 backup operation.
        //#
        //do_test pager1-9.0.1 {
        //  faultsim_delete_and_reopen
        //  db func a_string a_string
        //  execsql {
        //    PRAGMA cache_size = 10;
        //    BEGIN;
        //      CREATE TABLE ab(a, b, UNIQUE(a, b));
        //      INSERT INTO ab VALUES( a_string(200), a_string(300) );
        //      INSERT INTO ab SELECT a_string(200), a_string(300) FROM ab;
        //      INSERT INTO ab SELECT a_string(200), a_string(300) FROM ab;
        //      INSERT INTO ab SELECT a_string(200), a_string(300) FROM ab;
        //      INSERT INTO ab SELECT a_string(200), a_string(300) FROM ab;
        //      INSERT INTO ab SELECT a_string(200), a_string(300) FROM ab;
        //      INSERT INTO ab SELECT a_string(200), a_string(300) FROM ab;
        //      INSERT INTO ab SELECT a_string(200), a_string(300) FROM ab;
        //    COMMIT;
        //  }
        //} {}
        //do_test pager1-9.0.2 {
        //  sqlite3 db2 test.db2
        //  db2 eval { PRAGMA cache_size = 10 }
        //  sqlite3_backup B db2 main db main
        //  list [B step 10000] [B finish]
        //} {SQLITE_DONE SQLITE_OK}
        //do_test pager1-9.0.3 {
        // db one {SELECT md5sum(a, b) FROM ab}
        //} [db2 one {SELECT md5sum(a, b) FROM ab}]

        //do_test pager1-9.1.1 {
        //  execsql { UPDATE ab SET a = a_string(201) }
        //  sqlite3_backup B db2 main db main
        //  B step 30
        //} {SQLITE_OK}
        //do_test pager1-9.1.2 {
        //  execsql { UPDATE ab SET b = a_string(301) }
        //  list [B step 10000] [B finish]
        //} {SQLITE_DONE SQLITE_OK}
        //do_test pager1-9.1.3 {
        // db one {SELECT md5sum(a, b) FROM ab}
        //} [db2 one {SELECT md5sum(a, b) FROM ab}]
        //do_test pager1-9.1.4 { execsql { SELECT count(*) FROM ab } } {128}

        //do_test pager1-9.2.1 {
        //  execsql { UPDATE ab SET a = a_string(202) }
        //  sqlite3_backup B db2 main db main
        //  B step 30
        //} {SQLITE_OK}
        //do_test pager1-9.2.2 {
        //  execsql { 
        //    BEGIN;
        //      UPDATE ab SET b = a_string(301);
        //    ROLLBACK;
        //  }
        //  list [B step 10000] [B finish]
        //} {SQLITE_DONE SQLITE_OK}
        //do_test pager1-9.2.3 {
        // db one {SELECT md5sum(a, b) FROM ab}
        //} [db2 one {SELECT md5sum(a, b) FROM ab}]
        //do_test pager1-9.2.4 { execsql { SELECT count(*) FROM ab } } {128}
        //db close
        //db2 close

        //do_test pager1-9.3.1 {
        //  testvfs tv -default 1
        //  tv sectorsize 4096
        //  faultsim_delete_and_reopen

        //  execsql { PRAGMA page_size = 1024 }
        //  for {set ii 0} {$ii < 4} {incr ii} { execsql "CREATE TABLE t${ii}(a, b)" }
        //} {}
        //do_test pager1-9.3.2 {
        //  sqlite3 db2 test.db2

        //  execsql {
        //    PRAGMA page_size = 4096;
        //    PRAGMA synchronous = OFF;
        //    CREATE TABLE t1(a, b);
        //    CREATE TABLE t2(a, b);
        //  } db2

        //  sqlite3_backup B db2 main db main
        //  B step 30
        //  list [B step 10000] [B finish]
        //} {SQLITE_DONE SQLITE_OK}
        //do_test pager1-9.3.3 {
        //  db2 close
        //  db close
        //  tv delete
        //  file size test.db2
        //} [file size test.db]

        //do_test pager1-9.4.1 {
        //  faultsim_delete_and_reopen
        //  sqlite3 db2 test.db2
        //  execsql {
        //    PRAGMA page_size = 4096;
        //    CREATE TABLE t1(a, b);
        //    CREATE TABLE t2(a, b);
        //  } db2
        //  sqlite3_backup B db2 main db main
        //  list [B step 10000] [B finish]
        //} {SQLITE_DONE SQLITE_OK}
        //do_test pager1-9.4.2 {
        //  list [file size test.db2] [file size test.db]
        //} {1024 0}
        //db2 close

        //#-------------------------------------------------------------------------
        //# Test that regardless of the value returned by xSectorSize(), the
        //# minimum effective sector-size is 512 and the maximum 65536 bytes.
        //#
        //testvfs tv -default 1
        //foreach sectorsize {
        //    16
        //    32   64   128   256   512   1024   2048 
        //    4096 8192 16384 32768 65536 131072 262144
        //} {
        //  tv sectorsize $sectorsize
        //  tv devchar {}
        //  set eff $sectorsize
        //  if {$sectorsize < 512}   { set eff 512 }
        //  if {$sectorsize > 65536} { set eff 65536 }

        //  do_test pager1-10.$sectorsize.1 {
        //    faultsim_delete_and_reopen
        //    db func a_string a_string
        //    execsql {
        //      PRAGMA journal_mode = PERSIST;
        //      PRAGMA page_size = 1024;
        //      BEGIN;
        //        CREATE TABLE t1(a, b);
        //        CREATE TABLE t2(a, b);
        //        CREATE TABLE t3(a, b);
        //      COMMIT;
        //    }
        //    file size test.db-journal
        //  } [expr $sectorsize > 65536 ? 65536 : ($sectorsize<32 ? 512 : $sectorsize)]

        //  do_test pager1-10.$sectorsize.2 {
        //    execsql { 
        //      INSERT INTO t3 VALUES(a_string(300), a_string(300));
        //      INSERT INTO t3 SELECT * FROM t3;        /*  2 */
        //      INSERT INTO t3 SELECT * FROM t3;        /*  4 */
        //      INSERT INTO t3 SELECT * FROM t3;        /*  8 */
        //      INSERT INTO t3 SELECT * FROM t3;        /* 16 */
        //      INSERT INTO t3 SELECT * FROM t3;        /* 32 */
        //    }
        //  } {}

        //  do_test pager1-10.$sectorsize.3 {
        //    db close
        //    sqlite3 db test.db
        //    execsql { 
        //      PRAGMA cache_size = 10;
        //      BEGIN;
        //    }
        //    recursive_select 32 t3 {db eval "INSERT INTO t2 VALUES(1, 2)"}
        //    execsql {
        //      COMMIT;
        //      SELECT * FROM t2;
        //    }
        //  } {1 2}

        //  do_test pager1-10.$sectorsize.4 {
        //    execsql {
        //      CREATE TABLE t6(a, b);
        //      CREATE TABLE t7(a, b);
        //      CREATE TABLE t5(a, b);
        //      DROP TABLE t6;
        //      DROP TABLE t7;
        //    }
        //    execsql {
        //      BEGIN;
        //        CREATE TABLE t6(a, b);
        //    }
        //    recursive_select 32 t3 {db eval "INSERT INTO t5 VALUES(1, 2)"}
        //    execsql {
        //      COMMIT;
        //      SELECT * FROM t5;
        //    }
        //  } {1 2}

        //}
        //db close

        //tv sectorsize 4096
        //do_test pager1.10.x.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA auto_vacuum = none;
        //    PRAGMA page_size = 1024;
        //    CREATE TABLE t1(x);
        //  }
        //  for {set i 0} {$i<30} {incr i} {
        //    execsql { INSERT INTO t1 VALUES(zeroblob(900)) }
        //  }
        //  file size test.db
        //} {32768}
        //do_test pager1.10.x.2 {
        //  execsql {
        //    CREATE TABLE t2(x);
        //    DROP TABLE t2;
        //  }
        //  file size test.db
        //} {33792}
        //do_test pager1.10.x.3 {
        //  execsql {
        //    BEGIN;
        //    CREATE TABLE t2(x);
        //  }
        //  recursive_select 30 t1
        //  execsql {
        //    CREATE TABLE t3(x);
        //    COMMIT;
        //  }
        //} {}

        //db close
        //tv delete

        //testvfs tv -default 1
        //faultsim_delete_and_reopen
        //db func a_string a_string
        //do_execsql_test pager1-11.1 {
        //  PRAGMA journal_mode = DELETE;
        //  PRAGMA cache_size = 10;
        //  BEGIN;
        //    CREATE TABLE zz(top PRIMARY KEY);
        //    INSERT INTO zz VALUES(a_string(222));
        //    INSERT INTO zz SELECT a_string((SELECT 222+max(rowid) FROM zz)) FROM zz;
        //    INSERT INTO zz SELECT a_string((SELECT 222+max(rowid) FROM zz)) FROM zz;
        //    INSERT INTO zz SELECT a_string((SELECT 222+max(rowid) FROM zz)) FROM zz;
        //    INSERT INTO zz SELECT a_string((SELECT 222+max(rowid) FROM zz)) FROM zz;
        //    INSERT INTO zz SELECT a_string((SELECT 222+max(rowid) FROM zz)) FROM zz;
        //  COMMIT;
        //  BEGIN;
        //    UPDATE zz SET top = a_string(345);
        //} {delete}

        //proc lockout {method args} { return SQLITE_IOERR }
        //tv script lockout
        //tv filter {xWrite xTruncate xSync}
        //do_catchsql_test pager1-11.2 { COMMIT } {1 {disk I/O error}}

        //tv script {}
        //do_test pager1-11.3 {
        //  sqlite3 db2 test.db
        //  execsql {
        //    PRAGMA journal_mode = TRUNCATE;
        //    PRAGMA integrity_check;
        //  } db2
        //} {truncate ok}
        //do_test pager1-11.4 {
        //  db2 close
        //  file exists test.db-journal
        //} {0}
        //do_execsql_test pager1-11.5 { SELECT count(*) FROM zz } {32}
        //db close
        //tv delete

        //#-------------------------------------------------------------------------
        //# Test "PRAGMA page_size"
        //#
        //testvfs tv -default 1
        //tv sectorsize 1024
        //foreach pagesize {
        //    512   1024   2048 4096 8192 16384 32768 
        //} {
        //  faultsim_delete_and_reopen

        //  # The sector-size (according to the VFS) is 1024 bytes. So if the
        //  # page-size requested using "PRAGMA page_size" is greater than the
        //  # compile time value of SQLITE_MAX_PAGE_SIZE, then the effective 
        //  # page-size remains 1024 bytes.
        //  #
        //  set eff $pagesize
        //  if {$eff > $::SQLITE_MAX_PAGE_SIZE} { set eff 1024 }

        //  do_test pager1-12.$pagesize.1 {
        //    sqlite3 db2 test.db
        //    execsql "
        //      PRAGMA page_size = $pagesize;
        //      CREATE VIEW v AS SELECT * FROM sqlite_master;
        //    " db2
        //    file size test.db
        //  } $eff
        //  do_test pager1-12.$pagesize.2 {
        //    sqlite3 db2 test.db
        //    execsql { 
        //      SELECT count(*) FROM v;
        //      PRAGMA main.page_size;
        //    } db2
        //  } [list 1 $eff]
        //  do_test pager1-12.$pagesize.3 {
        //    execsql { 
        //      SELECT count(*) FROM v;
        //      PRAGMA main.page_size;
        //    }
        //  } [list 1 $eff]
        //  db2 close
        //}
        //db close
        //tv delete

        //#-------------------------------------------------------------------------
        //# Test specal "PRAGMA journal_mode=PERSIST" test cases.
        //#
        //# pager1-13.1.*: This tests a special case encountered in persistent 
        //#                journal mode: If the journal associated with a transaction
        //#                is smaller than the journal file (because a previous 
        //#                transaction left a very large non-hot journal file in the
        //#                file-system), then SQLite has to be careful that there is
        //#                not a journal-header left over from a previous transaction
        //#                immediately following the journal content just written.
        //#                If there is, and the process crashes so that the journal
        //#                becomes a hot-journal and must be rolled back by another
        //#                process, there is a danger that the other process may roll
        //#                back the aborted transaction, then continue copying data
        //#                from an older transaction from the remainder of the journal.
        //#                See the syncJournal() function for details.
        //#
        //# pager1-13.2.*: Same test as the previous. This time, throw an index into
        //#                the mix to make the integrity-check more likely to catch
        //#                errors.
        //#
        //testvfs tv -default 1
        //tv script xSyncCb
        //tv filter xSync
        //proc xSyncCb {method filename args} {
        //  set t [file tail $filename]
        //  if {$t == "test.db"} faultsim_save
        //  return SQLITE_OK
        //}
        //faultsim_delete_and_reopen
        //db func a_string a_string

        //# The UPDATE statement at the end of this test case creates a really big
        //# journal. Since the cache-size is only 10 pages, the journal contains 
        //# frequent journal headers.
        //#
        //do_execsql_test pager1-13.1.1 {
        //  PRAGMA page_size = 1024;
        //  PRAGMA journal_mode = PERSIST;
        //  PRAGMA cache_size = 10;
        //  BEGIN;
        //    CREATE TABLE t1(a INTEGER PRIMARY KEY, b BLOB);
        //    INSERT INTO t1 VALUES(NULL, a_string(400));
        //    INSERT INTO t1 SELECT NULL, a_string(400) FROM t1;          /*   2 */
        //    INSERT INTO t1 SELECT NULL, a_string(400) FROM t1;          /*   4 */
        //    INSERT INTO t1 SELECT NULL, a_string(400) FROM t1;          /*   8 */
        //    INSERT INTO t1 SELECT NULL, a_string(400) FROM t1;          /*  16 */
        //    INSERT INTO t1 SELECT NULL, a_string(400) FROM t1;          /*  32 */
        //    INSERT INTO t1 SELECT NULL, a_string(400) FROM t1;          /*  64 */
        //    INSERT INTO t1 SELECT NULL, a_string(400) FROM t1;          /* 128 */
        //  COMMIT;
        //  UPDATE t1 SET b = a_string(400);
        //} {persist}

        //if {$::tcl_platform(platform)!="windows"} {
        //# Run transactions of increasing sizes. Eventually, one (or more than one)
        //# of these will write just enough content that one of the old headers created 
        //# by the transaction in the block above lies immediately after the content
        //# journalled by the current transaction.
        //#
        //for {set nUp 1} {$nUp<64} {incr nUp} {
        //  do_execsql_test pager1-13.1.2.$nUp.1 { 
        //    UPDATE t1 SET b = a_string(399) WHERE a <= $nUp
        //  } {}
        //  do_execsql_test pager1-13.1.2.$nUp.2 { PRAGMA integrity_check } {ok} 

        //  # Try to access the snapshot of the file-system.
        //  #
        //  sqlite3 db2 sv_test.db
        //  do_test pager1-13.1.2.$nUp.3 {
        //    execsql { SELECT sum(length(b)) FROM t1 } db2
        //  } [expr {128*400 - ($nUp-1)}]
        //  do_test pager1-13.1.2.$nUp.4 {
        //    execsql { PRAGMA integrity_check } db2
        //  } {ok}
        //  db2 close
        //}
        //}

        //if {$::tcl_platform(platform)!="windows"} {
        //# Same test as above. But this time with an index on the table.
        //#
        //do_execsql_test pager1-13.2.1 {
        //  CREATE INDEX i1 ON t1(b);
        //  UPDATE t1 SET b = a_string(400);
        //} {}
        //for {set nUp 1} {$nUp<64} {incr nUp} {
        //  do_execsql_test pager1-13.2.2.$nUp.1 { 
        //    UPDATE t1 SET b = a_string(399) WHERE a <= $nUp
        //  } {}
        //  do_execsql_test pager1-13.2.2.$nUp.2 { PRAGMA integrity_check } {ok} 
        //  sqlite3 db2 sv_test.db
        //  do_test pager1-13.2.2.$nUp.3 {
        //    execsql { SELECT sum(length(b)) FROM t1 } db2
        //  } [expr {128*400 - ($nUp-1)}]
        //  do_test pager1-13.2.2.$nUp.4 {
        //    execsql { PRAGMA integrity_check } db2
        //  } {ok}
        //  db2 close
        //}
        //}

        //db close
        //tv delete

        //#-------------------------------------------------------------------------
        //# Test specal "PRAGMA journal_mode=OFF" test cases.
        //#
        //faultsim_delete_and_reopen
        //do_execsql_test pager1-14.1.1 {
        //  PRAGMA journal_mode = OFF;
        //  CREATE TABLE t1(a, b);
        //  BEGIN;
        //    INSERT INTO t1 VALUES(1, 2);
        //  COMMIT;
        //  SELECT * FROM t1;
        //} {off 1 2}
        //do_catchsql_test pager1-14.1.2 {
        //  BEGIN;
        //    INSERT INTO t1 VALUES(3, 4);
        //  ROLLBACK;
        //} {0 {}}
        //do_execsql_test pager1-14.1.3 {
        //  SELECT * FROM t1;
        //} {1 2}
        //do_catchsql_test pager1-14.1.4 {
        //  BEGIN;
        //    INSERT INTO t1(rowid, a, b) SELECT a+3, b, b FROM t1;
        //    INSERT INTO t1(rowid, a, b) SELECT a+3, b, b FROM t1;
        //} {1 {PRIMARY KEY must be unique}}
        //do_execsql_test pager1-14.1.5 {
        //  COMMIT;
        //  SELECT * FROM t1;
        //} {1 2 2 2}

        //#-------------------------------------------------------------------------
        //# Test opening and closing the pager sub-system with different values
        //# for the sqlite3_vfs.szOsFile variable.
        //#
        //faultsim_delete_and_reopen
        //do_execsql_test pager1-15.0 {
        //  CREATE TABLE tx(y, z);
        //  INSERT INTO tx VALUES('Ayutthaya', 'Beijing');
        //  INSERT INTO tx VALUES('London', 'Tokyo');
        //} {}
        //db close
        //for {set i 0} {$i<513} {incr i 3} {
        //  testvfs tv -default 1 -szosfile $i
        //  sqlite3 db test.db
        //  do_execsql_test pager1-15.$i.1 {
        //    SELECT * FROM tx;
        //  } {Ayutthaya Beijing London Tokyo}
        //  db close
        //  tv delete
        //}

        //#-------------------------------------------------------------------------
        //# Check that it is not possible to open a database file if the full path
        //# to the associated journal file will be longer than sqlite3_vfs.mxPathname.
        //#
        //testvfs tv -default 1
        //tv script xOpenCb
        //tv filter xOpen
        //proc xOpenCb {method filename args} {
        //  set ::file_len [string length $filename]
        //}
        //sqlite3 db test.db
        //db close
        //tv delete

        //for {set ii [expr $::file_len-5]} {$ii < [expr $::file_len+20]} {incr ii} {
        //  testvfs tv -default 1 -mxpathname $ii

        //  # The length of the full path to file "test.db-journal" is ($::file_len+8).
        //  # If the configured sqlite3_vfs.mxPathname value greater than or equal to
        //  # this, then the file can be opened. Otherwise, it cannot.
        //  #
        //  if {$ii >= [expr $::file_len+8]} {
        //    set res {0 {}}
        //  } else {
        //    set res {1 {unable to open database file}}
        //  }

        //  do_test pager1-16.1.$ii {
        //    list [catch { sqlite3 db test.db } msg] $msg
        //  } $res

        //  catch {db close}
        //  tv delete
        //}


        //#-------------------------------------------------------------------------
        //# Test the pagers response to the b-tree layer requesting illegal page 
        //# numbers:
        //#
        //#   + The locking page,
        //#   + Page 0,
        //#   + A page with a page number greater than (2^31-1).
        //#
        //# These tests will not work if SQLITE_DIRECT_OVERFLOW_READ is defined. In
        //# that case IO errors are sometimes reported instead of SQLITE_CORRUPT.
        //#
        //ifcapable !direct_read {
        //do_test pager1-18.1 {
        //  faultsim_delete_and_reopen
        //  db func a_string a_string
        //  execsql { 
        //    PRAGMA page_size = 1024;
        //    CREATE TABLE t1(a, b);
        //    INSERT INTO t1 VALUES(a_string(500), a_string(200));
        //    INSERT INTO t1 SELECT a_string(500), a_string(200) FROM t1;
        //    INSERT INTO t1 SELECT a_string(500), a_string(200) FROM t1;
        //    INSERT INTO t1 SELECT a_string(500), a_string(200) FROM t1;
        //    INSERT INTO t1 SELECT a_string(500), a_string(200) FROM t1;
        //    INSERT INTO t1 SELECT a_string(500), a_string(200) FROM t1;
        //    INSERT INTO t1 SELECT a_string(500), a_string(200) FROM t1;
        //    INSERT INTO t1 SELECT a_string(500), a_string(200) FROM t1;
        //  }
        //} {}
        //do_test pager1-18.2 {
        //  set root [db one "SELECT rootpage FROM sqlite_master"]
        //  set lockingpage [expr (0x10000/1024) + 1]
        //  execsql {
        //    PRAGMA writable_schema = 1;
        //    UPDATE sqlite_master SET rootpage = $lockingpage;
        //  }
        //  sqlite3 db2 test.db
        //  catchsql { SELECT count(*) FROM t1 } db2
        //} {1 {database disk image is malformed}}
        //db2 close
        //do_test pager1-18.3.1 {
        //  execsql {
        //    CREATE TABLE t2(x);
        //    INSERT INTO t2 VALUES(a_string(5000));
        //  }
        //  set pgno [expr ([file size test.db] / 1024)-2]
        //  hexio_write test.db [expr ($pgno-1)*1024] 00000000
        //  sqlite3 db2 test.db
        //  # even though x is malformed, because typeof() does
        //  # not load the content of x, the error is not noticed.
        //  catchsql { SELECT typeof(x) FROM t2 } db2
        //} {0 text}
        //do_test pager1-18.3.2 {
        //  # in this case, the value of x is loaded and so the error is
        //  # detected
        //  catchsql { SELECT length(x||'') FROM t2 } db2
        //} {1 {database disk image is malformed}}
        //db2 close
        //do_test pager1-18.3.3 {
        //  execsql {
        //    DELETE FROM t2;
        //    INSERT INTO t2 VALUES(randomblob(5000));
        //  }
        //  set pgno [expr ([file size test.db] / 1024)-2]
        //  hexio_write test.db [expr ($pgno-1)*1024] 00000000
        //  sqlite3 db2 test.db
        //  # even though x is malformed, because length() and typeof() do
        //  # not load the content of x, the error is not noticed.
        //  catchsql { SELECT length(x), typeof(x) FROM t2 } db2
        //} {0 {5000 blob}}
        //do_test pager1-18.3.4 {
        //  # in this case, the value of x is loaded and so the error is
        //  # detected
        //  catchsql { SELECT length(x||'') FROM t2 } db2
        //} {1 {database disk image is malformed}}
        //db2 close
        //do_test pager1-18.4 {
        //  hexio_write test.db [expr ($pgno-1)*1024] 90000000
        //  sqlite3 db2 test.db
        //  catchsql { SELECT length(x||'') FROM t2 } db2
        //} {1 {database disk image is malformed}}
        //db2 close
        //do_test pager1-18.5 {
        //  sqlite3 db ""
        //  execsql {
        //    CREATE TABLE t1(a, b);
        //    CREATE TABLE t2(a, b);
        //    PRAGMA writable_schema = 1;
        //    UPDATE sqlite_master SET rootpage=5 WHERE tbl_name = 't1';
        //    PRAGMA writable_schema = 0;
        //    ALTER TABLE t1 RENAME TO x1;
        //  }
        //  catchsql { SELECT * FROM x1 }
        //} {1 {database disk image is malformed}}
        //db close

        //do_test pager1-18.6 {
        //  faultsim_delete_and_reopen
        //  db func a_string a_string
        //  execsql {
        //    PRAGMA page_size = 1024;
        //    CREATE TABLE t1(x);
        //    INSERT INTO t1 VALUES(a_string(800));
        //    INSERT INTO t1 VALUES(a_string(800));
        //  }

        //  set root [db one "SELECT rootpage FROM sqlite_master"]
        //  db close

        //  hexio_write test.db [expr ($root-1)*1024 + 8] 00000000
        //  sqlite3 db test.db
        //  catchsql { SELECT length(x) FROM t1 }
        //} {1 {database disk image is malformed}}
        //}

        //do_test pager1-19.1 {
        //  sqlite3 db ""
        //  db func a_string a_string
        //  execsql {
        //    PRAGMA page_size = 512;
        //    PRAGMA auto_vacuum = 1;
        //    CREATE TABLE t1(aa, ab, ac, ad, ae, af, ag, ah, ai, aj, ak, al, am, an,
        //                    ba, bb, bc, bd, be, bf, bg, bh, bi, bj, bk, bl, bm, bn,
        //                    ca, cb, cc, cd, ce, cf, cg, ch, ci, cj, ck, cl, cm, cn,
        //                    da, db, dc, dd, de, df, dg, dh, di, dj, dk, dl, dm, dn,
        //                    ea, eb, ec, ed, ee, ef, eg, eh, ei, ej, ek, el, em, en,
        //                    fa, fb, fc, fd, fe, ff, fg, fh, fi, fj, fk, fl, fm, fn,
        //                    ga, gb, gc, gd, ge, gf, gg, gh, gi, gj, gk, gl, gm, gn,
        //                    ha, hb, hc, hd, he, hf, hg, hh, hi, hj, hk, hl, hm, hn,
        //                    ia, ib, ic, id, ie, if, ig, ih, ii, ij, ik, il, im, ix,
        //                    ja, jb, jc, jd, je, jf, jg, jh, ji, jj, jk, jl, jm, jn,
        //                    ka, kb, kc, kd, ke, kf, kg, kh, ki, kj, kk, kl, km, kn,
        //                    la, lb, lc, ld, le, lf, lg, lh, li, lj, lk, ll, lm, ln,
        //                    ma, mb, mc, md, me, mf, mg, mh, mi, mj, mk, ml, mm, mn
        //    );
        //    CREATE TABLE t2(aa, ab, ac, ad, ae, af, ag, ah, ai, aj, ak, al, am, an,
        //                    ba, bb, bc, bd, be, bf, bg, bh, bi, bj, bk, bl, bm, bn,
        //                    ca, cb, cc, cd, ce, cf, cg, ch, ci, cj, ck, cl, cm, cn,
        //                    da, db, dc, dd, de, df, dg, dh, di, dj, dk, dl, dm, dn,
        //                    ea, eb, ec, ed, ee, ef, eg, eh, ei, ej, ek, el, em, en,
        //                    fa, fb, fc, fd, fe, ff, fg, fh, fi, fj, fk, fl, fm, fn,
        //                    ga, gb, gc, gd, ge, gf, gg, gh, gi, gj, gk, gl, gm, gn,
        //                    ha, hb, hc, hd, he, hf, hg, hh, hi, hj, hk, hl, hm, hn,
        //                    ia, ib, ic, id, ie, if, ig, ih, ii, ij, ik, il, im, ix,
        //                    ja, jb, jc, jd, je, jf, jg, jh, ji, jj, jk, jl, jm, jn,
        //                    ka, kb, kc, kd, ke, kf, kg, kh, ki, kj, kk, kl, km, kn,
        //                    la, lb, lc, ld, le, lf, lg, lh, li, lj, lk, ll, lm, ln,
        //                    ma, mb, mc, md, me, mf, mg, mh, mi, mj, mk, ml, mm, mn
        //    );
        //    INSERT INTO t1(aa) VALUES( a_string(100000) );
        //    INSERT INTO t2(aa) VALUES( a_string(100000) );
        //    VACUUM;
        //  }
        //} {}

        //#-------------------------------------------------------------------------
        //# Test a couple of special cases that come up while committing 
        //# transactions:
        //#
        //#   pager1-20.1.*: Committing an in-memory database transaction when the 
        //#                  database has not been modified at all.
        //#
        //#   pager1-20.2.*: As above, but with a normal db in exclusive-locking mode.
        //#
        //#   pager1-20.3.*: Committing a transaction in WAL mode where the database has
        //#                  been modified, but all dirty pages have been flushed to 
        //#                  disk before the commit.
        //#
        //do_test pager1-20.1.1 {
        //  catch {db close}
        //  sqlite3 db :memory:
        //  execsql {
        //    CREATE TABLE one(two, three);
        //    INSERT INTO one VALUES('a', 'b');
        //  }
        //} {}
        //do_test pager1-20.1.2 {
        //  execsql {
        //    BEGIN EXCLUSIVE;
        //    COMMIT;
        //  }
        //} {}

        //do_test pager1-20.2.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA locking_mode = exclusive;
        //    PRAGMA journal_mode = persist;
        //    CREATE TABLE one(two, three);
        //    INSERT INTO one VALUES('a', 'b');
        //  }
        //} {exclusive persist}
        //do_test pager1-20.2.2 {
        //  execsql {
        //    BEGIN EXCLUSIVE;
        //    COMMIT;
        //  }
        //} {}

        //ifcapable wal {
        //  do_test pager1-20.3.1 {
        //    faultsim_delete_and_reopen
        //    db func a_string a_string
        //    execsql {
        //      PRAGMA cache_size = 10;
        //      PRAGMA journal_mode = wal;
        //      BEGIN;
        //        CREATE TABLE t1(x);
        //        CREATE TABLE t2(y);
        //        INSERT INTO t1 VALUES(a_string(800));
        //        INSERT INTO t1 SELECT a_string(800) FROM t1;         /*   2 */
        //        INSERT INTO t1 SELECT a_string(800) FROM t1;         /*   4 */
        //        INSERT INTO t1 SELECT a_string(800) FROM t1;         /*   8 */
        //        INSERT INTO t1 SELECT a_string(800) FROM t1;         /*  16 */
        //        INSERT INTO t1 SELECT a_string(800) FROM t1;         /*  32 */
        //      COMMIT;
        //    }
        //  } {wal}
        //  do_test pager1-20.3.2 {
        //    execsql {
        //      BEGIN;
        //      INSERT INTO t2 VALUES('xxxx');
        //    }
        //    recursive_select 32 t1
        //    execsql COMMIT
        //  } {}
        //}

        //#-------------------------------------------------------------------------
        //# Test that a WAL database may not be opened if:
        //#
        //#   pager1-21.1.*: The VFS has an iVersion less than 2, or
        //#   pager1-21.2.*: The VFS does not provide xShmXXX() methods.
        //#
        //ifcapable wal {
        //  do_test pager1-21.0 {
        //    faultsim_delete_and_reopen
        //    execsql {
        //      PRAGMA journal_mode = WAL;
        //      CREATE TABLE ko(c DEFAULT 'abc', b DEFAULT 'def');
        //      INSERT INTO ko DEFAULT VALUES;
        //    }
        //  } {wal}
        //  do_test pager1-21.1 {
        //    testvfs tv -noshm 1
        //    sqlite3 db2 test.db -vfs tv
        //    catchsql { SELECT * FROM ko } db2
        //  } {1 {unable to open database file}}
        //  db2 close
        //  tv delete
        //  do_test pager1-21.2 {
        //    testvfs tv -iversion 1
        //    sqlite3 db2 test.db -vfs tv
        //    catchsql { SELECT * FROM ko } db2
        //  } {1 {unable to open database file}}
        //  db2 close
        //  tv delete
        //}

        //#-------------------------------------------------------------------------
        //# Test that a "PRAGMA wal_checkpoint":
        //#
        //#   pager1-22.1.*: is a no-op on a non-WAL db, and
        //#   pager1-22.2.*: does not cause xSync calls with a synchronous=off db.
        //#
        //ifcapable wal {
        //  do_test pager1-22.1.1 {
        //    faultsim_delete_and_reopen
        //    execsql {
        //      CREATE TABLE ko(c DEFAULT 'abc', b DEFAULT 'def');
        //      INSERT INTO ko DEFAULT VALUES;
        //    }
        //    execsql { PRAGMA wal_checkpoint }
        //  } {0 -1 -1}
        //  do_test pager1-22.2.1 {
        //    testvfs tv -default 1
        //    tv filter xSync
        //    tv script xSyncCb
        //    proc xSyncCb {args} {incr ::synccount}
        //    set ::synccount 0
        //    sqlite3 db test.db
        //    execsql {
        //      PRAGMA synchronous = off;
        //      PRAGMA journal_mode = WAL;
        //      INSERT INTO ko DEFAULT VALUES;
        //    }
        //    execsql { PRAGMA wal_checkpoint }
        //    set synccount
        //  } {0}
        //  db close
        //  tv delete
        //}

        //#-------------------------------------------------------------------------
        //# Tests for changing journal mode.
        //#
        //#   pager1-23.1.*: Test that when changing from PERSIST to DELETE mode,
        //#                  the journal file is deleted.
        //#
        //#   pager1-23.2.*: Same test as above, but while a shared lock is held
        //#                  on the database file.
        //#
        //#   pager1-23.3.*: Same test as above, but while a reserved lock is held
        //#                  on the database file.
        //#
        //#   pager1-23.4.*: And, for fun, while holding an exclusive lock.
        //#
        //#   pager1-23.5.*: Try to set various different journal modes with an
        //#                  in-memory database (only MEMORY and OFF should work).
        //#
        //#   pager1-23.6.*: Try to set locking_mode=normal on an in-memory database
        //#                  (doesn't work - in-memory databases always use
        //#                  locking_mode=exclusive).
        //#
        //do_test pager1-23.1.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA journal_mode = PERSIST;
        //    CREATE TABLE t1(a, b);
        //  }
        //  file exists test.db-journal
        //} {1}
        //do_test pager1-23.1.2 {
        //  execsql { PRAGMA journal_mode = DELETE }
        //  file exists test.db-journal
        //} {0}

        //do_test pager1-23.2.1 {
        //  execsql {
        //    PRAGMA journal_mode = PERSIST;
        //    INSERT INTO t1 VALUES('Canberra', 'ACT');
        //  }
        //  db eval { SELECT * FROM t1 } {
        //    db eval { PRAGMA journal_mode = DELETE }
        //  }
        //  execsql { PRAGMA journal_mode }
        //} {delete}
        //do_test pager1-23.2.2 {
        //  file exists test.db-journal
        //} {0}

        //do_test pager1-23.3.1 {
        //  execsql {
        //    PRAGMA journal_mode = PERSIST;
        //    INSERT INTO t1 VALUES('Darwin', 'NT');
        //    BEGIN IMMEDIATE;
        //  }
        //  db eval { PRAGMA journal_mode = DELETE }
        //  execsql { PRAGMA journal_mode }
        //} {delete}
        //do_test pager1-23.3.2 {
        //  file exists test.db-journal
        //} {0}
        //do_test pager1-23.3.3 {
        //  execsql COMMIT
        //} {}

        //do_test pager1-23.4.1 {
        //  execsql {
        //    PRAGMA journal_mode = PERSIST;
        //    INSERT INTO t1 VALUES('Adelaide', 'SA');
        //    BEGIN EXCLUSIVE;
        //  }
        //  db eval { PRAGMA journal_mode = DELETE }
        //  execsql { PRAGMA journal_mode }
        //} {delete}
        //do_test pager1-23.4.2 {
        //  file exists test.db-journal
        //} {0}
        //do_test pager1-23.4.3 {
        //  execsql COMMIT
        //} {}

        //do_test pager1-23.5.1 {
        //  faultsim_delete_and_reopen
        //  sqlite3 db :memory:
        //} {}
        //foreach {tn mode possible} {
        //  2  off      1
        //  3  memory   1
        //  4  persist  0
        //  5  delete   0
        //  6  wal      0
        //  7  truncate 0
        //} {
        //  do_test pager1-23.5.$tn.1 {
        //    execsql "PRAGMA journal_mode = off"
        //    execsql "PRAGMA journal_mode = $mode"
        //  } [if $possible {list $mode} {list off}]
        //  do_test pager1-23.5.$tn.2 {
        //    execsql "PRAGMA journal_mode = memory"
        //    execsql "PRAGMA journal_mode = $mode"
        //  } [if $possible {list $mode} {list memory}]
        //}
        //do_test pager1-23.6.1 {
        //  execsql {PRAGMA locking_mode = normal}
        //} {exclusive}
        //do_test pager1-23.6.2 {
        //  execsql {PRAGMA locking_mode = exclusive}
        //} {exclusive}
        //do_test pager1-23.6.3 {
        //  execsql {PRAGMA locking_mode}
        //} {exclusive}
        //do_test pager1-23.6.4 {
        //  execsql {PRAGMA main.locking_mode}
        //} {exclusive}

        //#-------------------------------------------------------------------------
        //#
        //do_test pager1-24.1.1 {
        //  faultsim_delete_and_reopen
        //  db func a_string a_string
        //  execsql {
        //    PRAGMA cache_size = 10;
        //    PRAGMA auto_vacuum = FULL;
        //    CREATE TABLE x1(x, y, z, PRIMARY KEY(y, z));
        //    CREATE TABLE x2(x, y, z, PRIMARY KEY(y, z));
        //    INSERT INTO x2 VALUES(a_string(400), a_string(500), a_string(600));
        //    INSERT INTO x2 SELECT a_string(600), a_string(400), a_string(500) FROM x2;
        //    INSERT INTO x2 SELECT a_string(500), a_string(600), a_string(400) FROM x2;
        //    INSERT INTO x2 SELECT a_string(400), a_string(500), a_string(600) FROM x2;
        //    INSERT INTO x2 SELECT a_string(600), a_string(400), a_string(500) FROM x2;
        //    INSERT INTO x2 SELECT a_string(500), a_string(600), a_string(400) FROM x2;
        //    INSERT INTO x2 SELECT a_string(400), a_string(500), a_string(600) FROM x2;
        //    INSERT INTO x1 SELECT * FROM x2;
        //  }
        //} {}
        //do_test pager1-24.1.2 {
        //  execsql {
        //    BEGIN;
        //      DELETE FROM x1 WHERE rowid<32;
        //  }
        //  recursive_select 64 x2
        //} {}
        //do_test pager1-24.1.3 {
        //  execsql { 
        //      UPDATE x1 SET z = a_string(300) WHERE rowid>40;
        //    COMMIT;
        //    PRAGMA integrity_check;
        //    SELECT count(*) FROM x1;
        //  }
        //} {ok 33}

        //do_test pager1-24.1.4 {
        //  execsql {
        //    DELETE FROM x1;
        //    INSERT INTO x1 SELECT * FROM x2;
        //    BEGIN;
        //      DELETE FROM x1 WHERE rowid<32;
        //      UPDATE x1 SET z = a_string(299) WHERE rowid>40;
        //  }
        //  recursive_select 64 x2 {db eval COMMIT}
        //  execsql {
        //    PRAGMA integrity_check;
        //    SELECT count(*) FROM x1;
        //  }
        //} {ok 33}

        //do_test pager1-24.1.5 {
        //  execsql {
        //    DELETE FROM x1;
        //    INSERT INTO x1 SELECT * FROM x2;
        //  }
        //  recursive_select 64 x2 { db eval {CREATE TABLE x3(x, y, z)} }
        //  execsql { SELECT * FROM x3 }
        //} {}

        //#-------------------------------------------------------------------------
        //#
        //do_test pager1-25-1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    BEGIN;
        //      SAVEPOINT abc;
        //        CREATE TABLE t1(a, b);
        //      ROLLBACK TO abc;
        //    COMMIT;
        //  }
        //  db close
        //} {}
        //do_test pager1-25-2 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    SAVEPOINT abc;
        //      CREATE TABLE t1(a, b);
        //    ROLLBACK TO abc;
        //    COMMIT;
        //  }
        //  db close
        //} {}

        //#-------------------------------------------------------------------------
        //# Sector-size tests.
        //#
        //do_test pager1-26.1 {
        //  testvfs tv -default 1
        //  tv sectorsize 4096
        //  faultsim_delete_and_reopen
        //  db func a_string a_string
        //  execsql {
        //    PRAGMA page_size = 512;
        //    CREATE TABLE tbl(a PRIMARY KEY, b UNIQUE);
        //    BEGIN;
        //      INSERT INTO tbl VALUES(a_string(25), a_string(600));
        //      INSERT INTO tbl SELECT a_string(25), a_string(600) FROM tbl;
        //      INSERT INTO tbl SELECT a_string(25), a_string(600) FROM tbl;
        //      INSERT INTO tbl SELECT a_string(25), a_string(600) FROM tbl;
        //      INSERT INTO tbl SELECT a_string(25), a_string(600) FROM tbl;
        //      INSERT INTO tbl SELECT a_string(25), a_string(600) FROM tbl;
        //      INSERT INTO tbl SELECT a_string(25), a_string(600) FROM tbl;
        //      INSERT INTO tbl SELECT a_string(25), a_string(600) FROM tbl;
        //    COMMIT;
        //  }
        //} {}
        //do_execsql_test pager1-26.1 {
        //  UPDATE tbl SET b = a_string(550);
        //} {}
        //db close
        //tv delete

        //#-------------------------------------------------------------------------
        //#
        //do_test pager1.27.1 {
        //  faultsim_delete_and_reopen
        //  sqlite3_pager_refcounts db
        //  execsql {
        //    BEGIN;
        //      CREATE TABLE t1(a, b);
        //  }
        //  sqlite3_pager_refcounts db
        //  execsql COMMIT
        //} {}

        //#-------------------------------------------------------------------------
        //# Test that attempting to open a write-transaction with 
        //# locking_mode=exclusive in WAL mode fails if there are other clients on 
        //# the same database.
        //#
        //catch { db close }
        //ifcapable wal {
        //  do_multiclient_test tn {
        //    do_test pager1-28.$tn.1 {
        //      sql1 { 
        //        PRAGMA journal_mode = WAL;
        //        CREATE TABLE t1(a, b);
        //        INSERT INTO t1 VALUES('a', 'b');
        //      }
        //    } {wal}
        //    do_test pager1-28.$tn.2 { sql2 { SELECT * FROM t1 } } {a b}

        //    do_test pager1-28.$tn.3 { sql1 { PRAGMA locking_mode=exclusive } } {exclusive}
        //    do_test pager1-28.$tn.4 { 
        //      csql1 { BEGIN; INSERT INTO t1 VALUES('c', 'd'); }
        //    } {1 {database is locked}}
        //    code2 { db2 close ; sqlite3 db2 test.db }
        //    do_test pager1-28.$tn.4 { 
        //      sql1 { INSERT INTO t1 VALUES('c', 'd'); COMMIT }
        //    } {}
        //  }
        //}

        //#-------------------------------------------------------------------------
        //# Normally, when changing from journal_mode=PERSIST to DELETE the pager
        //# attempts to delete the journal file. However, if it cannot obtain a
        //# RESERVED lock on the database file, this step is skipped.
        //#
        //do_multiclient_test tn {
        //  do_test pager1-28.$tn.1 {
        //    sql1 { 
        //      PRAGMA journal_mode = PERSIST;
        //      CREATE TABLE t1(a, b);
        //      INSERT INTO t1 VALUES('a', 'b');
        //    }
        //  } {persist}
        //  do_test pager1-28.$tn.2 { file exists test.db-journal } 1
        //  do_test pager1-28.$tn.3 { sql1 { PRAGMA journal_mode = DELETE } } delete
        //  do_test pager1-28.$tn.4 { file exists test.db-journal } 0

        //  do_test pager1-28.$tn.5 {
        //    sql1 { 
        //      PRAGMA journal_mode = PERSIST;
        //      INSERT INTO t1 VALUES('c', 'd');
        //    }
        //  } {persist}
        //  do_test pager1-28.$tn.6 { file exists test.db-journal } 1
        //  do_test pager1-28.$tn.7 {
        //    sql2 { BEGIN; INSERT INTO t1 VALUES('e', 'f'); }
        //  } {}
        //  do_test pager1-28.$tn.8  { file exists test.db-journal } 1
        //  do_test pager1-28.$tn.9  { sql1 { PRAGMA journal_mode = DELETE } } delete
        //  do_test pager1-28.$tn.10 { file exists test.db-journal } 1

        //  do_test pager1-28.$tn.11 { sql2 COMMIT } {}
        //  do_test pager1-28.$tn.12 { file exists test.db-journal } 0

        //  do_test pager1-28-$tn.13 {
        //    code1 { set channel [db incrblob -readonly t1 a 2] }
        //    sql1 {
        //      PRAGMA journal_mode = PERSIST;
        //      INSERT INTO t1 VALUES('g', 'h');
        //    }
        //  } {persist}
        //  do_test pager1-28.$tn.14 { file exists test.db-journal } 1
        //  do_test pager1-28.$tn.15 {
        //    sql2 { BEGIN; INSERT INTO t1 VALUES('e', 'f'); }
        //  } {}
        //  do_test pager1-28.$tn.16 { sql1 { PRAGMA journal_mode = DELETE } } delete
        //  do_test pager1-28.$tn.17 { file exists test.db-journal } 1

        //  do_test pager1-28.$tn.17 { csql2 { COMMIT } } {1 {database is locked}}
        //  do_test pager1-28-$tn.18 { code1 { read $channel } } c
        //  do_test pager1-28-$tn.19 { code1 { close $channel } } {}
        //  do_test pager1-28.$tn.20 { sql2 { COMMIT } } {}
        //}

        //do_test pager1-29.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA page_size = 1024;
        //    PRAGMA auto_vacuum = full;
        //    PRAGMA locking_mode=exclusive;
        //    CREATE TABLE t1(a, b);
        //    INSERT INTO t1 VALUES(1, 2);
        //  }
        //  file size test.db
        //} [expr 1024*3]
        //do_test pager1-29.2 {
        //  execsql {
        //    PRAGMA page_size = 4096;
        //    VACUUM;
        //  }
        //  file size test.db
        //} [expr 4096*3]

        //#-------------------------------------------------------------------------
        //# Test that if an empty database file (size 0 bytes) is opened in 
        //# exclusive-locking mode, any journal file is deleted from the file-system
        //# without being rolled back. And that the RESERVED lock obtained while
        //# doing this is not released.
        //#
        //do_test pager1-30.1 {
        //  db close
        //  delete_file test.db
        //  delete_file test.db-journal
        //  set fd [open test.db-journal w]
        //  seek $fd [expr 512+1032*2]
        //  puts -nonewline $fd x
        //  close $fd

        //  sqlite3 db test.db
        //  execsql {
        //    PRAGMA locking_mode=EXCLUSIVE;
        //    SELECT count(*) FROM sqlite_master;
        //    PRAGMA lock_status;
        //  }
        //} {exclusive 0 main reserved temp closed}

        //#-------------------------------------------------------------------------
        //# Test that if the "page-size" field in a journal-header is 0, the journal
        //# file can still be rolled back. This is required for backward compatibility -
        //# versions of SQLite prior to 3.5.8 always set this field to zero.
        //#
        //if {$tcl_platform(platform)=="unix"} {
        //do_test pager1-31.1 {
        //  faultsim_delete_and_reopen
        //  execsql {
        //    PRAGMA cache_size = 10;
        //    PRAGMA page_size = 1024;
        //    CREATE TABLE t1(x, y, UNIQUE(x, y));
        //    INSERT INTO t1 VALUES(randomblob(1500), randomblob(1500));
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(1500), randomblob(1500) FROM t1;
        //    BEGIN;
        //      UPDATE t1 SET y = randomblob(1499);
        //  }
        //  copy_file test.db test.db2
        //  copy_file test.db-journal test.db2-journal

        //  hexio_write test.db2-journal 24 00000000
        //  sqlite3 db2 test.db2
        //  execsql { PRAGMA integrity_check } db2
        //} {ok}
        //}

        //#-------------------------------------------------------------------------
        //# Test that a database file can be "pre-hinted" to a certain size and that
        //# subsequent spilling of the pager cache does not result in the database
        //# file being shrunk.
        //#
        //catch {db close}
        //forcedelete test.db

        //do_test pager1-32.1 {
        //  sqlite3 db test.db
        //  execsql {
        //    CREATE TABLE t1(x, y);
        //  }
        //  db close
        //  sqlite3 db test.db
        //  execsql {
        //    BEGIN;
        //    INSERT INTO t1 VALUES(1, randomblob(10000));
        //  }
        //  file_control_chunksize_test db main 1024
        //  file_control_sizehint_test db main 20971520; # 20MB
        //  execsql {
        //    PRAGMA cache_size = 10;
        //    INSERT INTO t1 VALUES(1, randomblob(10000));
        //    INSERT INTO t1 VALUES(2, randomblob(10000));
        //    INSERT INTO t1 SELECT x+2, randomblob(10000) from t1;
        //    INSERT INTO t1 SELECT x+4, randomblob(10000) from t1;
        //    INSERT INTO t1 SELECT x+8, randomblob(10000) from t1;
        //    INSERT INTO t1 SELECT x+16, randomblob(10000) from t1;
        //    SELECT count(*) FROM t1;
        //    COMMIT;
        //  }
        //  db close
        //  file size test.db
        //} {20971520}

        //# Cleanup 20MB file left by the previous test.
        //forcedelete test.db

        //#-------------------------------------------------------------------------
        //# Test that if a transaction is committed in journal_mode=DELETE mode,
        //# and the call to unlink() returns an ENOENT error, the COMMIT does not
        //# succeed.
        //#
        //if {$::tcl_platform(platform)=="unix"} {
        //  do_test pager1-33.1 {
        //    sqlite3 db test.db
        //    execsql {
        //      CREATE TABLE t1(x);
        //      INSERT INTO t1 VALUES('one');
        //      INSERT INTO t1 VALUES('two');
        //      BEGIN;
        //        INSERT INTO t1 VALUES('three');
        //        INSERT INTO t1 VALUES('four');
        //    }
        //    forcedelete bak-journal
        //    file rename test.db-journal bak-journal

        //    catchsql COMMIT
        //  } {1 {disk I/O error}}

        //  do_test pager1-33.2 {
        //    file rename bak-journal test.db-journal
        //    execsql { SELECT * FROM t1 }
        //  } {one two}
        //}

        //#-------------------------------------------------------------------------
        //# Test that appending pages to the database file then moving those pages
        //# to the free-list before the transaction is committed does not cause
        //# an error.
        //#
        //foreach {tn pragma strsize} {
        //  1 { PRAGMA mmap_size = 0 } 2400
        //  2 { }                       2400
        //  3 { PRAGMA mmap_size = 0 } 4400
        //  4 { }                       4400
        //} {
        //  reset_db
        //  db func a_string a_string
        //  db eval $pragma
        //  do_execsql_test 34.$tn.1 {
        //    CREATE TABLE t1(a, b);
        //    INSERT INTO t1 VALUES(1, 2);
        //  }
        //  do_execsql_test 34.$tn.2 {
        //    BEGIN;
        //    INSERT INTO t1 VALUES(2, a_string($strsize));
        //    DELETE FROM t1 WHERE oid=2;
        //    COMMIT;
        //    PRAGMA integrity_check;
        //  } {ok}
        //}

        //#-------------------------------------------------------------------------
        //#
        //reset_db
        //do_test 35 {
        //  sqlite3 db test.db

        //  execsql {
        //    CREATE TABLE t1(x, y);
        //    PRAGMA journal_mode = WAL;
        //    INSERT INTO t1 VALUES(1, 2);
        //  }

        //  execsql {
        //    BEGIN;
        //      CREATE TABLE t2(a, b);
        //  }

        //  hexio_write test.db-shm [expr 16*1024] [string repeat 0055 8192]
        //  catchsql ROLLBACK
        //} {0 {}}

        //do_multiclient_test tn {
        //  sql1 {
        //    PRAGMA auto_vacuum = 0;
        //    CREATE TABLE t1(x, y);
        //    INSERT INTO t1 VALUES(1, 2);
        //  }

        //  do_test 36.$tn.1 { 
        //    sql2 { PRAGMA max_page_count = 2 }
        //    list [catch { sql2 { CREATE TABLE t2(x) } } msg] $msg
        //  } {1 {database or disk is full}}

        //  sql1 { PRAGMA checkpoint_fullfsync = 1 }
        //  sql1 { CREATE TABLE t2(x) }

        //  do_test 36.$tn.2 { 
        //    sql2 { INSERT INTO t2 VALUES('xyz') }
        //    list [catch { sql2 { CREATE TABLE t3(x) } } msg] $msg
        //  } {1 {database or disk is full}}
        //}

        //forcedelete test1 test2
        //foreach {tn uri} {
        //  1   {file:?mode=memory&cache=shared}
        //  2   {file:one?mode=memory&cache=shared}
        //  3   {file:test1?cache=shared}
        //  4   {file:test2?another=parameter&yet=anotherone}
        //} {
        //  do_test 37.$tn {
        //    catch { db close }
        //    sqlite3_shutdown
        //    sqlite3_config_uri 1
        //    sqlite3 db $uri

        //    db eval {
        //      CREATE TABLE t1(x);
        //      INSERT INTO t1 VALUES(1);
        //      SELECT * FROM t1;
        //    }
        //  } {1}

        //  do_execsql_test 37.$tn.2 {
        //    VACUUM;
        //    SELECT * FROM t1;
        //  } {1}

        //  db close
        //  sqlite3_shutdown
        //  sqlite3_config_uri 0
        //}

        //do_test 38.1 {
        //  catch { db close }
        //  forcedelete test.db
        //  set fd [open test.db w]
        //  puts $fd "hello world"
        //  close $fd
        //  sqlite3 db test.db
        //  catchsql { CREATE TABLE t1(x) }
        //} {1 {file is encrypted or is not a database}}
        //do_test 38.2 {
        //  catch { db close }
        //  forcedelete test.db
        //} {}

        //do_test 39.1 {
        //  sqlite3 db test.db
        //  execsql {
        //    PRAGMA auto_vacuum = 1;
        //    CREATE TABLE t1(x);
        //    INSERT INTO t1 VALUES('xxx');
        //    INSERT INTO t1 VALUES('two');
        //    INSERT INTO t1 VALUES(randomblob(400));
        //    INSERT INTO t1 VALUES(randomblob(400));
        //    INSERT INTO t1 VALUES(randomblob(400));
        //    INSERT INTO t1 VALUES(randomblob(400));
        //    BEGIN;
        //    UPDATE t1 SET x = 'one' WHERE rowid=1;
        //  }
        //  set ::stmt [sqlite3_prepare db "SELECT * FROM t1 ORDER BY rowid" -1 dummy]
        //  sqlite3_step $::stmt
        //  sqlite3_column_text $::stmt 0
        //} {one}
        //do_test 39.2 {
        //  execsql { CREATE TABLE t2(x) }
        //  sqlite3_step $::stmt
        //  sqlite3_column_text $::stmt 0
        //} {two}
        //do_test 39.3 {
        //  sqlite3_finalize $::stmt
        //  execsql COMMIT
        //} {}

        //do_execsql_test 39.4 {
        //  PRAGMA auto_vacuum = 2;
        //  CREATE TABLE t3(x);
        //  CREATE TABLE t4(x);

        //  DROP TABLE t2;
        //  DROP TABLE t3;
        //  DROP TABLE t4;
        //}
        //do_test 39.5 {
        //  db close
        //  sqlite3 db test.db
        //  execsql {
        //    PRAGMA cache_size = 1;
        //    PRAGMA incremental_vacuum;
        //    PRAGMA integrity_check;
        //  }
        //} {ok}

        //do_test 40.1 {
        //  reset_db
        //  execsql {
        //    PRAGMA auto_vacuum = 1;
        //    CREATE TABLE t1(x PRIMARY KEY);
        //    INSERT INTO t1 VALUES(randomblob(1200));
        //    PRAGMA page_count;
        //  }
        //} {6}
        //do_test 40.2 {
        //  execsql {
        //    INSERT INTO t1 VALUES(randomblob(1200));
        //    INSERT INTO t1 VALUES(randomblob(1200));
        //    INSERT INTO t1 VALUES(randomblob(1200));
        //  }
        //} {}
        //do_test 40.3 {
        //  db close
        //  sqlite3 db test.db
        //  execsql {
        //    PRAGMA cache_size = 1;
        //    CREATE TABLE t2(x);
        //    PRAGMA integrity_check;
        //  }
        //} {ok}

        //do_test 41.1 {
        //  reset_db
        //  execsql {
        //    CREATE TABLE t1(x PRIMARY KEY);
        //    INSERT INTO t1 VALUES(randomblob(200));
        //    INSERT INTO t1 SELECT randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200) FROM t1;
        //  }
        //} {}
        //do_test 41.2 {
        //  testvfs tv -default 1
        //  tv sectorsize 16384;
        //  tv devchar [list]
        //  db close
        //  sqlite3 db test.db
        //  execsql {
        //    PRAGMA cache_size = 1;
        //    DELETE FROM t1 WHERE rowid%4;
        //    PRAGMA integrity_check;
        //  }
        //} {ok}
        //db close
        //tv delete

        //set pending_prev [sqlite3_test_control_pending_byte 0x1000000]
        //do_test 42.1 {
        //  reset_db
        //  execsql {
        //    CREATE TABLE t1(x, y);
        //    INSERT INTO t1 VALUES(randomblob(200), randomblob(200));
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //  }
        //  db close
        //  sqlite3_test_control_pending_byte 0x0010000
        //  sqlite3 db test.db
        //  db eval { PRAGMA mmap_size = 0 }
        //  catchsql { SELECT sum(length(y)) FROM t1 }
        //} {1 {database disk image is malformed}}
        //do_test 42.2 {
        //  reset_db
        //  execsql {
        //    CREATE TABLE t1(x, y);
        //    INSERT INTO t1 VALUES(randomblob(200), randomblob(200));
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //    INSERT INTO t1 SELECT randomblob(200), randomblob(200) FROM t1;
        //  }
        //  db close

        //  testvfs tv -default 1
        //  tv sectorsize 16384;
        //  tv devchar [list]
        //  sqlite3 db test.db -vfs tv
        //  execsql { UPDATE t1 SET x = randomblob(200) }
        //} {}
        //db close
        //tv delete
        //sqlite3_test_control_pending_byte $pending_prev

        //do_test 43.1 {
        //  reset_db
        //  execsql {
        //    CREATE TABLE t1(x, y);
        //    INSERT INTO t1 VALUES(1, 2);
        //    CREATE TABLE t2(x, y);
        //    INSERT INTO t2 VALUES(1, 2);
        //    CREATE TABLE t3(x, y);
        //    INSERT INTO t3 VALUES(1, 2);
        //  }
        //  db close
        //  sqlite3 db test.db

        //  db eval { PRAGMA mmap_size = 0 }
        //  db eval { SELECT * FROM t1 }
        //  sqlite3_db_status db CACHE_MISS 0
        //} {0 2 0}

        //do_test 43.2 {
        //  db eval { SELECT * FROM t2 }
        //  sqlite3_db_status db CACHE_MISS 1
        //} {0 3 0}

        //do_test 43.3 {
        //  db eval { SELECT * FROM t3 }
        //  sqlite3_db_status db CACHE_MISS 0
        //} {0 1 0}

    }
}