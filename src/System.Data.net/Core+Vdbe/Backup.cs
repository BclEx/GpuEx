using Pid = System.UInt32;
using IPage = Core.PgHdr;
using Core.IO;
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class Backup : IBackup
    {
        public Context DestCtx;         // Destination database handle
        public Btree Dest;              // Destination b-tree file
        public uint DestSchema;         // Original schema cookie in destination
        public bool DestLocked;          // True once a write-transaction is open on pDest

        public Pid NextId;               // Page number of the next source page to copy
        public Context SrcCtx;          // Source database handle
        public Btree Src;               // Source b-tree file

        public RC RC_;                   // Backup process error code

        // These two variables are set by every call to backup_step(). They are read by calls to backup_remaining() and backup_pagecount().
        public Pid Remaining;           // Number of pages left to copy */
        public Pid Pagecount;           // Total number of pages to copy */

        public bool IsAttached;         // True once backup has been registered with pager
        public Backup Next;             // Next backup associated with source pager

        static Btree FindBtree(Context errorCtx, Context ctx, string dbName)
        {
            int i = Parse.FindDbName(ctx, dbName);
            if (i == 1)
            {
                RC rc = 0;
                Parse parse = new Parse();
                if (parse == null)
                {
                    Main.Error(errorCtx, RC.NOMEM, "out of memory");
                    rc = RC.NOMEM;
                }
                else
                {
                    parse.Ctx = ctx;
                    if (parse.OpenTempDatabase() != 0)
                    {
                        Main.Error(errorCtx, parse.RC, "%s", parse.ErrMsg);
                        rc = RC.ERROR;
                    }
                    C._tagfree(errorCtx, ref parse.ErrMsg);
                }
                if (rc != 0)
                    return null;
            }

            if (i < 0)
            {
                Main.Error(errorCtx, RC.ERROR, "unknown database %s", dbName);
                return null;
            }
            return ctx.DBs[i].Bt;
        }

        static RC SetDestPgsz(Backup p)
        {
            RC rc = p.Dest.SetPageSize(p.Src.GetPageSize(), -1, false);
            return rc;
        }

        public static Backup Init(Context destCtx, string destDbName, Context srcCtx, string srcDbName)
        {
            // Lock the source database handle. The destination database handle is not locked in this routine, but it is locked in
            // sqlite3_backup_step(). The user is required to ensure that no other thread accesses the destination handle for the duration
            // of the backup operation.  Any attempt to use the destination database connection while a backup is in progress may cause
            // a malfunction or a deadlock.
            MutexEx.Enter(srcCtx.Mutex);
            MutexEx.Enter(destCtx.Mutex);

            Backup p;
            if (srcCtx == destCtx)
            {
                Main.Error(destCtx, RC.ERROR, "source and destination must be distinct");
                p = null;
            }
            else
            {
                // Allocate space for a new sqlite3_backup object... EVIDENCE-OF: R-64852-21591 The sqlite3_backup object is created by a
                // call to sqlite3_backup_init() and is destroyed by a call to sqlite3_backup_finish().
                p = new Backup();
                if (p == null)
                    Main.Error(destCtx, RC.NOMEM, null);
            }

            // If the allocation succeeded, populate the new object.
            if (p != null)
            {
                p.Src = FindBtree(destCtx, srcCtx, srcDbName);
                p.Dest = FindBtree(destCtx, destCtx, destDbName);
                p.DestCtx = destCtx;
                p.SrcCtx = srcCtx;
                p.NextId = 1;
                p.IsAttached = false;

                if (p.Src == null || p.Dest == null || SetDestPgsz(p) == RC.NOMEM)
                {
                    // One (or both) of the named databases did not exist or an OOM error was hit.  The error has already been written into the
                    // pDestDb handle.  All that is left to do here is free the sqlite3_backup structure.
                    //C._free(ref p);
                    p = null;
                }
            }
            if (p != null)
                p.Src.Backups++;

            MutexEx.Leave(destCtx.Mutex);
            MutexEx.Leave(srcCtx.Mutex);
            return p;
        }

        static bool IsFatalError(RC rc)
        {
            return (rc != RC.OK && rc != RC.BUSY && C._ALWAYS(rc != RC.LOCKED));
        }

        static RC BackupOnePage(Backup p, Pid srcPg, byte[] srcData, bool update)
        {
            Pager destPager = p.Dest.get_Pager();
            int srcPgsz = p.Src.GetPageSize();
            int destPgsz = p.Dest.GetPageSize();
            int copy = Math.Min(srcPgsz, destPgsz);
            long end = (long)srcPg * (long)srcPgsz;
            RC rc = RC.OK;

            Debug.Assert(p.Src.GetReserveNoMutex() >= 0);
            Debug.Assert(p.DestLocked);
            Debug.Assert(!IsFatalError(p.RC_));
            Debug.Assert(srcPg != Btree.PENDING_BYTE_PAGE(p.Src.Bt));
            Debug.Assert(srcData != null);

            // Catch the case where the destination is an in-memory database and the page sizes of the source and destination differ.
            if (srcPgsz != destPgsz && destPager.get_MemoryDB)
                rc = RC.READONLY;

#if HAS_CODEC
            int srcReserve = p.Src.GetReserveNoMutex();
            int destReserve = p.Dest.GetReserve();


            // Backup is not possible if the page size of the destination is changing and a codec is in use.
            if (srcPgsz != destPgsz && Pager.GetCodec(destPager) != null)
                rc = RC.READONLY;

            // Backup is not possible if the number of bytes of reserve space differ between source and destination.  If there is a difference, try to
            // fix the destination to agree with the source.  If that is not possible, then the backup cannot proceed.
            if (srcReserve != destReserve)
            {
                uint newPgsz = (uint)srcPgsz;
                rc = destPager.SetPageSize(ref newPgsz, srcReserve);
                if (rc == RC.OK && newPgsz != srcPgsz)
                    rc = RC.READONLY;
            }
#endif

            // This loop runs once for each destination page spanned by the source page. For each iteration, variable iOff is set to the byte offset
            // of the destination page.
            for (long off = end - (long)srcPgsz; rc == RC.OK && off < end; off += destPgsz)
            {
                IPage destPg = null;
                uint dest = (uint)(off / destPgsz) + 1;
                if (dest == Btree.PENDING_BYTE_PAGE(p.Dest.Bt))
                    continue;
                if ((rc = destPager.Acquire(dest, ref destPg, false)) == RC.OK && (rc = Pager.Write(destPg)) == RC.OK)
                {
                    byte[] destData = Pager.GetData(destPg);

                    // Copy the data from the source page into the destination page. Then clear the Btree layer MemPage.isInit flag. Both this module
                    // and the pager code use this trick (clearing the first byte of the page 'extra' space to invalidate the Btree layers
                    // cached parse of the page). MemPage.isInit is marked "MUST BE FIRST" for this purpose.
                    Buffer.BlockCopy(srcData, (int)(off % srcPgsz), destData, (int)(off % destPgsz), copy);
                    Pager.GetExtra(destPg).IsInit = false;
                }
                Pager.Unref(destPg);
            }

            return rc;
        }

        static RC BackupTruncateFile(VFile file, int size)
        {
            long current;
            RC rc = file.get_FileSize(out current);
            if (rc == RC.OK && current > size)
                rc = file.Truncate(size);
            return rc;
        }

        static void AttachBackupObject(Backup p)
        {
            Debug.Assert(p.Src.HoldsMutex());
            Backup pp = (Backup)p.Src.get_Pager().Backup;
            p.Next = pp;
            p.Src.get_Pager().Backup = p;
            p.IsAttached = true;
        }

        public RC Step(int pages)
        {
            MutexEx.Enter(SrcCtx.Mutex);
            Src.Enter();
            if (DestCtx != null)
                MutexEx.Enter(DestCtx.Mutex);

            RC rc = RC_;
            if (!IsFatalError(rc))
            {
                Pager srcPager = Src.get_Pager(); // Source pager
                Pager destPager = Dest.get_Pager(); // Dest pager
                Pid srcPage = 0; // Size of source db in pages
                bool closeTrans = false; // True if src db requires unlocking

                // If the source pager is currently in a write-transaction, return SQLITE_BUSY immediately.
                rc = (DestCtx != null && Src.Bt.InTransaction == TRANS.WRITE ? RC.BUSY : RC.OK);

                // Lock the destination database, if it is not locked already.
                if (rc == RC.OK && !DestLocked && (rc = Dest.BeginTrans(2)) == RC.OK)
                {
                    DestLocked = true;
                    Dest.GetMeta(Btree.META.SCHEMA_VERSION, ref DestSchema);
                }

                // If there is no open read-transaction on the source database, open one now. If a transaction is opened here, then it will be closed
                // before this function exits.
                if (rc == RC.OK && !Src.IsInReadTrans())
                {
                    rc = Src.BeginTrans(0);
                    closeTrans = true;
                }

                // Do not allow backup if the destination database is in WAL mode and the page sizes are different between source and destination
                int pgszSrc = Src.GetPageSize(); // Source page size
                int pgszDest = Dest.GetPageSize(); // Destination page size
                IPager.JOURNALMODE destMode = Dest.get_Pager().GetJournalMode(); // Destination journal mode
                if (rc == RC.OK && destMode == IPager.JOURNALMODE.WAL && pgszSrc != pgszDest)
                    rc = RC.READONLY;

                // Now that there is a read-lock on the source database, query the source pager for the number of pages in the database.
                srcPage = Src.LastPage();
                Debug.Assert(srcPage >= 0);
                for (int ii = 0; (pages < 0 || ii < pages) && NextId <= (Pid)srcPage && rc == 0; ii++)
                {
                    Pid srcPg = NextId; // Source page number
                    if (srcPg != Btree.PENDING_BYTE_PAGE(Src.Bt))
                    {
                        IPage srcPgAsObj = null; // Source page object
                        rc = srcPager.Acquire(srcPg, ref srcPgAsObj, false);
                        if (rc == RC.OK)
                        {
                            rc = BackupOnePage(p, srcPg, Pager.GetData(srcPgAsObj), false);
                            Pager.Unref(srcPgAsObj);
                        }
                    }
                    NextId++;
                }
                if (rc == RC.OK)
                {
                    Pagecount = srcPage;
                    Remaining = (srcPage + 1 - NextId);
                    if (NextId > srcPage)
                        rc = RC.DONE;
                    else if (!IsAttached)
                        AttachBackupObject(p);
                }

                // Update the schema version field in the destination database. This is to make sure that the schema-version really does change in
                // the case where the source and destination databases have the same schema version.
                if (rc == RC.DONE)
                {
                    if (srcPage == null)
                    {
                        rc = Dest.NewDb();
                        srcPage = 1;
                    }
                    if (rc == RC.OK || rc == RC.DONE)
                        rc = Dest.UpdateMeta(Btree.META.SCHEMA_VERSION, DestSchema + 1);
                    if (rc == RC.OK)
                    {
                        if (DestCtx != null)
                            Main.ResetAllSchemasOfConnection(DestCtx);
                        if (destMode == IPager.JOURNALMODE.WAL)
                            rc = Dest.SetVersion(2);
                    }
                    if (rc == RC.OK)
                    {
                        // Set nDestTruncate to the final number of pages in the destination database. The complication here is that the destination page
                        // size may be different to the source page size. 
                        //
                        // If the source page size is smaller than the destination page size, round up. In this case the call to sqlite3OsTruncate() below will
                        // fix the size of the file. However it is important to call sqlite3PagerTruncateImage() here so that any pages in the 
                        // destination file that lie beyond the nDestTruncate page mark are journalled by PagerCommitPhaseOne() before they are destroyed
                        // by the file truncation.

                        Debug.Assert(pgszSrc == Src.GetPageSize());
                        Debug.Assert(pgszDest == Dest.GetPageSize());
                        Pid destTruncate;
                        if (pgszSrc < pgszDest)
                        {
                            int ratio = pgszDest / pgszSrc;
                            destTruncate = (Pid)((srcPage + ratio - 1) / ratio);
                            if (destTruncate == Btree.PENDING_BYTE_PAGE(Dest.Bt))
                                destTruncate--;
                        }
                        else
                            destTruncate = (Pid)(srcPage * (pgszSrc / pgszDest));
                        Debug.Assert(destTruncate > 0);

                        if (pgszSrc < pgszDest)
                        {
                            // If the source page-size is smaller than the destination page-size, two extra things may need to happen:
                            //
                            //   * The destination may need to be truncated, and
                            //
                            //   * Data stored on the pages immediately following the pending-byte page in the source database may need to be
                            //     copied into the destination database.
                            int size = (int)(pgszSrc * srcPage);
                            VFile file = destPager.get_File();
                            Debug.Assert(file != null);
                            Debug.Assert((long)destTruncate * (long)pgszDest >= size || (destTruncate == (int)(Btree.PENDING_BYTE_PAGE(Dest.Bt) - 1) && size >= VFile.PENDING_BYTE && size <= VFile.PENDING_BYTE + pgszDest));

                            // This block ensures that all data required to recreate the original database has been stored in the journal for pDestPager and the
                            // journal synced to disk. So at this point we may safely modify the database file in any way, knowing that if a power failure
                            // occurs, the original database will be reconstructed from the journal file.
                            uint dstPage;
                            destPager.Pages(out dstPage);
                            for (Pid pg = destTruncate; rc == RC.OK && pg <= (Pid)dstPage; pg++)
                            {
                                if (pg != Btree.PENDING_BYTE_PAGE(Dest.Bt))
                                {
                                    IPage pgAsObj;
                                    rc = destPager.Acquire(pg, ref pgAsObj, false);
                                    if (rc == RC.OK)
                                    {
                                        rc = Pager.Write(pgAsObj);
                                        Pager.Unref(pgAsObj);
                                    }
                                }
                            }
                            if (rc == RC.OK)
                                rc = destPager.CommitPhaseOne(null, true);

                            // Write the extra pages and truncate the database file as required.
                            long end = Math.Min(VFile.PENDING_BYTE + pgszDest, size);
                            for (long off = VFile.PENDING_BYTE + pgszSrc; rc == RC.OK && off < end; off += pgszSrc)
                            {
                                Pid srcPg = (Pid)((off / pgszSrc) + 1);
                                PgHdr srcPgAsObj = null;
                                rc = srcPager.Acquire(srcPg, ref srcPgAsObj, false);
                                if (rc == RC.OK)
                                {
                                    byte[] data = Pager.GetData(srcPgAsObj);
                                    rc = file.Write(data, pgszSrc, off);
                                }
                                Pager.Unref(srcPgAsObj);
                            }
                            if (rc == RC.OK)
                                rc = BackupTruncateFile(file, (int)size);

                            // Sync the database file to disk. 
                            if (rc == RC.OK)
                                rc = destPager.Sync();
                        }
                        else
                        {
                            destPager.TruncateImage(destTruncate);
                            rc = destPager.CommitPhaseOne(null, false);
                        }

                        // Finish committing the transaction to the destination database.
                        if (rc == RC.OK && (rc = Dest.CommitPhaseTwo(false)) == RC.OK)
                            rc = RC.DONE;
                    }
                }

                // If bCloseTrans is true, then this function opened a read transaction on the source database. Close the read transaction here. There is
                // no need to check the return values of the btree methods here, as "committing" a read-only transaction cannot fail.
                if (closeTrans)
                {
#if !DEBUG || COVERAGE_TEST
                    RC rc2 = Src.CommitPhaseOne(null);
                    rc2 |= Src.CommitPhaseTwo(false);
                    Debug.Assert(rc2 == RC.OK);
#else
                    Src.CommitPhaseOne(null);
                    Src.CommitPhaseTwo(false);
#endif
                }

                if (rc == RC.IOERR_NOMEM)
                    rc = RC.NOMEM;
                RC_ = rc;
            }
            if (DestCtx != null)
                MutexEx.Leave(DestCtx.Mutex);
            Src.Leave();
            MutexEx.Leave(SrcCtx.Mutex);
            return rc;
        }

        public static RC Finish(Backup p)
        {
            // Enter the mutexes 
            if (p == null) return RC.OK;
            Context srcCtx = p.SrcCtx; // Source database connection
            MutexEx.Enter(srcCtx.Mutex);
            p.Src.Enter();
            if (p.DestCtx != null)
                MutexEx.Enter(p.DestCtx.Mutex);

            // Detach this backup from the source pager.
            if (p.DestCtx != null)
                p.Src.Backups--;
            if (p.IsAttached)
            {
                Backup pp = (Backup)p.Src.get_Pager().Backup;
                while (pp != p)
                    pp = (pp).Next;
                p.Src.get_Pager().Backup = p.Next;
            }

            // If a transaction is still open on the Btree, roll it back.
            p.Dest.Rollback(RC.OK);

            // Set the error code of the destination database handle.
            RC rc = (p.RC_ == RC.DONE ? RC.OK : p.RC_);
            Main.Error(p.DestCtx, rc, null);

            // Exit the mutexes and free the backup context structure.
            if (p.DestCtx != null)
                Main.LeaveMutexAndCloseZombie(p.DestCtx.Mutex);
            p.Src.Leave();
            //// EVIDENCE-OF: R-64852-21591 The sqlite3_backup object is created by a call to sqlite3_backup_init() and is destroyed by a call to sqlite3_backup_finish().
            //if (p.DestCtx != null)
            //    C._free(ref p);
            Main.LeaveMutexAndCloseZombie(mutex);
            return rc;
        }

        //public static int Remaining(Backup p)
        //{
        //    return (int)p.Remaining;
        //}

        //public static int Pagecount(Backup p)
        //{
        //    return (int)p.Pagecount;
        }

        public static void Update(Backup backup, Pid page, byte[] data)
        {
            for (Backup p = backup; p != null; p = p.Next)
            {
                Debug.Assert(MutexEx.Held(p.Src.Bt.Mutex));
                if (!IsFatalError(p.RC_) && page < p.NextId)
                {
                    // The backup process p has already copied page iPage. But now it has been modified by a transaction on the source pager. Copy
                    // the new data into the backup.
                    Debug.Assert(p.DestCtx != null);
                    MutexEx.Enter(p.DestCtx.Mutex);
                    RC rc = BackupOnePage(p, page, data, true);
                    MutexEx.Leave(p.DestCtx.Mutex);
                    Debug.Assert(rc != RC.BUSY && rc != RC.LOCKED);
                    if (rc != RC.OK)
                        p.RC_ = rc;
                }
            }
        }

        public static void Restart(Backup backup)
        {
            for (Backup p = backup; p != null; p = p.Next)
            {
                Debug.Assert(MutexEx.Held(p.Src.Bt.Mutex));
                p.NextId = 1;
            }
        }

#if !OMIT_VACUUM
        public static RC BtreeCopyFile(Btree to, Btree from)
        {
            to.Enter();
            from.Enter();

            Debug.Assert(to.IsInTrans());
            RC rc;
            VFile fd = to.get_Pager().get_File(); // File descriptor for database pTo
            if (true) //fd->Methods)
            {
                long bytes = from.GetPageSize() * (long)from.LastPage();
                rc = fd.FileControl(VFile.FCNTL.OVERWRITE, ref bytes);
                if (rc == RC.NOTFOUND) rc = RC.OK;
                if (rc != RC.OK) goto copy_finished;
            }

            // Set up an sqlite3_backup object. sqlite3_backup.pDestDb must be set to 0. This is used by the implementations of sqlite3_backup_step()
            // and sqlite3_backup_finish() to detect that they are being called from this function, not directly by the user.
            Backup b = new Backup();
            b.SrcCtx = (Context)from.Ctx;
            b.Src = from;
            b.Dest = to;
            b.NextId = 1;

            // 0x7FFFFFFF is the hard limit for the number of pages in a database file. By passing this as the number of pages to copy to
            // sqlite3_backup_step(), we can guarantee that the copy finishes within a single call (unless an error occurs). The assert() statement
            // checks this assumption - (p->rc) should be set to either SQLITE_DONE or an error code.
            b.Step(0x7FFFFFFF);
            Debug.Assert(b.RC_ != RC.OK);
            rc = Finish(b);
            if (rc == RC.OK)
                to.Bt.BtsFlags &= ~Btree.BTS.PAGESIZE_FIXED;
            else
                b.Dest.get_Pager().ClearCache();

            Debug.Assert(!to.IsInTrans());
        copy_finished:
            from.Leave();
            to.Leave();
            return rc;
        }
#endif
    }
}
