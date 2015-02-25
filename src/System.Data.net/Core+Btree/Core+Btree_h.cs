using System;
namespace Core
{
    public delegate bool RefAction<T>(ref T value);

    #region CollSeq

    public class CollSeq
    {
        public string Name;				// Name of the collating sequence, UTF-8 encoded
        public TEXTENCODE Encode;		// Text encoding handled by xCmp()
        public object User;				// First argument to xCmp()
        public Func<object, int, string, int, string, int> Cmp;
        public RefAction<object> Del;	// Destructor for pUser
        public CollSeq memcpy() { return (this == null ? null : (CollSeq)MemberwiseClone()); }
    }

    #endregion

    #region Schema

    [Flags]
    public enum SCHEMA : byte
    {
        SchemaLoaded = 0x0001,      // The schema has been loaded
        UnresetViews = 0x0002,      // Some views have defined column names
        Empty = 0x0004,             // The file is empty (length 0 bytes)
    }

    public static class E
    {
        public static bool DbHasProperty(BContext D, int I, SCHEMA P) { return (((D).DBs[I].Schema.Flags & (P)) == (P)); }
        public static bool DbHasAnyProperty(BContext D, int I, SCHEMA P) { return (((D).DBs[I].Schema.Flags & (P)) != 0); }
        public static SCHEMA DbSetProperty(BContext D, int I, SCHEMA P) { return (D).DBs[I].Schema.Flags |= (P); }
        public static SCHEMA DbClearProperty(BContext D, int I, SCHEMA P) { return (D).DBs[I].Schema.Flags &= ~(P); }
    }

    public class Schema
    {
        public int SchemaCookie;		// Database schema version number for this file
        public int Generation;			// Generation counter.  Incremented with each change
        public Hash TableHash;			// All tables indexed by name
        public Hash IndexHash;			// All (named) indices indexed by name
        public Hash TriggerHash;		// All triggers indexed by name
        public Hash FKeyHash;			// All foreign keys by referenced table name
        public Table SeqTable;		    // The sqlite_sequence table used by AUTOINCREMENT
        public byte FileFormat;		    // Schema format version for this file
        public TEXTENCODE Encode;		// Text encoding used by this database
        public SCHEMA Flags;			// Flags associated with this schema
        public int CacheSize;			// Number of pages to use in the cache

        internal Schema memcpy()
        {
            if (this == null)
                return null;
            Schema cp = (Schema)MemberwiseClone();
            return cp;
        }

        internal void memset()
        {
            if (this != null)
            {
                SchemaCookie = 0;
                TableHash = new Hash();
                IndexHash = new Hash();
                TriggerHash = new Hash();
                FKeyHash = new Hash();
                SeqTable = null;
            }
        }
    }

    #endregion

    #region IVdbe

    public interface IVdbe
    {
        UnpackedRecord AllocUnpackedRecord(KeyInfo keyInfo, byte[] space, int spaceLength, out object free);
        void RecordUnpack(KeyInfo keyInfo, int keyLength, byte[] key, UnpackedRecord p);
        void DeleteUnpackedRecord(UnpackedRecord r);
        int RecordCompare(int cells, byte[] cellKey, uint offset_, UnpackedRecord idxKey);
    }

    #endregion
}