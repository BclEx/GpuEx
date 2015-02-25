namespace Core
{
    // sqlite3.h
    public abstract class VAlloc
    {
        public abstract byte[] Alloc(int bytes);            // Memory allocation function
        public abstract void Free(byte[] prior);			// Free a prior allocation
        public abstract byte[] Realloc(byte[] prior, int bytes);	// Resize an allocation
        public abstract int Size(byte[] p);			    // Return the size of an allocation
        public abstract int Roundup(int bytes);			    // Round up request size to allocation size
        public abstract RC Init(object appData);			    // Initialize the memory allocator
        public abstract void Shutdown(object appData);		// Deinitialize the memory allocator
        public object AppData;					        // Argument to xInit() and xShutdown()
    }
}
