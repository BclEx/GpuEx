using System;

namespace Core
{
    public static partial class C
    {
#if !OMIT_BUILTIN_TEST
        public struct BenignMallocHooks
        {
            public Action BenignBegin;
            public Action BenignEnd;
            public BenignMallocHooks(Action benignBegin, Action benignEnd)
            {
                BenignBegin = benignBegin;
                BenignEnd = benignEnd;
            }
        }
        static BenignMallocHooks g_BenignMallocHooks = new BenignMallocHooks(null, null);

        public static void _benignalloc_hook(Action benignBegin, Action benignEnd)
        {
            g_BenignMallocHooks.BenignBegin = benignBegin;
            g_BenignMallocHooks.BenignEnd = benignEnd;
        }

        public static void _benignalloc_begin()
        {
            if (g_BenignMallocHooks.BenignBegin != null)
                g_BenignMallocHooks.BenignBegin();
        }

        public static void _benignalloc_end()
        {
            if (g_BenignMallocHooks.BenignEnd != null)
                g_BenignMallocHooks.BenignEnd();
        }
#else
        public static void _benignalloc_begin() { }
        public static void _benignalloc_end() { }
#endif
    }
}
