// tclUnix.h --
//
//	This file reads in UNIX-related header files and sets up UNIX-related macros for Tcl's UNIX core.  It should be the
//	only file that contains #ifdefs to handle different flavors of UNIX.  This file sets up the union of all UNIX-related
//	things needed by any of the Tcl core files.  This file depends on configuration #defines in tclConfig.h
//
// Copyright 1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that this copyright notice appears in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#ifndef __TCL_UNIX_H__
#define __TCL_UNIX_H__

// The following #defines are used to distinguish between different UNIX systems.  These #defines are normally set by the "config" script
// based on information it gets by looking in the include and library areas.  The defaults below are for BSD-based systems like SunOS or Ultrix.
//
#define TCL_GETTOD		0 // 1 means there exists a library procedure "gettimeofday" (e.g. BSD systems).  0 means have to use "times" instead.
#define TCL_GETWD		0 // 1 means there exists a library procedure "getwd" (e.g. BSD systems).  0 means have to use "getcwd" instead.
#define TCL_SYS_ERRLIST 0 // 1 means that the array sys_errlist is defined as part of the C library.
#define TCL_SYS_TIME_H	0 // 1 means there exists an include file <sys/time.h> (e.g. BSD derivatives).
#define TCL_SYS_WAIT_H	0 // 1 means there exists an include file <sys/wait.h> that defines constants related to the results of "wait".
#define TCL_UNION_WAIT	0 // 1 means that the "wait" system call returns a structure of type "union wait" (e.g. BSD systems).  0 means "wait" returns an int (e.g. System V and POSIX).
#define TCL_PID_T		0 // 1 means that <sys/types> defines the type pid_t.  0 means that it doesn't.
#define TCL_UID_T		0 // 1 means that <sys/types> defines the type uid_t.  0 means that it doesn't.

//#define HAVE_MKSTEMP
//#define HAVE_GETHOSTNAME

#include <RuntimeOS.h>
#include <sys/types.h>
#include <sys/stat.h>

// On systems without symbolic links (i.e. S_IFLNK isn't defined) define "lstat" to use "stat" instead.
#ifndef S_IFLNK
#define lstat stat
#endif

// Define macros to query file type bits, if they're not already defined.
#ifndef S_ISREG
# ifdef S_IFREG
# define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
# else
# define S_ISREG(m) 0
# endif
#endif
#ifndef S_ISDIR
# ifdef S_IFDIR
# define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
# else
# define S_ISDIR(m) 0
# endif
#endif
#ifndef S_ISCHR
# ifdef S_IFCHR
# define S_ISCHR(m) (((m) & S_IFMT) == S_IFCHR)
# else
# define S_ISCHR(m) 0
# endif
#endif
#ifndef S_ISBLK
# ifdef S_IFBLK
# define S_ISBLK(m) (((m) & S_IFMT) == S_IFBLK)
# else
# define S_ISBLK(m) 0
# endif
#endif
#ifndef S_ISFIFO
# ifdef S_IFIFO
# define S_ISFIFO(m) (((m) & S_IFMT) == S_IFIFO)
# else
# define S_ISFIFO(m) 0
# endif
#endif
#ifndef S_ISLNK
# ifdef S_IFLNK
# define S_ISLNK(m) (((m) & S_IFMT) == S_IFLNK)
# else
# define S_ISLNK(m) 0
# endif
#endif
#ifndef S_ISSOCK
# ifdef S_IFSOCK
# define S_ISSOCK(m) (((m) & S_IFMT) == S_IFSOCK)
# else
# define S_ISSOCK(m) 0
# endif
#endif

// Make sure that MAXPATHLEN is defined.
#ifndef MAXPATHLEN
#ifdef PATH_MAX
# define MAXPATHLEN PATH_MAX
#else
# define MAXPATHLEN 2048
#endif
#endif

// Define pid_t and uid_t if they're not already defined.
#if !TCL_PID_T
#define pid_t int
#endif
#if !TCL_UID_T
#define uid_t int
#endif

//// Variables provided by the C library:
//#if defined(_sgi) || defined(__sgi)
//#define environ _environ
//#endif
//extern char **environ;

// uClinux can't do fork(), only vfork()
#define NO_FORK

//// Library procedures used by Tcl but not declared in a header file:
//#ifndef _CRAY
//extern int access(const char *path, int mode);
//extern int chdir(const char *path);
//extern int close(int fd);
//extern int dup2(int src, int dst);
//extern void endpwent();
//// extern int execvp(const char *name, char **argv);
//extern void _exit(int status);
//// extern pid_t fork();
//// extern uid_t geteuid();
//// extern pid_t getpid();
//// extern char *getcwd(char *buffer, int size);
//extern char * getwd(char *buffer);
//// extern int kill(pid_t pid, int sig);
//// extern long lseek(int fd, int offset, int whence);
//extern char *mktemp(char *template_);
//#if !(defined(sparc) || defined(_IBMR2))
//extern int open(const char *path, int flags, ...);
//#endif
//extern int pipe(int *fdPtr);
//// extern int read(int fd, char *buf, int numBytes);
//// extern int readlink(const char *path, char *buf, int size);
//extern int unlink(const char *path);
//// extern int write(int fd, char *buf, int numBytes);
//#endif /* _CRAY */

#endif /* _TCLUNIX */
