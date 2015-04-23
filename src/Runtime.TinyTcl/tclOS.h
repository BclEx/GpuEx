/*
 * tclUnix.h --
 *
 *	This file reads in UNIX-related header files and sets up
 *	UNIX-related macros for Tcl's UNIX core.  It should be the
 *	only file that contains #ifdefs to handle different flavors
 *	of UNIX.  This file sets up the union of all UNIX-related
 *	things needed by any of the Tcl core files.  This file
 *	depends on configuration #defines in tclConfig.h
 *
 *	The material in this file was originally contributed by
 *	Karl Lehenbauer, Mark Diekhans and Peter da Silva.
 *
 * Copyright 1991 Regents of the University of California
 * Permission to use, copy, modify, and distribute this
 * software and its documentation for any purpose and without
 * fee is hereby granted, provided that this copyright
 * notice appears in all copies.  The University of California
 * makes no representations about the suitability of this
 * software for any purpose.  It is provided "as is" without
 * express or implied warranty.
 *
 * $Id: tclUnix.h,v 1.1.1.1 2001/04/29 20:35:04 karll Exp $
 */

#ifndef _TCLUNIX
#define _TCLUNIX

/*
 * The following #defines are used to distinguish between different
 * UNIX systems.  These #defines are normally set by the "config" script
 * based on information it gets by looking in the include and library
 * areas.  The defaults below are for BSD-based systems like SunOS
 * or Ultrix.
 *
 * TCL_GETTOD -			1 means there exists a library procedure
 *				"gettimeofday" (e.g. BSD systems).  0 means
 *				have to use "times" instead.
 * TCL_GETWD -			1 means there exists a library procedure
 *				"getwd" (e.g. BSD systems).  0 means
 *				have to use "getcwd" instead.
 * TCL_SYS_ERRLIST -		1 means that the array sys_errlist is
 *				defined as part of the C library.
 * TCL_SYS_TIME_H -		1 means there exists an include file
 *				<sys/time.h> (e.g. BSD derivatives).
 * TCL_SYS_WAIT_H -		1 means there exists an include file
 *				<sys/wait.h> that defines constants related
 *				to the results of "wait".
 * TCL_UNION_WAIT -		1 means that the "wait" system call returns
 *				a structure of type "union wait" (e.g. BSD
 *				systems).  0 means "wait" returns an int
 *				(e.g. System V and POSIX).
 * TCL_PID_T -			1 means that <sys/types> defines the type
 *				pid_t.  0 means that it doesn't.
 * TCL_UID_T -			1 means that <sys/types> defines the type
 *				uid_t.  0 means that it doesn't.
 */

#define TCL_GETTOD 0
#define TCL_SYS_ERRLIST 0
#define TCL_SYS_TIME_H 0
#define TCL_SYS_WAIT_H 0
#define TCL_UNION_WAIT 0
#define TCL_PID_T 0
#define TCL_UID_T 0
#define TCL_PW 0

//#define HAVE_MKSTEMP
//#define HAVE_GETHOSTNAME

#include <RuntimeOS.h>
#include <sys/types.h>
#include <sys/stat.h>

/*
 * On systems without symbolic links (i.e. S_IFLNK isn't defined)
 * define "lstat" to use "stat" instead.
 */

#ifndef S_IFLNK
#   define lstat stat
#endif

/*
 * Define macros to query file type bits, if they're not already
 * defined.
 */

#ifndef S_ISREG
#   ifdef S_IFREG
#       define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#   else
#       define S_ISREG(m) 0
#   endif
# endif
#ifndef S_ISDIR
#   ifdef S_IFDIR
#       define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#   else
#       define S_ISDIR(m) 0
#   endif
# endif
#ifndef S_ISCHR
#   ifdef S_IFCHR
#       define S_ISCHR(m) (((m) & S_IFMT) == S_IFCHR)
#   else
#       define S_ISCHR(m) 0
#   endif
# endif
#ifndef S_ISBLK
#   ifdef S_IFBLK
#       define S_ISBLK(m) (((m) & S_IFMT) == S_IFBLK)
#   else
#       define S_ISBLK(m) 0
#   endif
# endif
#ifndef S_ISFIFO
#   ifdef S_IFIFO
#       define S_ISFIFO(m) (((m) & S_IFMT) == S_IFIFO)
#   else
#       define S_ISFIFO(m) 0
#   endif
# endif
#ifndef S_ISLNK
#   ifdef S_IFLNK
#       define S_ISLNK(m) (((m) & S_IFMT) == S_IFLNK)
#   else
#       define S_ISLNK(m) 0
#   endif
# endif
#ifndef S_ISSOCK
#   ifdef S_IFSOCK
#       define S_ISSOCK(m) (((m) & S_IFMT) == S_IFSOCK)
#   else
#       define S_ISSOCK(m) 0
#   endif
# endif

/*
 * Make sure that MAXPATHLEN is defined.
 */

#ifndef MAXPATHLEN
#   ifdef PATH_MAX
#       define MAXPATHLEN PATH_MAX
#   else
#       define MAXPATHLEN 2048
#   endif
#endif

/*
 * Define pid_t and uid_t if they're not already defined.
 */

#if ! TCL_PID_T
#   define pid_t int
#endif
#if ! TCL_UID_T
#   define uid_t int
#endif

/*
 * Variables provided by the C library:
 */

#if defined(_sgi) || defined(__sgi)
#define environ _environ
#endif
extern char **environ;

/* uClinux can't do fork(), only vfork() */
#define NO_FORK

/*
 * Library procedures used by Tcl but not declared in a header file:
 */

#ifndef _CRAY
extern int	access	   _ANSI_ARGS_((CONST char *path, int mode));
extern int	chdir	   _ANSI_ARGS_((CONST char *path));
extern int	close	   _ANSI_ARGS_((int fd));
extern int	dup2	   _ANSI_ARGS_((int src, int dst));
extern void	endpwent   _ANSI_ARGS_((void));
/* extern int	execvp	   _ANSI_ARGS_((CONST char *name, char **argv)); */
extern void	_exit 	   _ANSI_ARGS_((int status));
/* extern pid_t	fork	   _ANSI_ARGS_((void)); */
/* extern uid_t	geteuid	   _ANSI_ARGS_((void)); */
/* extern pid_t	getpid	   _ANSI_ARGS_((void)); */
/* extern char *	getcwd 	   _ANSI_ARGS_((char *buffer, int size)); */
extern char *	getwd  	   _ANSI_ARGS_((char *buffer));
/* extern int	kill	   _ANSI_ARGS_((pid_t pid, int sig)); */
/* extern long	lseek	   _ANSI_ARGS_((int fd, int offset, int whence)); */
extern char *	mktemp	   _ANSI_ARGS_((char *template_));
#if !(defined(sparc) || defined(_IBMR2))
extern int	open	   _ANSI_ARGS_((CONST char *path, int flags, ...));
#endif
extern int	pipe	   _ANSI_ARGS_((int *fdPtr));
/* extern int	read	   _ANSI_ARGS_((int fd, char *buf, int numBytes)); */
/*extern int	readlink   _ANSI_ARGS_((CONST char *path, char *buf, int size));*/
extern int	unlink 	   _ANSI_ARGS_((CONST char *path));
/* extern int	write	   _ANSI_ARGS_((int fd, char *buf, int numBytes)); */
#endif /* _CRAY */

#endif /* _TCLUNIX */