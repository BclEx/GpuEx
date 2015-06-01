#include <Runtime.h>
#include "futils.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

#define BUF_SIZE 1024 

typedef	struct chunk CHUNK;
#define	CHUNKINITSIZE 4
struct chunk
{
	CHUNK *next;
	char data[CHUNKINITSIZE]; // actually of varying length
};
static CHUNK *chunklist;

// Return TRUE if a filename is a directory. Nonexistant files return FALSE.
bool isadir(char *name)
{
	struct stat statbuf;
	if (stat(name, &statbuf) < 0)
		return false;
	return S_ISDIR(statbuf.st_mode);
}

// Copy one file to another, while possibly preserving its modes, times, and modes.  Returns TRUE if successful, or FALSE on a failure with an
// error message output.  (Failure is not indicted if the attributes cannot be set.)
bool copyfile(char *srcname, char *destname, bool setmodes)
{
	struct stat statbuf1;
	if (stat(srcname, &statbuf1) < 0)
	{
		perror(srcname);
		return false;
	}
	struct stat statbuf2;
	if (stat(destname, &statbuf2) < 0)
	{
		statbuf2.st_ino = -1;
		statbuf2.st_dev = -1;
	}
	if (statbuf1.st_dev == statbuf2.st_dev && statbuf1.st_ino == statbuf2.st_ino)
	{
		fprintf(stderr, "Copying file \"%s\" to itself\n", srcname);
		return false;
	}
	//
	FILE *rfd = _fopen(srcname, "r");
	if (!rfd)
	{
		perror(srcname);
		return false;
	}
	FILE *wfd = _fopen(destname, statbuf1.st_mode);
	if (!wfd)
	{
		perror(destname);
		_fclose(rfd);
		return false;
	}
	//
	char *buf = malloc(BUF_SIZE);
	int rcc;
	while ((rcc = _fread(buf, 1, BUF_SIZE, rfd)) > 0)
	{
		char *bp = buf;
		while (rcc > 0)
		{
			int wcc = _fwrite(bp, 1, rcc, wfd);
			if (wcc < 0)
			{
				perror(destname);
				goto error_exit;
			}
			bp += wcc;
			rcc -= wcc;
		}
	}
	if (rcc < 0)
	{
		perror(srcname);
		goto error_exit;
	}
	_fclose(rfd);
	if (_fclose(wfd) < 0)
	{
		perror(destname);
		return false;
	}
	if (setmodes)
	{
		chmod(destname, statbuf1.st_mode);
		chown(destname, statbuf1.st_uid, statbuf1.st_gid);
		struct utimbuf times;
		times.actime = statbuf1.st_atime;
		times.modtime = statbuf1.st_mtime;
		utime(destname, &times);
	}
	return true;

error_exit:
	_fclose(rfd);
	_fclose(wfd);
	return false;
}

// Build a path name from the specified directory name and file name. If the directory name is NULL, then the original filename is returned.
// The built path is in a static area, and is overwritten for each call.
char *buildname(char *dirname, char *filename)
{
	if (!dirname || *dirname == '\0')
		return filename;
	char *cp = strrchr(filename, '/');
	if (cp)
		filename = cp + 1;
	static char buf[PATHLEN];
	strcpy(buf, dirname);
	strcat(buf, "/");
	strcat(buf, filename);
	return buf;
}

int main(int argc, char	**argv)
{
	atexit(RuntimeSentinel::ClientShutdown);
	RuntimeSentinel::ClientInitialize();
	char *lastarg = argv[argc - 1];
	bool dirflag = isadir(lastarg);
	if (argc > 3 && !dirflag)
	{
		fprintf(stderr, "%s: not a directory\n", lastarg);
		exit(1);
	}
	while (argc-- > 2)
	{
		char *srcname = argv[1];
		char *destname = lastarg;
		if (dirflag)
			destname = buildname(destname, srcname);
		copyfile(*++argv, destname, false);
	}
	exit(0);
}
