#include <Runtime.h>
#include <string.h>
//#include <sys/types.h>
//#include <fcntl.h>
//#include <errno.h>

#define CAT_BUF_SIZE 4096

int cat_read_size = CAT_BUF_SIZE;
char colon[2] = { ':', ' ' };
char nl = '\n';

void dumpfile(FILE *f)
{
	int nred;
	static char readbuf[CAT_BUF_SIZE];
	while ((nred = _fread(readbuf, cat_read_size, 1, f)) > 0)
		_fwrite(readbuf, nred, 1, stdout);
}

int main(int argc, char **argv)
{
	if (argc <= 1)
		dumpfile(stdin);
	else
	{
		for (int i = 1; i < argc; i++)
		{
			FILE *f = _fopen(argv[i], "r");
			if (!f)
			{
				_fwrite(argv[0], strlen(argv[0]), 1, stderr);
				_fwrite(colon, 2, 1, stderr);
				_fwrite(argv[i], strlen(argv[i]), 1, stderr);
				_fwrite(colon, 2, 1, stderr);
				_fwrite(strerror(errno), strlen(strerror(errno)), 1, stderr);
				_fwrite(&nl, 1, 1, stderr);
			}
			else
			{
				dumpfile(f);
				_fclose(f);
			}
		}
	}
}
