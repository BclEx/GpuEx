/* autogenerated - do not edit */
#include "jim.h"

int Jim_initjimshInit(Jim_Interp *interp)
{
	if (Jim_PackageProvide(interp, "$pkgname", "1.0", JIM_ERRMSG)) return JIM_ERR;
	return Jim_EvalSource(interp, "$basename", 1,
		"proc _jimsh_init {} {\n"
		"	rename _jimsh_init {}\n"
		"	global jim::exe jim::argv0 tcl_interactive auto_path tcl_platform\n"
		"\n"
		"	# Stash the result of [info nameofexecutable] now, before a possible [cd]\n"
		"	if {[exists jim::argv0]} {\n"
		"		if {[string match \"*/*\" $jim::argv0]} {\n"
		"			set jim::exe [file join [pwd] $jim::argv0]\n"
		"		} else {\n"
		"			foreach path [split [env PATH \"\"] $tcl_platform(pathSeparator)] {\n"
		"				set exec [file join [pwd] [string map {\\ /} $path] $jim::argv0]\n"
		"				if {[file executable $exec]} {\n"
		"					set jim::exe $exec\n"
		"					break\n"
		"				}\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"\n"
		"	# Add to the standard auto_path\n"
		"	lappend p {*}[split [env JIMLIB {}] $tcl_platform(pathSeparator)]\n"
		"	if {[exists jim::exe]} {\n"
		"		lappend p [file dirname $jim::exe]\n"
		"	}\n"
		"	lappend p {*}$auto_path\n"
		"	set auto_path $p\n"
		"\n"
		"	if {$tcl_interactive && [env HOME {}] ne \"\"} {\n"
		"		foreach src {.jimrc jimrc.tcl} {\n"
		"			if {[file exists [env HOME]/$src]} {\n"
		"				uplevel #0 source [env HOME]/$src\n"
		"				break\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"	return \"\"\n"
		"}\n"
		"\n"
		"if {$tcl_platform(platform) eq \"windows\"} {\n"
		"	set jim::argv0 [string map {\\ /} $jim::argv0]\n"
		"}\n"
		"\n"
		"_jimsh_init\n");
}