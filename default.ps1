properties { 
  $base_dir = resolve-path .
  $build_dir = "$base_dir\_build"
  $sln_file = "$base_dir\GpuEx.sln"
  $tools_dir = "$base_dir\tools"
  $version = "1.0.0"
  $config_cpu = "Release.cpu"
  $config_cpuD = "Debug.cpu"
  $config_cu = "Release.cu"
  $config_cuD = "Debug.cu"
  $run_tests = $false
  $full_compile = $false
}
Framework "4.0"
	
task default -depends Package

task Clean {
	remove-item -force -recurse $build_dir -ErrorAction SilentlyContinue
}

task Init -depends Clean {
	new-item $build_dir -itemType directory
}

task Compile -depends Init {
	if ($full_compile) {
		# 11
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\11.Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=11" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\11.Win32.Debug\;Configuration=$config_cuD;Platform=Win32;CUARCH=11" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\11.x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=11" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\11.x64.Debug\;Configuration=$config_cuD;Platform=x64;CUARCH=11" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m

		# cpu
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\cpu.Win32.Release\;Configuration=$config_cpu;Platform=Win32;CUARCH=cpu" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\cpu.Win32.Debug\;Configuration=$config_cpuD;Platform=Win32;CUARCH=cpu" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\cpu.x64.Release\;Configuration=$config_cpu;Platform=x64;CUARCH=cpu" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\cpu.x64.Debug\;Configuration=$config_cpuD;Platform=x64;CUARCH=cpu" /m
		#xcopy "$build_dir\cpu.Win32.Release\." "$build_dir\Win32.Release\" /Y ; rm "$build_dir\cpu.Win32.Release\" -force -recurse
		xcopy "$build_dir\cpu.Win32.Debug\." "$build_dir\Win32.Debug\" /Y ; rm "$build_dir\cpu.Win32.Debug\" -force -recurse
		#xcopy "$build_dir\cpu.x64.Release\." "$build_dir\x64.Release\" /Y ; rm "$build_dir\cpu.x64.Release\" -force -recurse
		xcopy "$build_dir\cpu.x64.Debug\." "$build_dir\x64.Debug\" /Y ; rm "$build_dir\cpu.x64.Debug\" -force -recurse
		# 20
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\20.Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=20" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\20.Win32.Debug\;Configuration=$config_cuD;Platform=Win32;CUARCH=20" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\20.x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=20" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\20.x64.Debug\;Configuration=$config_cuD;Platform=x64;CUARCH=20" /m
		#xcopy "$build_dir\20.Win32.Release\." "$build_dir\Win32.Release\" /Y ; rm "$build_dir\20.Win32.Release\" -force -recurse
		xcopy "$build_dir\20.Win32.Debug\." "$build_dir\Win32.Debug\" /Y ; rm "$build_dir\20.Win32.Debug\" -force -recurse
		#xcopy "$build_dir\20.x64.Release\." "$build_dir\x64.Release\" /Y ; rm "$build_dir\20.x64.Release\" -force -recurse
		xcopy "$build_dir\20.x64.Debug\." "$build_dir\x64.Debug\" /Y ; rm "$build_dir\20.x64.Debug\" -force -recurse
		# 30
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\30.Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=30" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\30.Win32.Debug\;Configuration=$config_cuD;Platform=Win32;CUARCH=30" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\30.x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=30" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\30.x64.Debug\;Configuration=$config_cuD;Platform=x64;CUARCH=30" /m
		#xcopy "$build_dir\30.Win32.Release\." "$build_dir\Win32.Release\" /Y ; rm "$build_dir\30.Win32.Release\" -force -recurse
		xcopy "$build_dir\30.Win32.Debug\." "$build_dir\Win32.Debug\" /Y ; rm "$build_dir\30.Win32.Debug\" -force -recurse
		#xcopy "$build_dir\30.x64.Release\." "$build_dir\x64.Release\" /Y ; rm "$build_dir\30.x64.Release\" -force -recurse
		xcopy "$build_dir\30.x64.Debug\." "$build_dir\x64.Debug\" /Y ; rm "$build_dir\30.x64.Debug\" -force -recurse
		# 35
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\35.Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=35" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\35.Win32.Debug\;Configuration=$config_cuD;Platform=Win32;CUARCH=35" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\35.x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=35" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\35.x64.Debug\;Configuration=$config_cuD;Platform=x64;CUARCH=35" /m
		#xcopy "$build_dir\35.Win32.Release\." "$build_dir\Win32.Release\" /Y ; rm "$build_dir\35.Win32.Release\" -force -recurse
		xcopy "$build_dir\35.Win32.Debug\." "$build_dir\Win32.Debug\" /Y ; rm "$build_dir\35.Win32.Debug\" -force -recurse
		#xcopy "$build_dir\35.x64.Release\." "$build_dir\x64.Release\" /Y ; rm "$build_dir\35.x64.Release\" -force -recurse
		xcopy "$build_dir\35.x64.Debug\." "$build_dir\x64.Debug\" /Y ; rm "$build_dir\35.x64.Debug\" -force -recurse
		# 50
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\50.Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=50" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\50.Win32.Debug\;Configuration=$config_cuD;Platform=Win32;CUARCH=50" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\50.x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=50" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\50.x64.Debug\;Configuration=$config_cuD;Platform=x64;CUARCH=50" /m
		#xcopy "$build_dir\50.Win32.Release\." "$build_dir\Win32.Release\" /Y ; rm "$build_dir\50.Win32.Release\" -force -recurse
		xcopy "$build_dir\50.Win32.Debug\." "$build_dir\Win32.Debug\" /Y ; rm "$build_dir\50.Win32.Debug\" -force -recurse
		#xcopy "$build_dir\50.x64.Release\." "$build_dir\x64.Release\" /Y ; rm "$build_dir\50.x64.Release\" -force -recurse
		xcopy "$build_dir\50.x64.Debug\." "$build_dir\x64.Debug\" /Y ; rm "$build_dir\50.x64.Debug\" -force -recurse
		# 52
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\52.Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=52" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\52.Win32.Debug\;Configuration=$config_cuD;Platform=Win32;CUARCH=52" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\52.x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=52" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\52.x64.Debug\;Configuration=$config_cuD;Platform=x64;CUARCH=52" /m
		#xcopy "$build_dir\52.Win32.Release\." "$build_dir\Win32.Release\" /Y ; rm "$build_dir\52.Win32.Release\" -force -recurse
		xcopy "$build_dir\52.Win32.Debug\." "$build_dir\Win32.Debug\" /Y ; rm "$build_dir\52.Win32.Debug\" -force -recurse
		#xcopy "$build_dir\52.x64.Release\." "$build_dir\x64.Release\" /Y ; rm "$build_dir\52.x64.Release\" -force -recurse
		xcopy "$build_dir\52.x64.Debug\." "$build_dir\x64.Debug\" /Y ; rm "$build_dir\52.x64.Debug\" -force -recurse
		# 60
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\60.Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=60" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\60.Win32.Debug\;Configuration=$config_cuD;Platform=Win32;CUARCH=60" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\60.x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=60" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\60.x64.Debug\;Configuration=$config_cuD;Platform=x64;CUARCH=60" /m
		#xcopy "$build_dir\60.Win32.Release\." "$build_dir\Win32.Release\" /Y ; rm "$build_dir\60.Win32.Release\" -force -recurse
		#xcopy "$build_dir\60.Win32.Debug\." "$build_dir\Win32.Debug\" /Y ; rm "$build_dir\60.Win32.Debug\" -force -recurse
		#xcopy "$build_dir\60.x64.Release\." "$build_dir\x64.Release\" /Y ; rm "$build_dir\60.x64.Release\" -force -recurse
		#xcopy "$build_dir\60.x64.Debug\." "$build_dir\x64.Debug\" /Y ; rm "$build_dir\60.x64.Debug\" -force -recurse
	} else {
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Win32.Release\;Configuration=$config_cpu;Platform=Win32;CUARCH=cpu" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Win32.Release\;Configuration=$config_cu;Platform=Win32;CUARCH=52" /m
		#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\x64.Release\;Configuration=$config_cpu;Platform=x64;CUARCH=cpu" /m
		msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\x64.Release\;Configuration=$config_cu;Platform=x64;CUARCH=52" /m
	}
}

task Test -depends Compile -precondition { return $run_tests } {
	$old = pwd
	cd $build_dir
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.11.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.20.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.30.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.35.Tests.dll" /noshadow
	cd $old
}

task Dependency -precondition { return $false } {
	$package_files = @(Get-ChildItem src -include *packages.config -recurse)
	foreach ($package in $package_files)
	{
		& $tools_dir\NuGet.exe install $package.FullName -o packages
	}
}

task Package -depends Dependency, Compile, Test {
#task Package {
	$spec_files = @(Get-ChildItem $base_dir\src -include *.nuspec -recurse)
	foreach ($spec in $spec_files)
	{
		& $tools_dir\NuGet.exe pack $spec.FullName -o $build_dir -Symbols -BasePath $base_dir
	}

	#$old = pwd
	#cd $build_dir
	#$spec_files = @(Get-ChildItem $base_dir\src -include *.autopkg -recurse)
	#foreach ($spec in $spec_files)
	#{
	#	Write-NuGetPackage $spec.FullName
	#}
	#cd $old
}

task Push {
	$spec_files = @(Get-ChildItem $base_dir -include *.nupkg -recurse)
	foreach ($spec in $spec_files)
	{
		& $tools_dir\NuGet.exe push $spec.FullName
	}
}