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
	# cpu
	msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cpu;Platform=x64;LC=cpu;LD=" /m
	msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cpuD;Platform=x64;LC=cpu;LD=" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cpu;Platform=x64;LC=cpu;LD=V" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cpuD;Platform=x64;LC=cpu;LD=V" /m
	# 11
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=11;LD=" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuDPlatform=x64;;LC=11;LD=" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=11;LD=V" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuD;Platform=x64;LC=11;LD=V" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	# 20
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=20;LD=" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuD;Platform=x64;LC=20;LD=" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=20;LD=V" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuD;Platform=x64;LC=20;LD=V" /m
	# 30
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=30;LD=" /m
	msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuD;Platform=x64;LC=30;LD=" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=30;LD=V" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuD;Platform=x64;LC=30;LD=V" /m
	# 35
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=35;LD=" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuD;Platform=x64;LC=35;LD=" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Release\;Configuration=$config_cu;Platform=x64;LC=35;LD=V" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\Debug\;Configuration=$config_cuD;Platform=x64;LC=35;LD=V" /m
}

task Test -depends Compile -precondition { return $run_tests } {
	$old = pwd
	cd $build_dir
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.11.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.11V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.20.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.20V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.30.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.30V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.35.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Debug\Runtime.35V.Tests.dll" /noshadow
	cd $old
}

task Dependency -precondition { return $false } {
	$package_files = @(Get-ChildItem src -include *packages.config -recurse)
	foreach ($package in $package_files)
	{
		& $tools_dir\NuGet.exe install $package.FullName -o packages
	}
}

#task Package -depends Dependency, Compile, Test {
task Package  {
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

task Push -depends Package {
	$spec_files = @(Get-ChildItem $release_dir -include *.nupkg -recurse)
	foreach ($spec in $spec_files)
	{
		& $tools_dir\NuGet.exe push $spec.FullName -s {NuGetServerUrl}
	}
}