properties { 
  $base_dir = resolve-path .
  $build_dir = "$base_dir\build"
  $packageinfo_dir = "$base_dir\nuspecs"
  $release_dir = "$base_dir\Release"
  $sln_file = "$base_dir\GpuStructs.sln"
  $tools_dir = "$base_dir\tools"
  $version = "1.0.0"
  $config_cpu = "Release.cpu"
  $config_cpuD = "Debug.cpu"
  $config_cu = "Release.cu"
  $config_cuD = "Debug.cu"
  $run_tests = $true
}
Framework "4.0"
	
task default -depends Package

task Clean {
	remove-item -force -recurse $build_dir -ErrorAction SilentlyContinue
	remove-item -force -recurse $release_dir -ErrorAction SilentlyContinue
}

task Init -depends Clean {
	new-item $build_dir -itemType directory 
	new-item $release_dir -itemType directory 
}

task Compile -depends Init {
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.cpu\;Configuration=$config_cpu;LC=cpu;LD=" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.cpuD\;Configuration=$config_cpuD;LC=cpu;LD=" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.cpuV\;Configuration=$config_cpu;LC=cpu;LD=V" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.cpuVD\;Configuration=$config_cpuD;LC=cpu;LD=V" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.11\;Configuration=$config_cu;LC=11;LD=" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.11D\;Configuration=$config_cuD;LC=11;LD=" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.11V\;Configuration=$config_cu;LC=11;LD=V" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.11VD\;Configuration=$config_cuD;LC=11;LD=V" /t:"Runtime\Runtime:Rebuild;Runtime\Runtime_cu_Tests:Rebuild" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.20\;Configuration=$config_cu;LC=20;LD=" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.20D\;Configuration=$config_cuD;LC=20;LD=" /m
	msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.20V\;Configuration=$config_cu;LC=20;LD=V" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.20VD\;Configuration=$config_cuD;LC=20;LD=V" /m
	
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.30\;Configuration=$config_cu;LC=30;LD=" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.30V\;Configuration=$config_cu;LC=30;LD=V" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.35\;Configuration=$config_cu;LC=35;LD=" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\;IntDir=$build_dir\obj.35V\;Configuration=$config_cu;LC=35;LD=V" /m
}

task Test -depends Compile -precondition { return $run_tests } {
	$old = pwd
	cd $build_dir
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.11.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.11V.Tests.dll" /noshadow
	& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.20.Tests.dll" /noshadow
	& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.20V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.30.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.30V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.35.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\Runtime.35V.Tests.dll" /noshadow
	cd $old
}

task Dependency {
	$package_files = @(Get-ChildItem src -include *packages.config -recurse)
	foreach ($package in $package_files)
	{
		Write-Host $package.FullName
		& $tools_dir\NuGet.exe install $package.FullName -o packages
	}
}

task Package -depends Dependency, Compile, Test {
	$spec_files = @(Get-ChildItem $packageinfo_dir)
	foreach ($spec in $spec_files)
	{
		& $tools_dir\NuGet.exe pack $spec.FullName -o $release_dir -Version $version -Symbols -BasePath $base_dir
	}
}
