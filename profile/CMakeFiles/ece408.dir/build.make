# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /build

# Include any dependencies generated for this target.
include CMakeFiles/ece408.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ece408.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ece408.dir/flags.make

CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o: /src/src/main.cu
CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o: CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o.depend
CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o: CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o"
	cd /build/CMakeFiles/ece408.dir/src && /usr/bin/cmake -E make_directory /build/CMakeFiles/ece408.dir/src/.
	cd /build/CMakeFiles/ece408.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/build/CMakeFiles/ece408.dir/src/./ece408_generated_main.cu.o -D generated_cubin_file:STRING=/build/CMakeFiles/ece408.dir/src/./ece408_generated_main.cu.o.cubin.txt -P /build/CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o.cmake

# Object files for target ece408
ece408_OBJECTS =

# External object files for target ece408
ece408_EXTERNAL_OBJECTS = \
"/build/CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o"

ece408: CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o
ece408: CMakeFiles/ece408.dir/build.make
ece408: /usr/local/cuda/lib64/libcudart_static.a
ece408: /usr/lib/x86_64-linux-gnu/librt.so
ece408: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
ece408: /usr/lib/x86_64-linux-gnu/libpthread.so
ece408: /usr/lib/x86_64-linux-gnu/libsz.so
ece408: /usr/lib/x86_64-linux-gnu/libz.so
ece408: /usr/lib/x86_64-linux-gnu/libdl.so
ece408: /usr/lib/x86_64-linux-gnu/libm.so
ece408: /usr/lib/x86_64-linux-gnu/libz.so
ece408: /usr/lib/x86_64-linux-gnu/libdl.so
ece408: /usr/lib/x86_64-linux-gnu/libm.so
ece408: CMakeFiles/ece408.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ece408"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ece408.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ece408.dir/build: ece408

.PHONY : CMakeFiles/ece408.dir/build

CMakeFiles/ece408.dir/requires:

.PHONY : CMakeFiles/ece408.dir/requires

CMakeFiles/ece408.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ece408.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ece408.dir/clean

CMakeFiles/ece408.dir/depend: CMakeFiles/ece408.dir/src/ece408_generated_main.cu.o
	cd /build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /src /src /build /build /build/CMakeFiles/ece408.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ece408.dir/depend

