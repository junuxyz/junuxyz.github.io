+++
title = "VSCode C++ Configuration"
date = 2025-07-15T21:39:10+09:00
draft = true
categories = ['ML']
+++

# VSCode C++ Configuration

Unlike interpreted languages like Python that does not require prior compilation, C and C++ needs compilation.

we can use compilers like g++ or gcc and compile files/directories but using commands to compile, run, and debug them everytime is too inefficient.

This is why most C/C++ Programmers use pre-defined build/compile/run scripts for them to handle most cases with less typing/keys.

Although makefile is much

Here's mine for c/c++ compile and run. 


### Note.
My environment is
- VSCode
- WSL(Windows Subsystem for Linux) 2
- gcc, g++ compiler
- c, c++23

### Note 2.
Also if you want to use my configuration, you need to download **Command Variable** VSCode Extension. This is to handle whether the file to compile is C or C++ and match them either to gcc or g++ according to the file extension.


### Note 3. 
Strongly recommend you to copy paste the code to your User Tasks. You can find it using CtrlI(or Cmd)+Shift+P and find "Tasks: Open User Task".

I've made the "Compile and Run Active File" task to be default in line 40, so you can just press Ctrl(or Cmd)+Shift+B and it will run the task automatically.


This is the `tasks.json` code with the DOCSTRING for explanation of what each commands mean:

```cpp
{
  "version": "2.0.0",
  
  "tasks": [
    {
      /**
       * COMPILE TASK: Build C/C++ Files
       * 
       * Compiles the currently active C/C++ file with strict warnings.
       * This is the default build task (Ctrl+Shift+B).
       * 
       * Features:
       * - Automatic compiler selection (g++ for C++, gcc for C)
       * - Automatic standard selection (c++23 for C++, c23 for C)
       * - Strict warning and error checking
       * - Creates executable in same directory as source
       */
      "label": "Compile",
      "type": "shell",
      "command": "${input:chooseCompiler}",
      "args": [
        "${input:chooseStandard}",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-o",
        "${fileDirname}/${fileBasenameNoExtension}",
        "${file}"
      ],
      
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      },
      
      "group": {
        "kind": "build",
        "isDefault": true
      },
      
      "problemMatcher": "$gcc",
      "detail": "Compiles the active C/C++ file with strict warnings"
    },
    
    {
      /**
       * RUN TASK: Execute Compiled Program
       * 
       * Runs the executable for the current file.
       * Checks if executable exists before attempting to run.
       */
      "label": "Run",
      "type": "shell",
      "command": "${fileDirname}/${fileBasenameNoExtension}",
      
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      },
      
      "group": "test",
      "detail": "Runs the compiled executable"
    },
    
    {
      /**
       * DEBUG COMPILE TASK: Build with Debug Symbols
       * 
       * Compiles with debug symbols for use with debuggers like GDB.
       * Includes -g flag and colored diagnostics.
       */
      "label": "Compile Debug",
      "type": "shell",
      "command": "${input:chooseCompiler}",
      "args": [
        "${input:chooseStandard}",
        "-g",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fdiagnostics-color=always",
        "-o",
        "${fileDirname}/${fileBasenameNoExtension}",
        "${file}"
      ],
      
      "group": "build",
      "problemMatcher": "$gcc",
      "detail": "Compiles with debug symbols (-g flag)"
    },
    
    {
      /**
       * RELEASE COMPILE TASK: Build Optimized Version
       * 
       * Compiles with optimizations for production/performance testing.
       * Uses -O2 optimization and defines NDEBUG.
       */
      "label": "Compile Release",
      "type": "shell",
      "command": "${input:chooseCompiler}",
      "args": [
        "${input:chooseStandard}",
        "-O2",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-o",
        "${fileDirname}/${fileBasenameNoExtension}",
        "${file}"
      ],
      
      "group": "build",
      "problemMatcher": "$gcc",
      "detail": "Compiles with optimizations for release"
    },
    
    {
      /**
       * CLEAN TASK: Remove Build Artifacts
       * 
       * Removes the compiled executable and any temporary files.
       */
      "label": "Clean",
      "type": "shell",
      "command": "rm",
      "args": [
        "-f",
        "${fileDirname}/${fileBasenameNoExtension}"
      ],
      
      "group": "build",
      "detail": "Removes compiled executable"
    }
  ],
  
  /**
   * INPUT VARIABLES
   * 
   * Dynamic variables that resolve based on current file context.
   * Requires the "Command Variable" VS Code extension.
   */
  "inputs": [
    {
      /**
       * COMPILER SELECTION
       * 
       * Maps file extensions to appropriate compilers:
       * - .cpp, .cc, .cxx → g++ (C++ compiler)
       * - .c → gcc (C compiler)
       */
      "id": "chooseCompiler",
      "type": "command",
      "command": "extension.commandvariable.file.fileAsKey",
      "args": {
        ".cpp": "g++",
        ".cc": "g++",
        ".cxx": "g++",
        ".c": "gcc"
      }
    },
    {
      /**
       * LANGUAGE STANDARD SELECTION
       * 
       * Maps file extensions to appropriate language standards:
       * - C++ files → -std=c++23
       * - C files → -std=c23
       */
      "id": "chooseStandard",
      "type": "command",
      "command": "extension.commandvariable.file.fileAsKey",
      "args": {
        ".cpp": "-std=c++23",
        ".cc": "-std=c++23",
        ".cxx": "-std=c++23",
        ".c": "-std=c23"
      }
    }
  ]
}
```