+++
title = "(shallow) Dive into VSCode Debugger"
date = 2025-07-16T21:45:17+09:00
draft = false
categories = ['ML']
+++

# (shallow) Dive into VSCode Debugger


## Intro

I know debugging skills are very important and one of the "must have" skills for developers. However I did not explicitly tried to learn how to use and utilize VSCode debugger effectively. While reading [this](https://www.learncpp.com/cpp-tutorial/using-an-integrated-debugger-stepping/) during my entry to c++, I thought now was the right time to look into features VS Code gives, which were worth note taking. Today is just a shallow dive and hope to learn deeper when I need it.

>Don’t neglect learning to use a debugger. As your programs get more complicated, the amount of time you spend learning to use the integrated debugger effectively will pale in comparison to amount of time you save finding and fixing issues.
>***learncpp***

## Installation

Configuring development environment is especially important when it comes to C/C++. Since VS Code only has built-in supports for TypeScript, JavaScript, and Node.js, if we are using other language such as Python or C++, we would need to install debugging extensions from the VS Code Marketplace (use Ctrl/Cmd + Shift + X as a shortcut)

If what you are trying to compile is simple enough(especially if it's just one file you're trying to debug), default configuration works fine. So to experience the VS Code debugging, just click 'Run and Debug' section on the left corner and click the run and debug (or just type `F5` as a shortcut). If the directory you're trying to compile is complex, you would need to configure/edit `.vscode/launch.json` manually (which I will explore more in the [future](https://code.visualstudio.com/docs/debugtest/debugging-configuration) ..)


## Breakpoints

  <img src="/attachments/breakpoints.png" width="400" alt="breakpoints">

*this image is from the VS Code Docs!*

Breakpoints are one of the main features of debugging. They capture the state of variables, call stack, and loaded scripts on the lines where breakpoints are placed.
Note that you can actually modify values in the state variables as you want!

You can click on the **editor margin** or use F9 on the current line to set/unset breakpoints.


### Conditional Breakpoints

This is a feature I've never used before but seems very helpful. Conditional Breakpoint only hits when it meets the condition specified.

There are three types of conditions:
- **Expression condition**: The breakpoint is hit whenever the expression evaluates to `true`.
- **Hit count**: controls how many times a breakpoint needs to be hit before it interrupts execution. Whether a hit count is respected, and the exact syntax of the expression, can vary among debugger extensions.
- **Wait for breakpoint**: The breakpoint is activated when another breakpoint is hit

For example (Expression condition), if I set `getValue() = 3` in line 11 in the below code, it will pass the breakpoint.

![[shallow-dive-into-vscode-debugger-expression-1.png|400]]

But if I set `getValue() = 4` 

![[shallow-dive-into-vscode-debugger-expression-2.png|400]]

it will correctly stop at the breakpoint.

![[shallow-dive-into-vscode-debugger-expression-3.png|400]]

To create conditional breakpoint, right-click in the editor margin and select **Add Conditional Breakpoint**.


### Log Points

A logpoint is a variant of breakpoint, but works differently. Instead of stopping at a point, whenever it passes that line, it prints a log message in the debug console.
This feature is for easy/light log message addings.

For example if I 
![[shallow-dive-into-vscode-debugger-log-message-1.png|400]]

![[Pasted image shallow-dive-into-vscode-debugger-log-message-2.png|400]]



## Debugging C++ in VS Code

> Visual Studio Code supports the following debuggers for C/C++ depending on the operating system you are using:
> - **Linux**: GDB
> - **macOS**: LLDB or GDB
> - **Windows**: the Visual Studio Windows Debugger or GDB (using Cygwin or MinGW)

since I use WSL(Windows Subsystem for Linux) 2 as my dev environment, I use GDB debuggers.


## WIP

I will add some more (something beyond tutorial resources) information and practical tips on debugging to this post as my journey continues ..

## Resources
https://code.visualstudio.com/docs/debugtest/debugging
https://www.youtube.com/watch?v=3HiLLByBWkg
https://code.visualstudio.com/docs/cpp/cpp-debug