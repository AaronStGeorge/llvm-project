//===- jit-calc.cpp - MLIR Based JIT-Compiled Calculator ------------------===//
//
// We want a basic command line calculator... but wouldn't it be fun if it were
// massively over-engineered? We could implement a simple interpreter to do the
// parse-y and calculation-y bits of a calculator in maybe eighty lines of C++.
// But eighty lines of C++ isn't cool, you know what's cool? Eight million lines
// of C++. So let's use LLVM!
//
// By using LLVM, we can create an entirely different type of calculator. To do
// a calculation, a normal calculator executes instructions from its binary. If
// we smuggle a compiler inside the calculator binary, it can *generate* the
// instructions required to do a calculation. The calculator binary won't need
// to physically contain the instructions necessary to do the calculation, it
// can build them on the fly.
//
// This is hot stuff! Using LLVM, we can cut the calculator's binary size down
// by *dozens* of bytes, and we'll only need to pack a few gigs of compiler code
// in there to do so. Excellent. Trade.
//
//===----------------------------------------------------------------------===//
//
// Next Steps:
//  - learn enough about ORC APIs to have some idea of tasks that need to be
//    done to jit compile 2 + 2, execute the resulting code, and put the result
//    somewhere that could be output using printf call from C++.
//
//===----------------------------------------------------------------------===//

#include <iostream>

int main(int argc, char **argv) { std::cout << "jit-calc!" << std::endl; }
