import cppyy
cppyy.cppdef("""
#include <iostream>
void hello() { std::cout << "C++ JIT is working!" << std::endl; }
""")
cppyy.gbl.hello()