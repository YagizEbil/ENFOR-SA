// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VMesh.h for the primary calling header

#include "verilated.h"

#include "VMesh__Syms.h"
#include "VMesh___024root.h"

void VMesh___024root___ctor_var_reset(VMesh___024root* vlSelf);

VMesh___024root::VMesh___024root(VMesh__Syms* symsp, const char* name)
    : VerilatedModule{name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VMesh___024root___ctor_var_reset(this);
}

void VMesh___024root::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

VMesh___024root::~VMesh___024root() {
}
