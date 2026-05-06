// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VMacUnit_netlist_hdfit.h for the primary calling header

#include "verilated.h"

#include "VMacUnit_netlist_hdfit__Syms.h"
#include "VMacUnit_netlist_hdfit___024root.h"

void VMacUnit_netlist_hdfit___024root___ctor_var_reset(VMacUnit_netlist_hdfit___024root* vlSelf);

VMacUnit_netlist_hdfit___024root::VMacUnit_netlist_hdfit___024root(VMacUnit_netlist_hdfit__Syms* symsp, const char* name)
    : VerilatedModule{name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    VMacUnit_netlist_hdfit___024root___ctor_var_reset(this);
}

void VMacUnit_netlist_hdfit___024root::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

VMacUnit_netlist_hdfit___024root::~VMacUnit_netlist_hdfit___024root() {
}
