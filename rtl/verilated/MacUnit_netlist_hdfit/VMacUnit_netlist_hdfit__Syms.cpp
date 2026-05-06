// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "VMacUnit_netlist_hdfit__Syms.h"
#include "VMacUnit_netlist_hdfit.h"
#include "VMacUnit_netlist_hdfit___024root.h"

// FUNCTIONS
VMacUnit_netlist_hdfit__Syms::~VMacUnit_netlist_hdfit__Syms()
{
}

VMacUnit_netlist_hdfit__Syms::VMacUnit_netlist_hdfit__Syms(VerilatedContext* contextp, const char* namep,VMacUnit_netlist_hdfit* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-12);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
}
