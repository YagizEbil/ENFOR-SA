// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VMACUNIT_NETLIST_HDFIT__SYMS_H_
#define VERILATED_VMACUNIT_NETLIST_HDFIT__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "VMacUnit_netlist_hdfit.h"

// INCLUDE MODULE CLASSES
#include "VMacUnit_netlist_hdfit___024root.h"

// SYMS CLASS (contains all model state)
class VMacUnit_netlist_hdfit__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    VMacUnit_netlist_hdfit* const __Vm_modelp;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    VMacUnit_netlist_hdfit___024root TOP;

    // CONSTRUCTORS
    VMacUnit_netlist_hdfit__Syms(VerilatedContext* contextp, const char* namep, VMacUnit_netlist_hdfit* modelp);
    ~VMacUnit_netlist_hdfit__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
