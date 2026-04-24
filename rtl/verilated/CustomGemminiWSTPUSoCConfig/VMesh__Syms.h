// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VMESH__SYMS_H_
#define VERILATED_VMESH__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "VMesh.h"

// INCLUDE MODULE CLASSES
#include "VMesh___024root.h"

// SYMS CLASS (contains all model state)
class VMesh__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    VMesh* const __Vm_modelp;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    VMesh___024root                TOP;

    // CONSTRUCTORS
    VMesh__Syms(VerilatedContext* contextp, const char* namep, VMesh* modelp);
    ~VMesh__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
