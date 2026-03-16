#ifndef PE_STATE__H
#define PE_STATE__H

#include <iostream>
#include <map>
#include <vector>
#include "utils.h"
#include "config.h"
#include "FaultList.h"

using namespace std;


class Pe
{
public:
    Pe(uint8_t r, uint8_t c) : peRow(r), peCol(c)
    {}

    ~Pe(){}

    /* Pointers to tile signals (see how these signals are mapped to the tile in the file signals.txt) */

    void *ptr_in_a;  // input a (propag left->right)
    void *ptr_in_b;  // input b (propag top->bottom)
    void *ptr_in_d;  // input d (propag top->bottom) - preloads only
   
    void *ptr_out_a = nullptr; // the last column has not out_a output register    
    void *ptr_out_b;
    void *ptr_out_c;

    void *ptr_valid;
    void *ptr_propagate;

    void *ptr_c1;
    void *ptr_c2;   

    void reset();

    void flipBitInA(uint8_t bit);
    void flipBitInB(uint8_t bit);
    void flipBitInD(uint8_t bit);

    void flipBitOutA(uint8_t bit);
    void flipBitOutB(uint8_t bit);
    void flipBitMacOut(uint8_t bit);

    void flipBitPropagB();
    void flipBitValid();

    void flipBitC1(uint8_t bit);
    void flipBitC2(uint8_t bit);

#ifdef ENABLE_PERMANENT_FAULTS
    void stuckAtBitInA(uint8_t bit, uint8_t polarity);
    void stuckAtBitInB(uint8_t bit, uint8_t polarity);
    void stuckAtBitInD(uint8_t bit, uint8_t polarity);
    void stuckAtBitOutC(uint8_t bit, uint8_t polarity);
    void stuckAtPropB(uint8_t bit, uint8_t pol);
    void stuckAtValid(uint8_t bit, uint8_t pol);
#endif
    
    Input_t getInputA()
    {
        return *(Input_t*)ptr_in_a;
    }
    
    Input_t getInputB()
    {
        return *(Input_t*)ptr_in_b;
    }

    Input_t getInputD()
    {
        return *(Input_t*)ptr_in_d;
    }

    Output_t getOutput()
    {
    #ifdef GEMM_OS // this will be the MAC unit output. this is currenlty pointing to pe_io_out, so that we can inject in the same part HDFIT does
        return *(Output_t*)ptr_out_c;
    #else // This will the the partial sum passed to downstream PEs
        return *(Output_t*)ptr_out_b;
    #endif
    }

    void setMacOut (Output_t data)
    {  
    #ifdef GEMM_OS
        *(Output_t*)ptr_out_c = data;
    #else
        *(Output_t*)ptr_out_b = data;
    #endif
    }

    uint8_t peRow, peCol;
};


void Pe::reset()
{
    *(Input_t*)ptr_in_a = 0;
    *(Input_t*)ptr_in_b = 0;
    *(Input_t*)ptr_in_d = 0;
    *(CData*)ptr_propagate = 0;
    *(CData*)ptr_valid = 0;
}


void Pe::flipBitInA(uint8_t bit)
{   
    uint32_t mask = 1U << bit; 
    *(Input_t*)ptr_in_a ^= mask;
}


void Pe::flipBitOutA(uint8_t bit)
{   
    if (ptr_out_a != nullptr) // the last column in the array has no register for in_a
    {
        uint32_t mask = 1U << bit; 
        *(Input_t*)ptr_out_a ^= mask;
    }
}


void Pe::flipBitInB(uint8_t bit)
{
    uint8_t mask = 1U << bit;
    *(Input_t*)ptr_in_b ^= mask;
}


void Pe::flipBitOutB(uint8_t bit)
{
    uint8_t mask = 1U << bit;
    *(Input_t*)ptr_out_b ^= mask;
}


/* 
    Notes for faults in D for OS: this wont do anything unless faults are injected during preloads. 
    That's because the in_d inputs (fault injected) are stored to C1 during preloads. During streaming,
    the in_d registers are no longer used.
*/
void Pe::flipBitInD(uint8_t bit)
{
    uint8_t mask = 1U << bit;
    *(Input_t*)ptr_in_d ^= mask;
}


void Pe::flipBitMacOut(uint8_t bit) // Funciona de acordo com o comentado na funcao stuckAtBitOutC
{  
    uint32_t mask = 1U << bit; 
    *(Output_t*)ptr_out_c ^= mask;
}


void Pe::flipBitC1(uint8_t bit)
{   
    uint32_t mask = 1U << bit; 
    *(Output_t*)ptr_c1 ^= mask;
}


void Pe::flipBitC2(uint8_t bit)
{   
    uint32_t mask = 1U << bit; 
    *(Output_t*)ptr_c2 ^= mask;
}


void Pe::flipBitValid()
{
    *(int*)ptr_valid = (*(int*)ptr_valid)^1U;  
}


void Pe::flipBitPropagB ()
{
    *(CData*)ptr_propagate = (*(CData*)ptr_propagate)^1U; 
}














#ifdef ENABLE_PERMANENT_FAULTS

#define IS_MSB_BIT_0 1

void Pe::stuckAtBitOutC(uint8_t bit, uint8_t polarity)
{  
#if IS_MSB_BIT_0     // bit = 7 - bit;
    bit = 31 - bit;  // Salvo's MSB(LSB) is bit 0(7)
#endif

    uint32_t mask = 1U << bit;

    if (polarity == 1) 
    #ifdef GEMM_OS
        *(Output_t*)ptr_c1 |= mask;
    #else
        *(Output_t*)ptr_out_b |= mask;
    #endif

    else
    {
        mask = ~mask;
    #ifdef GEMM_OS
        *(Output_t*)ptr_c1 &= mask;
    #else
        *(Output_t*)ptr_out_b &= mask;
    #endif
    }
}


void Pe::stuckAtBitInA(uint8_t bit, uint8_t polarity)
{  
#if IS_MSB_BIT_0     // bit = 7 - bit;
    bit = 7 - bit;  // Salvo's MSB(LSB) is bit 0(7)
#endif

    uint32_t mask = 1U << bit; 

    if (polarity == 1)
        *(Input_t*)ptr_in_a |= mask;

    else
    {
        mask = ~mask;
        *(Input_t*)ptr_in_a &= mask;
    }
}


void Pe::stuckAtBitInB(uint8_t bit, uint8_t polarity)
{ 
#if IS_MSB_BIT_0     // bit = 7 - bit;
    bit = 7 - bit;  // Salvo's MSB(LSB) is bit 0(7)
#endif

    uint32_t mask = 1U << bit; // e.g., 00001000

    if (polarity == 1)
        *(Input_t*)ptr_in_b |= mask;
    
    else
    {
        mask = ~mask;  //e.g., 11110111
        *(Input_t*)ptr_in_b &= mask;
    }
}


void Pe::stuckAtBitInD(uint8_t bit, uint8_t polarity)
{ 
#if IS_MSB_BIT_0     // bit = 7 - bit;
    bit = 7 - bit;  // Salvo's MSB(LSB) is bit 0(7)
#endif

    uint32_t mask = 1U << bit; // e.g., 00001000

    if (polarity == 1)
        *(Input_t*)ptr_in_d |= mask;
    
    else
    {
        mask = ~mask;  //e.g., 11110111
        *(Input_t*)ptr_in_d &= mask;
    }
}


void Pe::stuckAtPropB(uint8_t bit, uint8_t polarity) // ptr_propagate is propagate_b (last_s) in WS(OS) mode
{
    uint32_t mask = 1U << bit; // e.g., 00001000

    if (polarity == 1)
         *(CData*)ptr_propagate |= mask;
    
    else
    {
        mask = ~mask;  //e.g., 11110111
         *(CData*)ptr_propagate &= mask;
    }
} 



void Pe::stuckAtValid(uint8_t bit, uint8_t polarity) // ptr_propagate is propagate_b (last_s) in WS(OS) mode
{
    uint32_t mask = 1U << bit; // e.g., 00001000

    if (polarity == 1)
         *(CData*)ptr_valid |= mask;
    
    else
    {
        mask = ~mask;  //e.g., 11110111
         *(CData*)ptr_valid &= mask;
    }
} 
#endif // ENABLE_PERMANENT_FAULTS

#endif










/*

Verilator flags in https://veripool.org/guide/latest/exe_verilator.html
Verilator type definitions (see /usr/local/share/verilator/include/verilated.h)

typedef vluint8_t    CData;     ///< Verilated pack data, 1-8 bits
typedef vluint16_t   SData;     ///< Verilated pack data, 9-16 bits
typedef vluint32_t   IData;     ///< Verilated pack data, 17-32 bits
typedef vluint64_t   QData;     ///< Verilated pack data, 33-64 bits
typedef vluint32_t   EData;     ///< Verilated pack element of WData 

typedef unsigned char vluint8_t;  ///< 8-bit unsigned type
typedef vluint8_t    CData;      ///< Verilated pack data, 1-8 bits
(so CData = uint8_t)

typedef unsigned short int vluint16_t;  ///< 16-bit unsigned type
typedef vluint16_t   SData;            ///< Verilated pack data, 9-16 bits
(so SData = uint16_t)

typedef uint32_t vluint32_t;  ///< 32-bit unsigned type
typedef vluint32_t   IData;  ///< Verilated pack data, 17-32 bits
(so IData = uint32_t)

#define VL_SIG(name, msb, lsb) IData name  ///< Declare signal, 17-32 bits
#define VL_SIGW(name, msb, lsb, words) WData name[words]  ///< Declare signal, 65+ bits
#define VL_IN8(name, msb, lsb) CData name  ///< Declare input signal, 1-8 bits
#define VL_OUT8(name, msb, lsb) CData name  ///< Declare output signal, 1-8 bits
#define VL_OUT(name, msb, lsb) IData name  ///< Declare output signal, 17-32 bits

*/

