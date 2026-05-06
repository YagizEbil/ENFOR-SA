#ifndef MESH__H
#define MESH__H
    
#include <functional>
#include "Pe.h"
#include "config.h"
#include "utils.h"


typedef function<void(uint8_t)> callback_type;


class Mesh
{
public:
    Mesh () // context->meshGolden does not need pointers
    {
        pe.resize(DIM);
        
        for (uint8_t i = 0; i < DIM; i++)
            for (uint8_t j = 0; j < DIM; j++)
                 pe[i].push_back(new Pe(i, j)); 
    }

    Mesh (VL_DUT *tile): Mesh() 
    {
        this->dut = tile;
        loadPointers();

        /* Setup the callback functions to perform fault injections in all types of PE inputs 
            Ex.: to perform fault injections on input 1 (group IDX_MeshRegs) of PE(2,3), call cbApplyTransientFault[IDX_MeshRegs](2, 3); */
    
        cbApplyTransientFault.resize(NUMBER_OF_REGISTER_CLUSTERS);
        
        /* Transient data faults */
        cbApplyTransientFault[IDX_io_in_a]  = (callback_type)std::bind(&Mesh::flipBitInA,  this, std::placeholders::_1);
        cbApplyTransientFault[IDX_io_in_b]  = (callback_type)std::bind(&Mesh::flipBitInB,  this, std::placeholders::_1);
        cbApplyTransientFault[IDX_io_in_d]  = (callback_type)std::bind(&Mesh::flipBitInD,  this, std::placeholders::_1);
        
        cbApplyTransientFault[IDX_io_out_a] = (callback_type)std::bind(&Mesh::flipBitOutA,   this, std::placeholders::_1);
        cbApplyTransientFault[IDX_io_out_b] = (callback_type)std::bind(&Mesh::flipBitOutB,   this, std::placeholders::_1);
        cbApplyTransientFault[IDX_io_out_c] = (callback_type)std::bind(&Mesh::flipBitMacOut, this, std::placeholders::_1); 

        cbApplyTransientFault[IDX_c1] = (callback_type)std::bind(&Mesh::flipBitC1, this, std::placeholders::_1);
        cbApplyTransientFault[IDX_c2] = (callback_type)std::bind(&Mesh::flipBitC2, this, std::placeholders::_1); 

        /* Transient control faults */
        cbApplyTransientFault[IDX_valid] = (callback_type)std::bind(&Mesh::flipBitValid, this, std::placeholders::_1);
        cbApplyTransientFault[IDX_propagate] = (callback_type)std::bind(&Mesh::flipBitPropagB, this, std::placeholders::_1);

    #ifdef ENABLE_PERMANENT_FAULTS
        /* Permanent faults in data. */
        cbHoldPermanentFault.resize (NUMBER_OF_REGISTER_CLUSTERS);

        cbHoldPermanentFault[IDX_io_in_a]   = (callback_type)std::bind(&Mesh::holdPermaFaultInA,        this, std::placeholders::_1);
        cbHoldPermanentFault[IDX_io_in_b]   = (callback_type)std::bind(&Mesh::holdPermanentFaultInB,    this, std::placeholders::_1);
        cbHoldPermanentFault[IDX_io_in_d]   = (callback_type)std::bind(&Mesh::holdPermanentFaultInD,    this, std::placeholders::_1);
        cbHoldPermanentFault[IDX_io_out_c]  = (callback_type)std::bind(&Mesh::holdPermanentFaultOutC,   this, std::placeholders::_1);
        cbHoldPermanentFault[IDX_valid]     = (callback_type)std::bind(&Mesh::holdPermanentFaultValid,  this, std::placeholders::_1);
        cbHoldPermanentFault[IDX_propagate] = (callback_type)std::bind(&Mesh::holdPermanentFaultPropag, this, std::placeholders::_1);
    #endif
    }

    ~Mesh()
    {
        printfinish();

        if (dut != nullptr)
            delete dut;

        cbApplyTransientFault.clear();
    #ifdef ENABLE_PERMANENT_FAULTS
        cbHoldPermanentFault.clear();
    #endif
    
        for (uint8_t i = 0; i < DIM; i++)
            for (uint8_t j = 0; j < DIM; j++)
                delete pe[i][j];

        clearFaultList();
    }

    void loadPointers();
    void reset();

    Input_t getPrevInputA(uint8_t r, uint8_t col);
    Input_t getPrevInputB(uint8_t r, uint8_t col);

    void flipBitInA(uint8_t f);
    void flipBitInB(uint8_t f);
    void flipBitInD(uint8_t f);
    void flipBitOutA(uint8_t f);
    void flipBitOutB(uint8_t f);
    void flipBitMacOut(uint8_t f);
    void flipBitPropagB(uint8_t f);
    void flipBitValid(uint8_t f);

    void flipBitC1(uint8_t f);
    void flipBitC2(uint8_t f);

#ifdef ENABLE_PERMANENT_FAULTS
    void holdPermanentFaults();
    void holdPermaFaultInA(uint8_t idx);
    void holdPermanentFaultInB(uint8_t idx);
    void holdPermanentFaultInD(uint8_t idx);
    void holdPermanentFaultOutC(uint8_t idx);
    void holdPermanentFaultValid(uint8_t idx);
    void holdPermanentFaultPropag(uint8_t idx);
#endif

    void addFaultToList (Fault *fault)
    {
        faultList.push_back(fault);
    }

    void clearFaultList()
    {
        for (Fault* ptr: faultList) 
            delete ptr;  
        faultList.clear();
    }


    Output_t getMacOut()
    {  
        Fault *fault = faultList[0];
        return pe[fault->row][fault->col]->getOutput();
    }


    void setMacOut (Output_t data)
    {  
        Fault *fault = faultList[0];
        pe[fault->row][fault->col]->setMacOut(data);
    }


    // the Top verilated Mesh model
    VL_DUT *dut = nullptr;

    vector<vector<Pe*>>pe;

    vector<callback_type> cbApplyTransientFault;
    vector<callback_type> cbHoldPermanentFault;

    /* Mesh IO pins pointers */
    Input_t *ptr_mesh_in_a[DIM];
    Input_t *ptr_mesh_in_b[DIM];
    Input_t *ptr_mesh_in_d[DIM];

    Input_t *ptr_mesh_out_b[DIM];
    Output_t *ptr_mesh_out_c[DIM];
    
    CData *ptr_mesh_valid[DIM];
    CData *ptr_mesh_propagate[DIM];
    CData *ptr_mesh_dataflow[DIM];
    //CData *ptr_mesh_shift[DIM];

    vector<Fault*> faultList;
};


/* this loads the PE pointers by assigning each global verilated signal pointer (in the gemmini-pointers folder) to the PE pointers */
void Mesh::loadPointers()
{
/* includes the code with the pointer declarations to the verilated signals:
    1 - there must be one file for each config
    2 - the file names must match the <config_name> (i.e., passed in g++ -DCONFIG=<config_name>): name it like <config_name>.cc
*/
#ifdef GEMM_OS
#   include STR(gemmini-pointers/os/CONFIG.cc)
#else
#   include STR(gemmini-pointers/ws/CONFIG.cc)
#endif

//#include "gemmini-pointers/os/mediumosconfig_old_pointers.cc"

    /* PE pointers */
    for (uint8_t i = 0; i < DIM; i++)
    {
        for (uint8_t j = 0; j < DIM; j++)
        {
            pe[i][j]->ptr_in_a = in_a[i][j];
            pe[i][j]->ptr_in_b = in_b[i][j]; 
            pe[i][j]->ptr_in_d = in_d[i][j]; 

            pe[i][j]->ptr_out_a = out_a[i][j];      
            pe[i][j]->ptr_out_b = out_b[i][j]; 

            pe[i][j]->ptr_out_c = out_c[i][j];

            #if GEMM_OS
                pe[i][j]->ptr_mac_out_d = pe_io_out[i][j];
            #endif
 
            pe[i][j]->ptr_c1 = pe_c1[i][j];
            pe[i][j]->ptr_c2 = pe_c2[i][j];

            pe[i][j]->ptr_propagate = in_propag[i][j];
            pe[i][j]->ptr_valid = in_valid[i][j];
        }

        pe[0][i]->ptr_in_d = mesh_in_d[i];
    }

     /* Mesh IO signals */
    for (uint8_t i = 0; i < DIM; i++)
    {
        ptr_mesh_in_a[i] = (Input_t*)mesh_in_a[i];
        ptr_mesh_in_b[i] = (Input_t*)mesh_in_b[i];
        ptr_mesh_in_d[i] = (Input_t*)mesh_in_d[i];

        ptr_mesh_out_b[i] = (Input_t*)mesh_out_b[i];
        ptr_mesh_out_c[i] = (Output_t*)mesh_out_c[i];

        ptr_mesh_valid[i] = (CData*)mesh_valid[i];
        ptr_mesh_propagate[i] = (CData*)mesh_propagate[i];
        //ptr_mesh_dataflow[i] = (CData*)mesh_dataflow[i];
        //ptr_mesh_shift[i] = (CData*)mesh_shift[i];
     }
}


Input_t Mesh::getPrevInputA(uint8_t row, uint8_t col)
{   
    if (col == 0)
        return *((Input_t*)ptr_mesh_in_a[row]);

    else
        return pe[row][col-1]->getInputA();
}


Input_t Mesh::getPrevInputB(uint8_t row, uint8_t col)
{
    if (row == 0)
        return *((Input_t*)ptr_mesh_in_b[col]);

    else
        return pe[row-1][col]->getInputB();
}

/* io_in_a_0: flows from left to right

To flip the in_a register for pe(x,y), we flip the in_a register of pe(x,y-1)
If the target pe is the first column, we flip the mesh io_in_a_ input
*/
void Mesh::flipBitInA(uint8_t f)
{   
    Fault *fault = faultList[f];
    //pe[fault->row][fault->col]->flipBitInA(fault->bit);
    //return;

    if (fault->col == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        *(Input_t*)ptr_mesh_in_a[fault->row] ^= mask;
    }

    else
        pe[fault->row][fault->col-1]->flipBitInA(fault->bit);
}


/* io_b: flows from top to bottom
    
    To flip the in_b register for pe(x,y), we flip the in_b register of pe(x-1,y)
    If the target pe is the first row, we flip the mesh io_in_b_ input
*/
void Mesh::flipBitInB(uint8_t f)
{
    Fault *fault = faultList[f];
    //pe[fault->row][fault->col]->flipBitInB(fault->bit);
    //return;

    if (fault->row == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        *(Input_t*)ptr_mesh_in_b[fault->col] ^= mask;  
    }

    else
        pe[fault->row-1][fault->col]->flipBitInB(fault->bit);
}


/* 
    Notes for faults in D for OS: this wont do anything unless faults are injected during preloads. 
    That's because the in_d inputs (fault injected) are stored to C1 during preloads. During streaming,
    the in_d registers are no longer used.
*/

void Mesh::flipBitInD(uint8_t f) 
{
    Fault *fault = faultList[f];

    if (fault->row == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        *(Input_t*)ptr_mesh_in_d[fault->col] ^= mask;
    }

    else
        pe[fault->row-1][fault->col]->flipBitInD(fault->bit);
}


void Mesh::flipBitOutA(uint8_t f)
{
    Fault *fault = faultList[f];
    
    if (fault->col == 0)
        pe[fault->row][fault->col]->flipBitInA(fault->bit);
    
    else
        pe[fault->row][fault->col-1]->flipBitOutA(fault->bit);
}

void Mesh::flipBitOutB(uint8_t f)
{
    Fault *fault = faultList[f];
    //pe[fault->row][fault->col]->flipBitOutB(fault->bit);
    //return;

    if (fault->row == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        *(Input_t*)ptr_mesh_in_b[fault->col] ^= mask;   
    }

    else
        pe[fault->row-1][fault->col]->flipBitOutB(fault->bit);
}


void Mesh::flipBitC1(uint8_t f) // C1 and C2 registers are stationary after preloads. for now we can inject directly in them
{
    Fault *fault = faultList[f];
    pe[fault->row][fault->col]->flipBitC1(fault->bit);
}


void Mesh::flipBitC2(uint8_t f)
{
    Fault *fault = faultList[f];
    pe[fault->row][fault->col]->flipBitC2(fault->bit);
}


void Mesh::flipBitMacOut(uint8_t f)
{
    Fault *fault = faultList[f];
    pe[fault->row][fault->col]->flipBitMacOut(fault->bit);
}


void Mesh::flipBitPropagB(uint8_t f)
{
    Fault *fault = faultList[f];
    pe[fault->row][fault->col]->flipBitPropagB();
    return;
    
    if (fault->row == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        *(Input_t*)ptr_mesh_propagate[fault->col] ^= mask;  
    }

    else
        pe[fault->row-1][fault->col]->flipBitPropagB();
} 


void Mesh::flipBitValid(uint8_t f)
{
    Fault *fault = faultList[f];
    pe[fault->row][fault->col]->flipBitValid();
    return;

    if (fault->row == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        *(Input_t*)ptr_mesh_valid[fault->col] ^= mask;  
    }

    else
        pe[fault->row-1][fault->col]->flipBitValid();
}


#ifdef ENABLE_PERMANENT_FAULTS

void Mesh::holdPermanentFaults()
{
    for (uint8_t i = 0; i < faultList.size(); i++)
    {
        Fault *fault = faultList[i];

        if (fault->faultModel == FaultModel::FM_PERMANENT)
            cbHoldPermanentFault[fault->group](i);
    }
}


void Mesh::holdPermaFaultInA(uint8_t idx)
{
    Fault *fault = faultList[idx];

    if (fault->col == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        if (fault->pol == 1)
            *(Input_t*)ptr_mesh_in_a[fault->row] |= mask;

        else
        {
            mask = ~mask;
            *(Input_t*)ptr_mesh_in_a[fault->row] &= mask;
        }
    }

    else
        pe[fault->row][fault->col-1]->stuckAtBitInA(fault->bit, fault->pol);
}


void Mesh::holdPermanentFaultInB(uint8_t idx)
{
    Fault *fault = faultList[idx];

    if (fault->row == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        if (fault->pol == 1)
            *(Input_t*)ptr_mesh_in_b[fault->col] |= mask;

        else
        {
            mask = ~mask;
            *(Input_t*)ptr_mesh_in_b[fault->col] &= mask;
        }
    }

    else
        pe[fault->row-1][fault->col]->stuckAtBitInB(fault->bit, fault->pol);

}


void Mesh::holdPermanentFaultInD(uint8_t idx)
{
    Fault *fault = faultList[idx];

    if (fault->row == 0)
    {
        uint32_t mask = 1U << fault->bit; 

        if (fault->pol == 1)
            *(Input_t*)ptr_mesh_in_d[fault->col] |= mask;

        else
        {
            mask = ~mask;
            *(Input_t*)ptr_mesh_in_d[fault->col] &= mask;
        }
    }

    else
        pe[fault->row-1][fault->col]->stuckAtBitInD(fault->bit, fault->pol);
}


void Mesh::holdPermanentFaultOutC(uint8_t idx)
{
    Fault *fault = faultList[idx];
    pe[fault->row][fault->col]->stuckAtBitOutC(fault->bit, fault->pol);
}


void Mesh::holdPermanentFaultValid(uint8_t idx) 
{
    /*
        Note: example for DIM8: 
            even if the last PE is affected, e.g., PE77 (last row, last col), a permanent fault here will affect the whole column, 
            this is because the state register in PE77 is used during flush for all rows above this column. e.g., the outputs from the rows
            above flow through this defective register, so the whole thing gets corrupted. 
    */
    Fault *fault = faultList[idx];
    pe[fault->row][fault->col]->stuckAtValid(fault->bit, fault->pol);
}


void Mesh::holdPermanentFaultPropag(uint8_t idx)
{
    Fault *fault = faultList[idx];
    pe[fault->row][fault->col]->stuckAtPropB(fault->bit, fault->pol);
}
#endif


void Mesh::reset () // resets the ctrl signals and inputs only
{
    LOOP(i,DIM)
    {
        *(Input_t*)ptr_mesh_in_a[i] = 0;
        *(Input_t*)ptr_mesh_in_b[i] = 0;

        //*(Output_t*)ptr_mesh_out_b[i] = 0; // no. for ws this will erase the preloaded values.
        *(CData*)ptr_mesh_valid[i] = 0;
        *(CData*)ptr_mesh_propagate[i] = 0;
        
        //*(CData*)ptr_mesh_shift[i] = 0;

        LOOP(j,DIM) 
            pe[i][j]->reset();
    }
}

#endif
