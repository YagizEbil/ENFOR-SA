#ifndef GEMM_H
#define GEMM_H

#include "config.h"

#if EVAL_GEMMINI

#include <iostream>
#include <getopt.h>
#include <vector>
#include "verilated.h"
#include "Mesh.h"
#include "Context.h"
#include "utils.h"
#include "mat_utils.h"


/* evals to true if the PE is going to be assigned in iteration "iteration" of the systolic array */
#define WILL_PE_INPUT_BE_ASSIGNED(row,col,iteration)  (iteration == row + col + 0)
#define WILL_PE_OUTPUT_BE_ASSIGNED(row,col,iteration) (iteration == row + col + 1) // the output is written one cycle latter

#define PE_FIRST_CYCLE(i,j) ((i) + (j))
#define PE_LAST_CYCLE(i,j,STREAM_SIZE) ((i) + (j) + (STREAM_SIZE) - 1)

#define PE_ACTIVE(i,j,STREAM_SIZE,it) \
    ((it) >= PE_FIRST_CYCLE(i,j) && (it) <= PE_LAST_CYCLE(i,j,STREAM_SIZE))


Output_t Z[DIM][DIM]; // zero matrix


/* The systolic array computes C = A * B + D.
    - in OS mode: preloads D and propagates A and B 
    - in WS mode: preloads B and propagates A and D

     Check the file 'Gemmini-data-propagation-helper.txt' to see how Gemmini handles data proapagation control flow
*/

class Gemmini
{
public:
    Gemmini (VL_DUT *dut, const char *configId_)
    {
        context = new Context (dut, configId_, simOpt.faulty);
        mesh = new Mesh (dut);

    #ifdef USE_GL_INJECTION
        glMac = new MAC_UNIT();
    #endif
    }

    ~Gemmini()
    {
        printfinish();

        if (mesh != nullptr)
            delete mesh;

        if (context != nullptr)
            delete context;

    #if USE_GL_INJECTION
        if (glMac)
            delete glMac;
    #endif
    }

    uint32_t resetMesh();

#ifdef GEMM_OS
    uint32_t streamOS(
        const Input_t *inputA, 
        const Input_t *inputB, 
        uint16_t stream_size_delay_reg);

    uint32_t flushOS(Output_t *outputC, bool transpose_output=false);
    uint32_t preload(const Output_t *M);

#else // GEMM_WS
    uint32_t streamWS(
        const Input_t *inputA, 
        const Output_t *inputD, 
        Output_t *outputC,
        uint16_t stream_size_delay_reg);

    uint32_t preload(const Input_t *M);
#endif

    void addTransientFault(
        FaultModel fm, 
        int grp, 
        int row, 
        int col, 
        int bit,
        int ficycle,
        int cell, 
        bool silent) // set fault attributes for transient faults
    {
        mesh->addFaultToList(new Fault(fm, grp, row, col, bit, ficycle, cell, silent));
    }

    void addPermanentFault(
        FaultModel fm, 
        int grp, 
        int row, 
        int col, 
        int bit, 
        int pol, 
        int cell, 
        bool silent)  // set fault attributes for permanent faults
    {
        mesh->addFaultToList(new Fault(fm, grp, row, col, bit, 0, pol, cell, silent));
    }

    void addClientFaultToList(ClientFault fault) // used when Gemmini is used in a separate process (e.g., through main-server.cpp)
    {
        if (fault.faultModel == FM_TRANSIENT)
            addTransientFault(
                FaultModel::FM_TRANSIENT, 
                fault.target, 
                fault.row, 
                fault.col, 
                fault.bit,
                fault.ficycle,
                fault.cell, 
                fault.silent);
        
        else if (fault.faultModel == FM_PERMANENT)
            addPermanentFault(
                FaultModel::FM_PERMANENT, 
                fault.target, 
                fault.row, 
                fault.col, 
                fault.bit,
                fault.pol, 
                fault.cell, 
                fault.silent);
        
        else
             std::cerr << "[Server failed]: Invalid fault model = " << fault.faultModel << std::endl;
    }

    void clearFaultList()
    {
        mesh->clearFaultList();
    }


    uint32_t get_pe_active_rand_cycle(uint8_t row, uint8_t col, uint32_t streamSize) 
    {
        uint32_t first = PE_FIRST_CYCLE(row, col);
        uint32_t last  = PE_LAST_CYCLE(row, col, streamSize);

        uint32_t cycle = first + rand() % (last - first + 1);

        return cycle;
    }

        
    void debug_dump_c2(const char *msg)
    {
        printf ("%s\n", msg);
        LOOP(i,DIM)
        {
            LOOP(j,DIM)
                printf ("%d ", *(int*)mesh->pe[i][j]->ptr_c2);
            printf ("\n");
        }
    }

#ifdef ENABLE_HDFIT_VALIDATION // HDFIT was validated againts OS DIM8 only. ENABLE_HDFIT_VALIDATION must be defined in configs/<config header>
    void hdfit_prepare();
    void hdfit_inject(uint32_t it);
    uint32_t hdfit_assignment_offset;
#endif

    uint32_t cycle;
    uint32_t sizeFaultList;

#ifdef USE_GL_INJECTION
    MAC_UNIT *glMac;
#endif

    Mesh *mesh;
    Context *context;

    bool hasTransientFault = false; 
    bool hasPermanentFault = false;
};


// TODO: this should actuall call verilator->reset()
uint32_t Gemmini::resetMesh()
{
    debugprintf ("Gemmini: resetMesh()\n");

    /* I'm reseting Gemmini by preloading the zero matrix (we don't need to multiply zero by zero) */
    return preload(&Z[0][0]);
}


/* preload(): preloads the matrix to the systolic array internal PE states
    
    For C = AxB + D:
        OS: preloads the D matrix
        WS: preloads the B matrix
*/

#ifdef GEMM_OS
uint32_t Gemmini::preload(const Output_t *M)
#else
uint32_t Gemmini::preload(const Input_t *M) 
#endif
{
    debugprintf ("Gemmini: preload()\n");

    //for(uint8_t r = 0; r < DIM; r++)    
        //*mesh->ptr_mesh_dataflow[r] = 1;

    uint32_t steps = 0;

#if 0 // Magic preload for faster injection during computation. for FIs that do not consider the effects of preloads this trick can be used
    LOOP(i,DIM)
        LOOP(j,DIM)
        {
        #ifdef GEMM_OS
            *(IData*)mesh->pe[i][j]->ptr_c1 = M[i*DIM + j];
        #else
            *(IData*)mesh->pe[i][j]->ptr_c2 = M[i*DIM + j];
        #endif
        }
    return 2*DIM; // we have  LOOP(i, DIM) steps++; twice below
#endif

#if 0 // debug only
    printf ("C1 before preloading:\n");
    LOOP(i,DIM)
    {
        LOOP(j,DIM)
            printf ("%d ", *(int*)mesh->pe[i][j]->ptr_c1);
        printf ("\n");
    }
#endif
    
    LOOP(i, DIM)
    {
        *mesh->ptr_mesh_valid[i] = 1;
    #ifdef GEMM_OS
        *mesh->ptr_mesh_propagate[i] = 1;
    #else
        *mesh->ptr_mesh_propagate[i] = 0;  // WS: store weights in C2
    #endif
    }

    LOOP(i, DIM) // propagates the inputs as in b_x_y <- in_d. one cycle for the input to reach b_16_0 + one cycle to reach c1
    {   
        for(uint8_t c = 0; c < DIM; c++)
            *mesh->ptr_mesh_in_d[c] = M[(DIM-1-i)*DIM + c];  // as seen in PE_.sv: the weights stored in c1/c2 come from in_d register

    #ifdef ENABLE_PERMANENT_FAULTS  // defined in FaultList.h
        if (hasPermanentFault)
            mesh->holdPermanentFaults();
    #endif
        cycle = context->step();
        steps++;
    }

    LOOP(i, DIM)
    {
        *mesh->ptr_mesh_valid[i] = 0;
        *mesh->ptr_mesh_propagate[i] = 0;
    }

    LOOP(i, DIM)  // propagates from the b_x_y registers to the C1 register (OS still works w/o this(?)...)
    {
    #ifdef ENABLE_PERMANENT_FAULTS
        if (hasPermanentFault)
            mesh->holdPermanentFaults();
    #endif
        cycle = context->step(); 
        steps++;
    }

#if 0 // debug only
    debug_dump_c2("C2 after preloading");
#endif

    return steps;
}


#ifdef GEMM_OS

/* OS mode only: flushes the PE's accumulators to and output matix */
uint32_t Gemmini::flushOS(Output_t *outputC, bool transpose_output)
{
    debugprintf ("Gemmini: flushOS()\n");
    uint32_t steps = 0;

    bool isFaultyMode = sizeFaultList > 0;

    LOOP(i, DIM)
    {
        *mesh->ptr_mesh_valid[i] = 1;
        *mesh->ptr_mesh_propagate[i] = 1;
    }

    LOOP(i, DIM) 
    {
    #ifdef ENABLE_PERMANENT_FAULTS  // defined in FaultList.h
        if (hasPermanentFault)
            mesh->holdPermanentFaults();
    #endif

        cycle = context->step();
        steps++;
    }

    LOOP(i, DIM)
    {
        *mesh->ptr_mesh_valid[i] = 0;
        *mesh->ptr_mesh_propagate[i] = 0;
    }

    LOOP(i, DIM)
    {
    #ifdef ENABLE_PERMANENT_FAULTS
        if (hasPermanentFault) // removed for perf
           mesh->holdPermanentFaults();
    #endif

        cycle = context->step(); 
        steps++;

        /*  I'm transposing the output directly in "hw" here, if required with transpose_output=true. for perf only */
        if (!transpose_output)
            LOOP (j, DIM)
                outputC[(DIM-1-i)*DIM + j] = *mesh->ptr_mesh_out_c[j]; // the last rows are flushed first
  
        else
            LOOP (j, DIM)
                outputC[j*DIM + DIM-1-i] = *mesh->ptr_mesh_out_c[j];
    }

    return steps;
}


/* 
    OS Gemmini implementation. This propagates the matrices A and B through the array. If D is not zero, it must be preloaded before this computation 
    TODO: this can be generalized to any size 'd' in the form: inputA[DIM][d] and inputB[DIM][d])
*/
uint32_t Gemmini::streamOS(
        const Input_t *inputA, 
        const Input_t *inputB, 
        uint16_t stream_size_delay_reg)  // the number of columns in inputA (== number of rows in inputB)
{
    debugprintf ("Gemmini: streamOS()\n");

    uint32_t it = 0;
    bool finished = false;

    hasTransientFault = false; 
    hasPermanentFault = false;
    sizeFaultList = mesh->faultList.size();

    bool isFaultyMode = sizeFaultList > 0;

    if (sizeFaultList > 0)
    {
        hasTransientFault = mesh->faultList[0]->faultModel == FaultModel::FM_TRANSIENT;
        hasPermanentFault = mesh->faultList[0]->faultModel == FaultModel::FM_PERMANENT;
    }

    Fault *theTransientFault = nullptr;

    if (hasTransientFault) // assuming a single transient fault in the fault list
        theTransientFault = mesh->faultList[0];

    //mesh->reset();
    
#ifdef ENABLE_HDFIT_VALIDATION // HDFIT was validated againts OS DIM8 only. ENABLE_HDFIT_VALIDATION must be defined in configs/<config header>
    /*
        - This is a test to validate our fault injection method against HDFIT
        - the validation is done against a OS DIM8 config only.
        - To enable HDFIT injections, we need set the HDFIT variables (GlobalFiModInstNr, GlobalFiNumber and GlobalFiSignal) 
            according to the fault position and targets
        - See the file /rtl/lib/Gemmini/HDFIT-fault-map-os-DIM8.txt, which is a sketch showing how this was done
    */
    if (sizeFaultList > 0)
        hdfit_prepare();
#endif

    uint32_t timeout_iters = 10*stream_size_delay_reg;

    while (!finished && it < timeout_iters) // TIMEOUT_ITERS: 15000
    {   
        if (it < stream_size_delay_reg)
        {
            for(uint8_t r = 0; r < DIM; r++)
            {
                *mesh->ptr_mesh_in_a[r] = inputA[r*stream_size_delay_reg + it];
                *mesh->ptr_mesh_in_b[r] = inputB[r*stream_size_delay_reg + it];
            }
        }

        else if (it == stream_size_delay_reg) // set the mesh inputs to zero once all inputs were streamed
        {
            for(uint8_t r = 0; r < DIM; r++)
            {
                *mesh->ptr_mesh_in_a[r] = 0;
                *mesh->ptr_mesh_in_b[r] = 0;
                *mesh->ptr_mesh_valid[r] = 0;
                *mesh->ptr_mesh_propagate[r] = 0;
            }
        }

        if (it < DIM) // asserts the next input as valid
        {
            *mesh->ptr_mesh_valid[it] = 1;
            *mesh->ptr_mesh_propagate[it] = 0;
            //*mesh->ptr_mesh_shift[it] = 1; // this is only used when propag changes values (see PE.scala and PE.sv). therefore there's no need to reset this to zero for the same matmul
        }

    #ifndef USE_GL_INJECTION // RTL: faults are injected here. GL: faults are injected after the MAC operation by replacing the current MACs output with an output computed at GL 
        if (hasTransientFault)
        {
        /* Validation against HDFIT. HDFIT was validated againts OS DIM8 only */
        #ifdef ENABLE_HDFIT_VALIDATION // if using HDFIT instrumentation, this is how we enable the fault. We must not call mesh->cbApplyTransientFault
            hdfit_inject(it);
        #else 
            if (!theTransientFault->performed)
            {
                //if (WILL_PE_INPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it))
                //if (WILL_PE_OUTPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it)) // HDIFT val
                if (theTransientFault->ficycle == it)
                {
                    if (!theTransientFault->silent)
                        theTransientFault->print();

                    mesh->cbApplyTransientFault[theTransientFault->group](0);
                    theTransientFault->performed = true;
                }
            }
        #endif
        }
        
    #ifdef ENABLE_PERMANENT_FAULTS
        else 
        {
            for (uint32_t i = 0; i < sizeFaultList; i++) // can have multiple permament faults
            {
                Fault *fault = mesh->faultList[i];

                if (fault->faultModel == FaultModel::FM_PERMANENT)
                {
                    if (!fault->silent)
                       fault->print();
                    mesh->cbHoldPermanentFault[fault->group](i);
                }
            }
        }
    #endif // ENABLE_PERMANENT_FAULTS
    #endif // ifndef USE_GL_INJECTION

    #ifdef USE_GL_INJECTION // OS
        /* GL injections: 
            1. gather the inputs and accumulator state before the next step (this is what the MAC operation is going to use in the next opperation)
            2. run the RTL step
            3. compute the same MAC operation at GL (same inputs, same accumulator state)
            4. replace the Mesh PE output (RTL) with the GL one */
        if (hasTransientFault && !theTransientFault->performed)
        {
            if (WILL_PE_OUTPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it))
            {
                /* pass the inputs to be used in the next MAC computation to the GL mac unit */
                //glMac->io_in_a = mesh->pe[theTransientFault->row][theTransientFault->col]->getInputA(); // old version. using mediumosconfig_old_pointers.cc
                //glMac->io_in_b = mesh->pe[theTransientFault->row][theTransientFault->col]->getInputB(); // old version. using mediumosconfig_old_pointers.cc 
                
                glMac->io_in_a = mesh->getPrevInputA(theTransientFault->row, theTransientFault->col);
                glMac->io_in_b = mesh->getPrevInputB(theTransientFault->row, theTransientFault->col);
                glMac->io_in_c = mesh->pe[theTransientFault->row][theTransientFault->col]->getOutput();
            }
        }
    #endif

        cycle = context->step();   // runs a single-cycle DUT step. note that when this executes, the inputs also change again (this is why i saved them before)

    #ifdef USE_GL_INJECTION // OS
        if (hasTransientFault && !theTransientFault->performed)
        {   
            if (WILL_PE_OUTPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it))
            {
                glMac->GlobalFiModInstNr[0] = 1;   // asserts fiEnable
                //glMac->GlobalFiModInstNr[0] = 0; // [Debug only] disables FI. sanity check ok - this leads to no output errors
                glMac->GlobalFiSignal = 1;         // aseerts the fault mask. i think it must be a single bit always
                glMac->GlobalFiNumber = theTransientFault->cell; // points to the target cell
                
                /* eval the GL mac unit */
                glMac->eval();
            
                /* gets the faulty output of the GL MAC unit */
                Output_t glMacOut = glMac->io_out_d; 
           
            #if 0 // [Debug only]
                Output_t goldMesh = mesh->getMacOut(); // gets the output of the PE stored in faultList[0]
                Output_t goldHost = glMac->io_in_a*glMac->io_in_b + glMac->io_in_c; // golden MAC operation
                
                //assert(goldHost == goldMesh); // ok. passed
                //assert(glMacOut == goldHost);

                /* counts the number of bits flipped in the MAC output w.r.t the golden output */
                //uint8_t flippedBits = popcount(glMacOut^goldMesh); // TODO: check also the pattern in which the bits are flipped
                //printf("Flipped bits: %u\n", flippedBits);
            #endif
                /* replace the RTL mac uint output with the output computed at GL */
             
                mesh->setMacOut(glMacOut);
                //mesh->setMacOut(goldMesh); // [Debug only]. sanity check ok - this leads to no output errors
                theTransientFault->performed = true;
            }
        }
    #endif

    #ifdef ENABLE_HDFIT_VALIDATION
        finished = it == stream_size_delay_reg + 1;
    #else
        finished = it == stream_size_delay_reg;
    #endif

        it++;
    }

#ifdef ENABLE_PERMANENT_FAULTS
    if (hasPermanentFault)
        mesh->holdPermanentFaults();
#endif

    return it;
}

#endif // #ifdef GEMM_OS



#ifdef GEMM_WS

/*  WS: C = A.B + D
    - B must have been preloaded before
    - flow A and D through the array
    - if there's no bias D matrix, inputD is the zero matrix */

uint32_t Gemmini::streamWS(
        const Input_t *inputA, 
        const Output_t *inputD, 
        Output_t *outputC,
        uint16_t stream_size_delay_reg)
{    
    debugprintf ("Gemmini: streamWS()\n");

    uint32_t countPeOps[DIM][DIM];  // counts the number of operations per PE (should be DIM). The simulation finishes when the last PE (PE[DIM-1][DIM-1]) operates DIM times
    uint32_t it = 0;
    bool finished = false;
    
    bzero(countPeOps, sizeof(uint32_t)*DIM*DIM);

    hasTransientFault = false; 
    hasPermanentFault = false;
    sizeFaultList = mesh->faultList.size();

    bool isFaultyMode = sizeFaultList > 0;

    if (sizeFaultList > 0)
    {
        hasTransientFault = mesh->faultList[0]->faultModel == FaultModel::FM_TRANSIENT;
        hasPermanentFault = mesh->faultList[0]->faultModel == FaultModel::FM_PERMANENT;
    }

    Fault *theTransientFault = nullptr;

    if (hasTransientFault) // assuming a single transient fault in the fault list
        theTransientFault = mesh->faultList[0];

    mesh->reset();

    uint32_t timeout_iters = 10*stream_size_delay_reg;
    char msg[100];

    while (!finished && it < timeout_iters)
    {  
        /* Writes a whole column of matrix A to the Mesh io_in_a_x_y input pins */
        if (it < stream_size_delay_reg)
        {
            for(uint8_t r = 0; r < DIM; r++)
            {
                *mesh->ptr_mesh_in_a[r] = inputA[r*stream_size_delay_reg + it];
                *mesh->ptr_mesh_in_b[r] = inputD[r*stream_size_delay_reg + it]; // in WS we must feed the bias D matrix to in_b (check the Verilog to see why). once the D was already preloaded (in_d), the verilog code only cares that the two matrices to be multiplied are fed in in_a and in_b
            }
        }

        else if (it == stream_size_delay_reg) // set the mesh inputs to zero once all inputs were streamed
        {
            for(uint8_t r = 0; r < DIM; r++)
            {
                *mesh->ptr_mesh_in_a[r] = 0; // these are mesh input signals. it's ok to write them directly here, 
                *mesh->ptr_mesh_in_b[r] = 0; // the same way we write inputs, even if we inject the internal mesh ctrl signals
            }
        }

        if (it < DIM)
        {
            *mesh->ptr_mesh_valid[it] = 1;
            *mesh->ptr_mesh_propagate[it] = 1;
        }

    #ifndef USE_GL_INJECTION // RTL: faults are injected here. GL: faults are injected after the MAC operation by replacing the current MACs output with an output computed at GL 
        if (hasTransientFault)
        {
            if (!theTransientFault->performed)
            {
                //if (WILL_PE_INPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it))
                //if (WILL_PE_OUTPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it))
                if (theTransientFault->ficycle == it)
                {
                    if (!theTransientFault->silent)
                        theTransientFault->print();

                    mesh->cbApplyTransientFault[theTransientFault->group](0);
                    theTransientFault->performed = true;
                }
            }
        }
        
    #ifdef ENABLE_PERMANENT_FAULTS
        else 
        {
            for (uint32_t i = 0; i < sizeFaultList; i++) // can have multiple permament faults
            {
                Fault *fault = mesh->faultList[i];

                if (fault->faultModel == FaultModel::FM_PERMANENT)
                {
                    if (!fault->silent)
                       fault->print();
                    mesh->cbHoldPermanentFault[fault->group](i);
                }
            }
        }
    #endif // ENABLE_PERMANENT_FAULTS
    #endif // ifndef USE_GL_INJECTION

    #ifdef USE_GL_INJECTION  // WS
         /* GL injections: 
            1. gather the inputs and accumulator state before the next step (this is what the MAC operation is going to use in the next opperation)
            2. run the RTL step
            3. compute the same MAC operation at GL (same inputs, same accumulator state)
            4. replace the Mesh PE output (RTL) with the GL one */
        if (hasTransientFault && !theTransientFault->performed)
        {
            //if (WILL_PE_OUTPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it))
            if (theTransientFault->ficycle == it)
            {
                /* pass the inputs to be used in the next MAC computation to the GL mac unit */
                glMac->io_in_a = mesh->pe[theTransientFault->row][theTransientFault->col]->getInputA();

                /*  
                    in WS, the weights are stored in C1
                    ptr_mesh_in_b is actually transporting the D matrix

                    the first row starts with the D row
                    the rows below receive the partial sum from the PE above
                */

                glMac->io_in_b = mesh->pe[theTransientFault->row][theTransientFault->col]->getC2();

                if (theTransientFault->row == 0) // first row. the accumulator starts with the bias D matrix
                    glMac->io_in_c = mesh->pe[theTransientFault->row][theTransientFault->col]->getInputB(); // WS stream_bias
                    //glMac->io_in_c = 0; // WS stream

                else // glMac->io_in_c has to receive the out_b (partial sum) of the PE above
                    glMac->io_in_c = mesh->pe[theTransientFault->row-1][theTransientFault->col]->getOutput();
            }
        }
    #endif

        cycle = context->step();   // runs a single-cycle DUT step. note that when this executes, the inputs also change again (this is why i saved them before)
    
    #if 0
        sprintf(msg, "C2 in cycle %d", it);
        debug_dump_c2((const char*)msg);
        printf("\n");
    #endif

    #ifdef USE_GL_INJECTION // WS
        if (hasTransientFault && !theTransientFault->performed)
        {   
            //if (WILL_PE_OUTPUT_BE_ASSIGNED(theTransientFault->row, theTransientFault->col, it))
            if (theTransientFault->ficycle == it)
            {
                glMac->GlobalFiModInstNr[0] = 1;    // asserts fiEnable
                //glMac->GlobalFiModInstNr[0] = 0;  // [Debug only] disables FI. sanity check ok - this leads to no output errors
                glMac->GlobalFiSignal = 1;          // aseerts the fault mask. i think it must be a single bit always
                glMac->GlobalFiNumber = theTransientFault->cell; // points to the target cell
                
                /* eval the GL mac unit */
                glMac->eval();
            
                /* gets the faulty output of the GL MAC unit */
                Output_t glMacOut = glMac->io_out_d; 
           
            #if 0 // [Debug only]
                Output_t goldMesh = mesh->getMacOut(); // gets the output of the PE stored in faultList[0]
                Output_t goldHost = glMac->io_in_a*glMac->io_in_b + glMac->io_in_c; // golden MAC operation
                //printf("Gold: %d  glMacOut: %d  computes: %d\n", goldMesh, glMacOut, goldHost);

                assert(goldHost == goldMesh); // ok. passed

                /* counts the number of bits flipped in the MAC output w.r.t the golden output */
                //uint8_t flippedBits = popcount(glMacOut^goldMesh); // TODO: check also the pattern in which the bits are flipped
                //printf("Flipped bits: %u\n", flippedBits);
            #endif
                /* replace the RTL mac uint output with the output computed at GL */
                mesh->setMacOut(glMacOut);
                //mesh->setMacOut(goldMesh); // [Debug only]. sanity check ok - this leads to no output errors
                theTransientFault->performed = true;
            }
        }
    #endif

        LOOP(i,DIM) // stores each of the PEs outputs, if available, and checks for end of simulation
        {
            LOOP(j,DIM)
            {
                if (it == PE_LAST_CYCLE(i, j, DIM) + 1) // if it == one extra cycle after the last cycle. + 1 to propagate the pe out reg to the mesh reg (which is linked to the mesh out wire)
                    outputC[j*DIM + i] = *mesh->ptr_mesh_out_b[i];  // ptr_mesh_out_b[i] outputts per row, one row per cycle: C1i, C2i, C3i, ...

                #if 0 // old way. not used anymore
                      // i'm not using ptr_propagate and ptr_valid anymore as we inject them, 
                      // thus preventing from reading the output in the right moment
                      // this was causing trouble when injectin in ptr_propagate, where a full row and a full column would appear faulty
                      // when in reality, only a single colum should be affected (now multiple rows)

                countPeOps[i][j] += *(CData*)mesh->pe[i][j]->ptr_propagate; // When hitting the ptr_propagate signals, countPeOps may not increase, causing a timeout
                //countPeOps[i][j] += *(CData*)mesh->pe[i][j]->ptr_valid; // same as above
                if (countPeOps[i][j] == DIM+1) // DIM+1: one extra cycle so that the output appears in the Mesh output
                    outputC[j*DIM + i] = *mesh->ptr_mesh_out_b[i];// ptr_mesh_out_b[i] outputs per row: C1i, C2i, C3i, ...
                #endif
            }
        }

        finished = countPeOps[DIM-1][DIM-1] == DIM+1; // if the last PE computed the last sum, the matmul is finished
        it++;
    }

#if 0
    LOOP(i,DIM)
    {
        *mesh->ptr_mesh_valid[i] = 0;
        *mesh->ptr_mesh_propagate[i] = 0;
    }

    LOOP(i,DIM)
        cycle = context->step();
#endif

    return it;
}

#endif // #ifdef GEMM_WS



/* HDFIT was validated againts OS DIM8 only.  ENABLE_HDFIT_VALIDATION must be defined in configs/<config header> */

#ifdef ENABLE_HDFIT_VALIDATION

void Gemmini::hdfit_inject(uint32_t it)
{
    bool willPropag;

    if (mesh->faultList[0]->group == IDX_valid or mesh->faultList[0]->group == IDX_propagate)
        willPropag =  WILL_PE_INPUT_BE_ASSIGNED(mesh->faultList[0]->row, mesh->faultList[0]->col, it);
    else
        willPropag = WILL_PE_OUTPUT_BE_ASSIGNED(mesh->faultList[0]->row, mesh->faultList[0]->col, it);

    if (willPropag)
    {
        mesh->dut->GlobalFiSignal = 1U << mesh->faultList[0]->bit; // sets the HDFIT (hw-based) fault mask

        if (mesh->faultList[0]->group == IDX_io_out_c) // enables FI for a specific PE
        { 
            mesh->dut->GlobalFiModInstNr[0] = 1; // (to enable in the Mesh level)
            mesh->dut->GlobalFiModInstNr[1] = hdfit_assignment_offset + mesh->faultList[0]->row*DIM + mesh->faultList[0]->col;// (pass fiEnable to Tile tile level for PE 00)
            mesh->dut->GlobalFiModInstNr[2] = 634;// (pass fiEnable to PE)
            mesh->dut->GlobalFiModInstNr[3] = 633;// (pass fiEnable to MAC)
        }
        else // enables FI at the mesh level only
            mesh->dut->GlobalFiModInstNr[0] = 1; // enables the HDFIT's fiEnable signal in the Mesh-level only (=1)
    }
    else  // disables the HDFIT's fiEnable signal
    {
        mesh->dut->GlobalFiModInstNr[0] = 0;
        mesh->dut->GlobalFiModInstNr[1] = 0;
        mesh->dut->GlobalFiModInstNr[2] = 0;
        mesh->dut->GlobalFiModInstNr[3] = 0;
    }
}


void Gemmini::hdfit_prepare() // hard-coded for OS DIM8 only
{
    hdfit_assignment_offset = 0;

    if (mesh->faultList[0]->group == IDX_io_in_a)
    {
        hdfit_assignment_offset = 17;
        mesh->dut->GlobalFiNumber = hdfit_assignment_offset + mesh->faultList[0]->row*DIM + mesh->faultList[0]->col;
    }

    else if (mesh->faultList[0]->group == IDX_io_in_b)
    {
        hdfit_assignment_offset = 81;
        mesh->dut->GlobalFiNumber = hdfit_assignment_offset + mesh->faultList[0]->row*5 + mesh->faultList[0]->col*40;
    }

    else if (mesh->faultList[0]->group == IDX_io_out_c)
    {
        hdfit_assignment_offset = 635;
        mesh->dut->GlobalFiNumber = 2;
    }

    else if (mesh->faultList[0]->group == IDX_propagate)
    {
        hdfit_assignment_offset = 85;
        mesh->dut->GlobalFiNumber = hdfit_assignment_offset + mesh->faultList[0]->row*5 + mesh->faultList[0]->col*40;
    }

    else if (mesh->faultList[0]->group == IDX_valid)
    {
        hdfit_assignment_offset = 401;
        mesh->dut->GlobalFiNumber = hdfit_assignment_offset + mesh->faultList[0]->row + mesh->faultList[0]->col*DIM;
    }
}

#endif // ENABLE_HDFIT_VALIDATION

#endif // #if EVAL_GEMMINI
#endif
