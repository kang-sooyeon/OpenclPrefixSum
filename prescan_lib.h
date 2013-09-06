#ifndef PRESCAN_LIB_H
#define PRESCAN_LIB_H
#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>
#include <math.h>
#include <OpenCL/opencl.h>

#define DEBUG_INFO      (0)
int   GROUP_SIZE;
#define NUM_BANKS       (16)
#define MAX_ERROR       (1e-7)
#define SEPARATOR       ("----------------------------------------------------------------------\n")

#define min(A,B) ((A) < (B) ? (A) : (B))

static int count      = 16;

cl_device_id            ComputeDeviceId;
cl_command_queue        ComputeCommands;
cl_context              ComputeContext;
cl_program              ComputeProgram;
cl_kernel*              ComputeKernels;
cl_mem*                 ScanPartialSums;
unsigned int            ElementsAllocated;
unsigned int            LevelsAllocated;

enum KernelMethods {
  PRESCAN                             = 0,
  PRESCAN_STORE_SUM                   = 1,
  PRESCAN_STORE_SUM_NON_POWER_OF_TWO  = 2,
  PRESCAN_NON_POWER_OF_TWO            = 3,
  UNIFORM_ADD                         = 4
};

static const char* KernelNames[] = {
  "PreScanKernel",
  "PreScanStoreSumKernel",
  "PreScanStoreSumNonPowerOfTwoKernel",
  "PreScanNonPowerOfTwoKernel",
  "UniformAddKernel"
};

static const unsigned int KernelCount = sizeof(KernelNames) / sizeof(char *);

char * LoadProgramSourceFromFile(const char *filename);

int CreatePartialSumBuffers(unsigned int count);

void PreScanBuffer(
  cl_mem output_data,
  cl_mem input_data,
  unsigned int max_group_size,
  unsigned int max_work_item_count,
  unsigned int element_count
);

void ReleasePartialSums();
#endif
