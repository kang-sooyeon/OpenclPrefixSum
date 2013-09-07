#include "prescan.h"

int GROUP_SIZE = 256;
unsigned int ElementsAllocated = 0;
unsigned int LevelsAllocated = 0;
cl_mem* ScanPartialSums = NULL;
cl_kernel* ComputeKernels = NULL;
cl_program ComputeProgram = NULL;

char * LoadProgramSourceFromFile(const char *filename) {
  struct stat statbuf;
  FILE        *fh;
  char        *source;

  fh = fopen(filename, "r");
  if (fh == 0) return 0;

  stat(filename, &statbuf);
  source = (char *) malloc(statbuf.st_size + 1);
  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0';

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool IsPowerOfTwo(int n) {
  return ((n&(n-1))==0) ;
}

int floorPow2(int n) {
  int exp;
  frexp((float)n, &exp);
  return 1 << (exp - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int CreatePartialSumBuffers(unsigned int count, cl_context context) {
  ElementsAllocated = count;

  unsigned int group_size = GROUP_SIZE;
  unsigned int element_count = count;

  int level = 0;

  do {
    unsigned int group_count = (int)fmax(1, (int)ceil((float)element_count / (2.0f * group_size)));
    if (group_count > 1) {
      level++;
    }
    element_count = group_count;
  } while (element_count > 1);

  LevelsAllocated = level;
  ScanPartialSums = (cl_mem*) calloc(level, sizeof(cl_mem));

  element_count = count;
  level = 0;

  do {
    unsigned int group_count = (int)fmax(1, (int)ceil((float)element_count / (2.0f * group_size)));
    if (group_count > 1) {
      size_t buffer_size = group_count * sizeof(int);
      ScanPartialSums[level++] = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    }

    element_count = group_count;

  } while (element_count > 1);

  return CL_SUCCESS;
}

void ReleasePartialSums(void) {
  unsigned int i;
  for (i = 0; i < LevelsAllocated; i++) clReleaseMemObject(ScanPartialSums[i]);

  free(ScanPartialSums);
  ScanPartialSums = 0;
  ElementsAllocated = 0;
  LevelsAllocated = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int PreScan(
  cl_command_queue queue,
  size_t *global,
  size_t *local,
  size_t shared,
  cl_mem output_data,
  cl_mem input_data,
  unsigned int n,
  int group_index,
  int base_index
) {
#if DEBUG_INFO
  printf("PreScan: Global[%4d] Local[%4d] Shared[%4d] BlockIndex[%4d] BaseIndex[%4d] Entries[%d]\n",
    (int)global[0], (int)local[0], (int)shared, group_index, base_index, n);
#endif

  unsigned int k = PRESCAN;
  unsigned int a = 0;

  int err = CL_SUCCESS;
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &output_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &input_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, shared,         0);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &group_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &base_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &n);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to set kernel arguments!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  err = CL_SUCCESS;
  err |= clEnqueueNDRangeKernel(queue, ComputeKernels[k], 1, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to execute kernel!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  return CL_SUCCESS;
}

int PreScanStoreSum(
  cl_command_queue queue,
  size_t *global,
  size_t *local,
  size_t shared,
  cl_mem output_data,
  cl_mem input_data,
  cl_mem partial_sums,
  unsigned int n,
  int group_index,
  int base_index
) {
#if DEBUG_INFO
  printf("PreScan: Global[%4d] Local[%4d] Shared[%4d] BlockIndex[%4d] BaseIndex[%4d] Entries[%d]\n", 
    (int)global[0], (int)local[0], (int)shared, group_index, base_index, n);
#endif

  unsigned int k = PRESCAN_STORE_SUM;
  unsigned int a = 0;

  int err = CL_SUCCESS;
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &output_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &input_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &partial_sums);
  err |= clSetKernelArg(ComputeKernels[k],  a++, shared,         0);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &group_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &base_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &n);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to set kernel arguments!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  err = CL_SUCCESS;
  err |= clEnqueueNDRangeKernel(queue, ComputeKernels[k], 1, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to execute kernel!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  return CL_SUCCESS;
}

int PreScanStoreSumNonPowerOfTwo(
  cl_command_queue queue,
  size_t *global,
  size_t *local,
  size_t shared,
  cl_mem output_data,
  cl_mem input_data,
  cl_mem partial_sums,
  unsigned int n,
  int group_index,
  int base_index
) {
#if DEBUG_INFO
  printf("PreScanStoreSumNonPowerOfTwo: Global[%4d] Local[%4d] BlockIndex[%4d] BaseIndex[%4d] Entries[%d]\n",
    (int)global[0], (int)local[0], group_index, base_index, n);
#endif

  unsigned int k = PRESCAN_STORE_SUM_NON_POWER_OF_TWO;
  unsigned int a = 0;

  int err = CL_SUCCESS;
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &output_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &input_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &partial_sums);
  err |= clSetKernelArg(ComputeKernels[k],  a++, shared,         0);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &group_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &base_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &n);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to set kernel arguments!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  err = CL_SUCCESS;
  err |= clEnqueueNDRangeKernel(queue, ComputeKernels[k], 1, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to execute kernel!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  return CL_SUCCESS;
}

int PreScanNonPowerOfTwo(
  cl_command_queue queue,
  size_t *global,
  size_t *local,
  size_t shared,
  cl_mem output_data,
  cl_mem input_data,
  unsigned int n,
  int group_index,
  int base_index
) {
#if DEBUG_INFO
  printf("PreScanNonPowerOfTwo: Global[%4d] Local[%4d] BlockIndex[%4d] BaseIndex[%4d] Entries[%d]\n",
    (int)global[0], (int)local[0], group_index, base_index, n);
#endif

  unsigned int k = PRESCAN_NON_POWER_OF_TWO;
  unsigned int a = 0;

  int err = CL_SUCCESS;
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &output_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &input_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, shared,         0);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &group_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &base_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &n);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to set kernel arguments!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  err = CL_SUCCESS;
  err |= clEnqueueNDRangeKernel(queue, ComputeKernels[k], 1, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to execute kernel!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }
  return CL_SUCCESS;
}

int UniformAdd(
  cl_command_queue queue,
  size_t *global,
  size_t *local,
  cl_mem output_data,
  cl_mem partial_sums,
  unsigned int n,
  unsigned int group_offset,
  unsigned int base_index
) {
#if DEBUG_INFO
  printf("UniformAdd: Global[%4d] Local[%4d] BlockOffset[%4d] BaseIndex[%4d] Entries[%d]\n",
    (int)global[0], (int)local[0], group_offset, base_index, n);
#endif

  unsigned int k = UNIFORM_ADD;
  unsigned int a = 0;

  int err = CL_SUCCESS;
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &output_data);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_mem), &partial_sums);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(int),  0);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &group_offset);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &base_index);
  err |= clSetKernelArg(ComputeKernels[k],  a++, sizeof(cl_int), &n);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to set kernel arguments!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  err = CL_SUCCESS;
  err |= clEnqueueNDRangeKernel(queue, ComputeKernels[k], 1, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: %s: Failed to execute kernel!\n", KernelNames[k]);
    return EXIT_FAILURE;
  }

  return CL_SUCCESS;
}

int PreScanBufferRecursive(
  cl_command_queue queue,
  cl_mem output_data,
  cl_mem input_data,
  int max_group_size,
  int max_work_item_count,
  int element_count,
  int level
) {
  unsigned int group_size = max_group_size;
  unsigned int group_count = (int)fmax(1.0f, (int)ceil((float)element_count / (2.0f * group_size)));
  unsigned int work_item_count = 0;

  if (group_count > 1) work_item_count = group_size;
  else if (IsPowerOfTwo(element_count)) work_item_count = element_count / 2;
  else work_item_count = floorPow2(element_count);

  work_item_count = (work_item_count > max_work_item_count) ? max_work_item_count : work_item_count;

  unsigned int element_count_per_group = work_item_count * 2;
  unsigned int last_group_element_count = element_count - (group_count-1) * element_count_per_group;
  unsigned int remaining_work_item_count = (int)fmax(1.0f, last_group_element_count / 2);
  remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
  unsigned int remainder = 0;
  size_t last_shared = 0;

  if (last_group_element_count != element_count_per_group) {
    remainder = 1;

    if (!IsPowerOfTwo(last_group_element_count)) remaining_work_item_count = floorPow2(last_group_element_count);

    remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
    unsigned int padding = (2 * remaining_work_item_count) / NUM_BANKS;
    last_shared = sizeof(float) * (2 * remaining_work_item_count + padding);
  }

  remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
  size_t global[] = { (int)fmax(1, group_count - remainder) * work_item_count, 1 };
  size_t local[]  = { work_item_count, 1 };

  unsigned int padding = element_count_per_group / NUM_BANKS;
  size_t shared = sizeof(float) * (element_count_per_group + padding);

  cl_mem partial_sums = ScanPartialSums[level];
  int err = CL_SUCCESS;

  if (group_count > 1) {
    err = PreScanStoreSum(queue, global, local, shared, output_data, input_data, partial_sums, work_item_count * 2, 0, 0);
    if(err != CL_SUCCESS) return err;

    if (remainder) {
      size_t last_global[] = { 1 * remaining_work_item_count, 1 };
      size_t last_local[]  = { remaining_work_item_count, 1 };

      err = PreScanStoreSumNonPowerOfTwo(
        queue,
        last_global, last_local, last_shared,
        output_data, input_data, partial_sums,
        last_group_element_count,
        group_count - 1,
        element_count - last_group_element_count
      );

      if(err != CL_SUCCESS) return err;
    }

    err = PreScanBufferRecursive(queue, partial_sums, partial_sums, max_group_size, max_work_item_count, group_count, level + 1);
    if(err != CL_SUCCESS) return err;

    err = UniformAdd(queue, global, local, output_data, partial_sums,  element_count - last_group_element_count, 0, 0);
    if(err != CL_SUCCESS) return err;

    if (remainder) {
      size_t last_global[] = { 1 * remaining_work_item_count, 1 };
      size_t last_local[]  = { remaining_work_item_count, 1 };

      err = UniformAdd(
        queue,
        last_global, last_local,
        output_data, partial_sums,
        last_group_element_count,
        group_count - 1,
        element_count - last_group_element_count
      );

      if(err != CL_SUCCESS) return err;
    }
  } else if (IsPowerOfTwo(element_count)) {
    err = PreScan(queue, global, local, shared, output_data, input_data, work_item_count * 2, 0, 0);
    if(err != CL_SUCCESS) return err;
  } else {
    err = PreScanNonPowerOfTwo(queue, global, local, shared, output_data, input_data, element_count, 0, 0);
    if(err != CL_SUCCESS) return err;
  }

  return CL_SUCCESS;
}

void PreScanBuffer(
  cl_command_queue queue,
  cl_mem output_data,
  cl_mem input_data,
  unsigned int max_group_size,
  unsigned int max_work_item_count,
  unsigned int element_count
) {
  PreScanBufferRecursive(queue, output_data, input_data, max_group_size, max_work_item_count, element_count, 0);
}


