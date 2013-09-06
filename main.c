#include "prescan.h"

void ScanReference( int* reference, int* input, const unsigned int count) {
  reference[0] = 0;
  double total_sum = 0;

  unsigned int i = 1;
  for( i = 1; i < count; ++i) {
    total_sum += input[i-1];
    reference[i] = input[i-1] + reference[i-1];
  }

  if (total_sum != reference[count-1]) printf("Warning: Exceeding single-precision accuracy.  Scan will be inaccurate.\n");
}

int main(int argc, char **argv) {
  int i;
  uint64_t         t0 = 0;
  uint64_t         t1 = 0;
  uint64_t         t2 = 0;
  int              err = 0;
  cl_mem       output_buffer;
  cl_mem           input_buffer;

  // Create some random input data on the host
  int *int_data = (int*)calloc(count, sizeof(int));
  for (i = 0; i < count; i++) int_data[i] = i;

  // Connect to a CPU compute device
  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &ComputeDeviceId, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to locate a compute device!\n");
    return EXIT_FAILURE;
  }

  size_t returned_size = 0;
  size_t max_workgroup_size = 0;
  err = clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, &returned_size);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to retrieve device info!\n");
    return EXIT_FAILURE;
  }

  GROUP_SIZE = min( GROUP_SIZE, max_workgroup_size );

  cl_char vendor_name[1024] = {0};
  cl_char device_name[1024] = {0};
  err = clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
  err|= clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to retrieve device info!\n");
    return EXIT_FAILURE;
  }

  printf(SEPARATOR);
  printf("Connecting to %s %s...\n", vendor_name, device_name);

  // Load the compute program from disk into a cstring buffer
  printf(SEPARATOR);
  const char* filename = "./scan_kernel.cl";
  printf("Loading program '%s'...\n", filename);
  printf(SEPARATOR);

  char *source = LoadProgramSourceFromFile(filename);
  if(!source) {
    printf("Error: Failed to load compute program from file!\n");
    return EXIT_FAILURE;
  }

  // Create a compute ComputeContext
  ComputeContext = clCreateContext(0, 1, &ComputeDeviceId, NULL, NULL, &err);
  if (!ComputeContext) {
    printf("Error: Failed to create a compute ComputeContext!\n");
    return EXIT_FAILURE;
  }

  // Create a command queue
  ComputeCommands = clCreateCommandQueue(ComputeContext, ComputeDeviceId, 0, &err);
  if (!ComputeCommands){
    printf("Error: Failed to create a command ComputeCommands!\n");
    return EXIT_FAILURE;
  }

  // Create the compute program from the source buffer
  ComputeProgram = clCreateProgramWithSource(ComputeContext, 1, (const char **) & source, NULL, &err);
  if (!ComputeProgram || err != CL_SUCCESS) {
    printf("%s\n", source);
    printf("Error: Failed to create compute program!\n");
    return EXIT_FAILURE;
  }

  // Build the program executable
  err = clBuildProgram(ComputeProgram, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS){
    size_t length;
    char build_log[2048];
    printf("%s\n", source);
    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(ComputeProgram, ComputeDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, &length);
    printf("%s\n", build_log);
    return EXIT_FAILURE;
  }

  ComputeKernels = (cl_kernel*) calloc(KernelCount, sizeof(cl_kernel));
  for(i = 0; i < KernelCount; i++) {
    // Create each compute kernel from within the program
    ComputeKernels[i] = clCreateKernel(ComputeProgram, KernelNames[i], &err);
    if (!ComputeKernels[i] || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return EXIT_FAILURE;
    }

    size_t wgSize;
    err = clGetKernelWorkGroupInfo(ComputeKernels[i], ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL); 
    if(err) {
      printf("Error: Failed to get kernel work group size\n");
      return EXIT_FAILURE;
    }
    GROUP_SIZE = min( GROUP_SIZE, wgSize );
  }

  free(source);

  // Create the input buffer on the device
  size_t buffer_size = sizeof(int) * count;
  input_buffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
  if (!input_buffer) {
    printf("Error: Failed to allocate input buffer on device!\n");
    return EXIT_FAILURE;
  }

  // Fill the input buffer with the host allocated random data
  err = clEnqueueWriteBuffer(ComputeCommands, input_buffer, CL_TRUE, 0, buffer_size, int_data, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array!\n");
    return EXIT_FAILURE;
  }

  // Create the output buffer on the device
  output_buffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
  if (!output_buffer) {
    printf("Error: Failed to allocate result buffer on device!\n");
    return EXIT_FAILURE;
  }

  int* result = (int*)calloc(count, sizeof(int));

  err = clEnqueueWriteBuffer(ComputeCommands, output_buffer, CL_TRUE, 0, buffer_size, result, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array!\n");
    return EXIT_FAILURE;
  }

  CreatePartialSumBuffers(count);
  PreScanBuffer(output_buffer, input_buffer, GROUP_SIZE, GROUP_SIZE, count);

  err = clFinish(ComputeCommands);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to wait for command queue to finish! %d\n", err);
    return EXIT_FAILURE;
  }



  // Read back the results that were computed on the device
  err = clEnqueueReadBuffer(ComputeCommands, output_buffer, CL_TRUE, 0, buffer_size, result, 0, NULL, NULL);
  if (err) {
      printf("Error: Failed to read back results from the device!\n");
      return EXIT_FAILURE;
  }

  // Verify the results are correct
  int* reference = (int*) malloc( buffer_size);
  ScanReference(reference, int_data, count);

  float error = 0.0f;
  float diff = 0.0f;
  for(i = 0; i < count; i++) {
    diff = fabs(reference[i] - result[i]);
    error = diff > error ? diff : error;
  }

  if (error > MAX_ERROR) {
      printf("Error:   Incorrect results obtained! Max error = %f\n", error);
      return EXIT_FAILURE;
  } else {
      printf("Results Validated!\n");
      printf(SEPARATOR);
  }

  // Shutdown and cleanup
  ReleasePartialSums();
  for(i = 0; i < KernelCount; i++) clReleaseKernel(ComputeKernels[i]);
  clReleaseProgram(ComputeProgram);
  clReleaseMemObject(input_buffer);
  clReleaseMemObject(output_buffer);
  clReleaseCommandQueue(ComputeCommands);
  clReleaseContext(ComputeContext);

  free(ComputeKernels);
  free(int_data);
  free(reference);
  free(result);

  return 0;
}

