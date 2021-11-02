#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cl_utils.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

// #define USE_OPENCL 1
#define USE_OPENCL_OPT 1

using namespace aocl_utils;

#ifdef USE_OPENCL_OPT
#define AOCX_FILE_NAME "fc_dim_1"
#else
#define AOCX_FILE_NAME "fc_no_opt"
#endif

unsigned num_devices = 0;
cl_platform_id platform = NULL;
cl_context context = NULL;
cl_program program = NULL;

scoped_array<cl_device_id> device;
scoped_array<cl_command_queue> queue;
scoped_array<cl_kernel> my_kernel;

scoped_array<cl_mem> input_pV_buf;
scoped_array<cl_mem> input_pM_buf;
scoped_array<cl_mem> input_bias_buf;
scoped_array<cl_mem> input_vec_buffer;
scoped_array<cl_mem> output_pOut_buf;

// scoped_array<scoped_aligned_ptr<float>> input_a, input_b;
// scoped_array<scoped_aligned_ptr<float>> output;
// scoped_array<scoped_array<float>> ref_output;
// scoped_array<unsigned> n_per_device;
#define q7_t int8_t
#define q15_t int16_t
#define q31_t int32_t
#define q63_t int64_t

extern "C"
{
#ifndef USE_OPENCL
    void dense(const float *input,         // pointer to vector
               const float *weight,        // pointer to matrix
               const uint16_t dim_vec,     // length of the vector
               const uint16_t num_of_rows, // numCol of A
               const float *bias,
               float *output) // output operand
    {
        printf("dim_vec:%d\n", dim_vec);
        printf("num_of_rows:%d\n", num_of_rows);

        // printf("bias_shift:%d\n", bias_shift);
        // printf("out_shift:%d\n", out_shift);

        // printf("pV:\n");
        //     for (int j = 0; j < dim_vec; j++)
        //     {
        //         printf("%d ", pV[j]);
        //     }
        //     printf("\n");

        // printf("pM:\n");
        // for (int i = 0; i < num_of_rows; i++)
        // {
        //     for (int j = 0; j < dim_vec; j++)
        //     {
        //         printf("%d ", pM[i * dim_vec + j]);
        //     }
        //     printf("\n");
        // }
        const double start_time = getCurrentTimestamp();
        for (int i = 0; i < num_of_rows; i++)
        {
            float ip_out = bias[i];
            for (int j = 0; j < dim_vec; j++)
            {
                ip_out += input[j] * weight[i * dim_vec + j];
            }
            output[i] = ip_out;
        }

        const double end_time = getCurrentTimestamp();
        printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

        printf("pOut:\n");
        for (int j = 0; j < num_of_rows; j++)
        {
            printf("%.5f ", output[j]);
        }
        printf("\n");
    }
#else
    void dense(const float *input,         // pointer to vector
               const float *weight,        // pointer to matrix
               const uint16_t dim_vec,     // length of the vector
               const uint16_t num_of_rows, // numCol of A
               const float *bias,
               float *output) // output operand
    {
        printf("opencl dense\n");
        printf("dim_vec:%d\n", dim_vec);
        printf("num_of_rows:%d\n", num_of_rows);

        cl_int status;

        unsigned i = 0;
        printf("clCreateBuffer input_pV_buf\n");
        input_pV_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         dim_vec * sizeof(float), NULL, &status);
        checkError(status, "Failed to create buffer for input_pV_buf");

        printf("clCreateBuffer input_pM_buf\n");
        input_pM_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         dim_vec * num_of_rows * sizeof(float), NULL, &status);
        checkError(status, "Failed to create buffer for input_pM_buf");

        printf("clCreateBuffer input_bias_buf\n");
        input_bias_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                           num_of_rows * sizeof(float), NULL, &status);
        checkError(status, "Failed to create buffer for input_bias_buf");

        printf("clCreateBuffer output_pOut_buf\n");
        output_pOut_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            num_of_rows * sizeof(float), NULL, &status);
        checkError(status, "Failed to create buffer for output_pOut_buf");

        // printf("clCreateBuffer input_vec_buffer\n");
        // input_vec_buffer[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        //                                      sizeof(float), NULL, &status);
        // checkError(status, "Failed to create buffer for output");

        const double start_time = getCurrentTimestamp();
        scoped_array<cl_event> kernel_event(num_devices);
        scoped_array<cl_event> finish_event(num_devices);

        i = 0;
        cl_event write_event[3];
        printf("clEnqueueWriteBuffer input_pV_buf\n");
        status = clEnqueueWriteBuffer(queue[i], input_pV_buf[i], CL_FALSE,
                                      0, dim_vec * sizeof(float), input, 0, NULL, &write_event[0]);
        checkError(status, "Failed to transfer pV");

        printf("clEnqueueWriteBuffer input_pM_buf\n");
        status = clEnqueueWriteBuffer(queue[i], input_pM_buf[i], CL_FALSE,
                                      0, dim_vec * num_of_rows * sizeof(float), weight, 0, NULL, &write_event[1]);
        checkError(status, "Failed to transfer pM");

        printf("clEnqueueWriteBuffer input_bias_buf\n");
        status = clEnqueueWriteBuffer(queue[i], input_bias_buf[i], CL_FALSE,
                                      0, num_of_rows * sizeof(float), bias, 0, NULL, &write_event[2]);
        checkError(status, "Failed to transfer bias");

        unsigned argi = 0;

        printf("clSetKernelArg input_pV_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_pV_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg input_pM_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_pM_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg dim_vec\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(uint16_t), &dim_vec);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg num_of_rows\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(uint16_t), &num_of_rows);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg input_bias_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_bias_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg output_pOut_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &output_pOut_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

#ifdef USE_OPENCL_OPT
        const size_t global_work_size = num_of_rows;
#else
        const size_t global_work_size = 1;
#endif

        printf("Launching for device %d (%zd elements)\n", i, global_work_size);

        printf("clEnqueueNDRangeKernel\n");
        status = clEnqueueNDRangeKernel(queue[i], my_kernel[i], 1, NULL,
                                        &global_work_size, NULL, 3, write_event, &kernel_event[i]);
        checkError(status, "Failed to launch my_kernel");

        printf("clEnqueueReadBuffer\n");
        status = clEnqueueReadBuffer(queue[i], output_pOut_buf[i], CL_FALSE,
                                     0, num_of_rows * sizeof(float), output, 1, &kernel_event[i], &finish_event[i]);

        clReleaseEvent(write_event[0]);
        clReleaseEvent(write_event[1]);
        clReleaseEvent(write_event[2]);

        clWaitForEvents(num_devices, finish_event);

        const double end_time = getCurrentTimestamp();
        printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);
        i = 0;
        cl_ulong time_ns = getStartEndTime(kernel_event[i]);
        printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);

        i = 0;
        clReleaseEvent(kernel_event[i]);
        clReleaseEvent(finish_event[i]);

        printf("output:\n");
        for (int j = 0; j < num_of_rows; j++)
        {
            printf("%.5f ", output[j]);
        }
        printf("\n");
    }
#endif

#ifdef USE_OPENCL
void matmul(const float *input,              // pointer to vector
           const float *weight,             // pointer to matrix
           const uint16_t dim_vec,         // length of the vector
           const uint16_t num_of_rows,     // numCol of A
           float *output)
{
        float *bias = (float *)malloc(sizeof(float) * num_of_rows);
        memset(bias, 0, sizeof(float) * num_of_rows);
        dense(input, weight, dim_vec, num_of_rows, bias, output);
}
#else

void matmul(const float *input,              // pointer to vector
           const float *weight,             // pointer to matrix
           const uint16_t dim_vec,         // length of the vector
           const uint16_t num_of_rows,     // numCol of A
           float *output)
{
    for (int i = 0; i < num_of_rows; i++)
    {
        float ip_out = 0;
        for (int j = 0; j < dim_vec; j++)
        {
            ip_out += input[j] * weight[i * dim_vec + j];
        }
        output[i] = ip_out;
    }
}
#endif

void conv2D(const float *input,                                                // input image
            const uint16_t dim_im_in_x,                                        // input image dimention x
            const uint16_t dim_im_in_y,                                        // input image dimention y
            const uint16_t ch_im_in,                                           // number of input image channels
            const float *weight,                                               // kernel weights
            const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
            const uint16_t dim_kernel_x,                                       // filter kernel size x
            const uint16_t dim_kernel_y,                                       // filter kernel size y
            const uint16_t padding_x,                                          // padding sizes x
            const uint16_t padding_y,                                          // padding sizes y
            const uint16_t stride_x,                                           // stride x
            const uint16_t stride_y,                                           // stride y
            const float *bias,                                                 // bias
            float *output,                                                     // output image
            const uint16_t dim_im_out_x,                                       // output image dimension x
            const uint16_t dim_im_out_y                                        // output image dimension y
)
{
    int i, j, k, l, m, n;
    float conv_out = 0.0f;
    int in_row, in_col;

    printf("conv start\n");

    const double start_time = getCurrentTimestamp();

    // For each filter
    for (i = 0; i < ch_im_out; i++)
    {
        // For each image dimension
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
                conv_out = bias[i];
                // For each kernel dimension
                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_x; n++)
                    {
                        // if-for implementation
                        in_row = stride_y * j + m - padding_y;
                        in_col = stride_x * k + n - padding_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
                        {
                            // For each input channel
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out += input[(in_row * dim_im_in_x + in_col) * ch_im_in + l] *
                                            weight[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in +
                                               l];
                            }
                        }
                    }
                }
                output[i + (j * dim_im_out_x + k) * ch_im_out] = conv_out;
            }
        }
    }

    const double end_time = getCurrentTimestamp();
        printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);
}
}

bool init_opencl()
{
    cl_int status;

    printf("Initializing OpenCL\n");

    if (!setCwdToExeDir())
    {
        printf("ERROR: setCwdToExeDir Failed.\n");
        return false;
    }

    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    if (platform == NULL)
    {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);
    for (unsigned i = 0; i < num_devices; ++i)
    {
        printf("  %s\n", getDeviceName(device[i]).c_str());
    }

    context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    std::string binary_file = getBoardBinaryFile(AOCX_FILE_NAME, device[0]);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

    printf("clBuildProgram\n");
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    queue.reset(1);
    my_kernel.reset(1);
    input_pV_buf.reset(1);
    input_pM_buf.reset(1);
    input_bias_buf.reset(1);
    input_vec_buffer.reset(1);
    output_pOut_buf.reset(1);

    unsigned i = 0;
    printf("clCreateCommandQueue\n");
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    printf("clCreateKernel\n");

#ifdef USE_OPENCL_OPT
    const char *kernel_name = "dense_dim_1";

#else
    const char *kernel_name = "dense_no_opt";

#endif

    // const char *kernel_name = AOCX_FILE_NAME;
    my_kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create my_kernel");

    return true;
}

void cleanup()
{
    for (unsigned i = 0; i < num_devices; ++i)
    {
        if (my_kernel && my_kernel[i])
        {
            clReleaseKernel(my_kernel[i]);
        }
        if (queue && queue[i])
        {
            clReleaseCommandQueue(queue[i]);
        }
        if (input_pV_buf && input_pV_buf[i])
        {
            clReleaseMemObject(input_pV_buf[i]);
        }
        if (input_pM_buf && input_pM_buf[i])
        {
            clReleaseMemObject(input_pM_buf[i]);
        }
        if (input_bias_buf && input_bias_buf[i])
        {
            clReleaseMemObject(input_bias_buf[i]);
        }
    }

    if (program)
    {
        clReleaseProgram(program);
    }
    if (context)
    {
        clReleaseContext(context);
    }
}
