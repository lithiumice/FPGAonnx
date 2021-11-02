#define q7_t char
#define q15_t short
#define q31_t int
#define q63_t long

#define int8_t char
#define int16_t short
#define int32_t int
#define int64_t long

#define uint8_t unsigned char
#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long


__kernel void conv2D_no_opt(__global float *input,                                                // input image
            uint16_t dim_im_in_x,                                        // input image dimention x
            uint16_t dim_im_in_y,                                        // input image dimention y
            uint16_t ch_im_in,                                           // number of input image channels
            __global  float *weight,                                               // kernel weights
            uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
            uint16_t dim_kernel_x,                                       // filter kernel size x
            uint16_t dim_kernel_y,                                       // filter kernel size y
            uint16_t padding_x,                                          // padding sizes x
            uint16_t padding_y,                                          // padding sizes y
            uint16_t stride_x,                                           // stride x
            uint16_t stride_y,                                           // stride y
            __global  float *bias,                                                 // bias
            __global float *output,                                                     // output image
            uint16_t dim_im_out_x,                                       // output image dimension x
            uint16_t dim_im_out_y                                        // output image dimension y
)
{
    int i, j, k, l, m, n;
    float conv_out = 0.0f;
    int in_row, in_col;

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
}