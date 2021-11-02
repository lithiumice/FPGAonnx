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

__kernel void fc_no_opt(__global float *input,      // pointer to vector
                           __global float *weight,     // pointer to matrix
                           const uint16_t dim_vec,     // length of the vector
                           const uint16_t num_of_rows, // numCol of A
                           __global float *bias,       // bias
                           __global float *output)     // output operand
{
  for (int i = 0; i < num_of_rows; i++) {
    float ip_out = bias[i];
    for (int j = 0; j < dim_vec; j++) {
      ip_out += input[j] * weight[i * dim_vec + j];
    }
    output[i] = ip_out;
  }
}
