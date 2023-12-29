/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Dec 29 18:05:42 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "801944e5abc4c8f50e4e44d07ada2069"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Fri Dec 29 18:05:42 2023"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  lstm_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 420, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  lstm_1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 210, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  input_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 420, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  lstm_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1800, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  lstm_1_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 900, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  lstm_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3360, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  lstm_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 14400, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  lstm_peephole_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 180, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  lstm_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 240, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  lstm_1_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7200, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  lstm_1_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3600, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  lstm_1_peephole_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 90, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  lstm_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 120, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 27000, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 450, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 4, 4, 60, 60),
  1, &dense_1_dense_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_weights, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 15, 1, 1, 1), AI_STRIDE_INIT(4, 4, 60, 60, 60),
  1, &dense_2_dense_weights_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_2_dense_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  lstm_scratch0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 420, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1680, 1680),
  1, &lstm_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  lstm_1_scratch0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 210, 1, 1), AI_STRIDE_INIT(4, 4, 4, 840, 840),
  1, &lstm_1_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  input_0_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 14, 1, 30), AI_STRIDE_INIT(4, 4, 4, 56, 56),
  1, &input_0_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  lstm_output0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 60, 1, 30), AI_STRIDE_INIT(4, 4, 4, 240, 240),
  1, &lstm_output0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  lstm_1_output0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 1, 30), AI_STRIDE_INIT(4, 4, 4, 120, 120),
  1, &lstm_1_output0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  lstm_1_output00, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 900, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3600, 3600),
  1, &lstm_1_output0_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 1, 1), AI_STRIDE_INIT(4, 4, 4, 120, 120),
  1, &dense_dense_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  dense_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 1, 1), AI_STRIDE_INIT(4, 4, 4, 120, 120),
  1, &dense_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 4, 4, 60, 60),
  1, &dense_1_dense_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 4, 4, 60, 60),
  1, &dense_1_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_2_dense_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_2_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  lstm_kernel, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 14, 240, 1, 1), AI_STRIDE_INIT(4, 4, 56, 13440, 13440),
  1, &lstm_kernel_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  lstm_recurrent, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 60, 240, 1, 1), AI_STRIDE_INIT(4, 4, 240, 57600, 57600),
  1, &lstm_recurrent_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  lstm_peephole, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 180), AI_STRIDE_INIT(4, 4, 4, 720, 720),
  1, &lstm_peephole_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  lstm_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 240, 1, 1), AI_STRIDE_INIT(4, 4, 4, 960, 960),
  1, &lstm_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  lstm_1_kernel, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 60, 120, 1, 1), AI_STRIDE_INIT(4, 4, 240, 28800, 28800),
  1, &lstm_1_kernel_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  lstm_1_recurrent, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 30, 120, 1, 1), AI_STRIDE_INIT(4, 4, 120, 14400, 14400),
  1, &lstm_1_recurrent_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  lstm_1_peephole, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 90), AI_STRIDE_INIT(4, 4, 4, 360, 360),
  1, &lstm_1_peephole_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  lstm_1_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 120, 1, 1), AI_STRIDE_INIT(4, 4, 4, 480, 480),
  1, &lstm_1_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 900, 30, 1, 1), AI_STRIDE_INIT(4, 4, 3600, 108000, 108000),
  1, &dense_dense_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 1, 1), AI_STRIDE_INIT(4, 4, 4, 120, 120),
  1, &dense_dense_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_weights, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 30, 15, 1, 1), AI_STRIDE_INIT(4, 4, 120, 1800, 1800),
  1, &dense_1_dense_weights_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_2_chain,
  NULL, &dense_2_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_2_dense_weights, &dense_2_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_dense_layer, 5,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_2_dense_chain,
  NULL, &dense_2_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_1_chain,
  NULL, &dense_2_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_1_dense_weights, &dense_1_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_dense_layer, 4,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_1_dense_chain,
  NULL, &dense_1_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_chain,
  NULL, &dense_1_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &lstm_1_output00),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_dense_weights, &dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_dense_layer, 3,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_dense_chain,
  NULL, &dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  lstm_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &lstm_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &lstm_1_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 9, &lstm_1_kernel, &lstm_1_recurrent, &lstm_1_peephole, &lstm_1_bias, NULL, NULL, NULL, NULL, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &lstm_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  lstm_1_layer, 1,
  LSTM_TYPE, 0x0, NULL,
  lstm, forward_lstm,
  &lstm_1_chain,
  NULL, &dense_dense_layer, AI_STATIC, 
  .n_units = 30, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = false, 
  .reverse_seq = false, 
  .return_state = false, 
  .out_nl = nl_func_tanh_array_f32, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .cell_clip = 3e+38, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  lstm_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &lstm_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 9, &lstm_kernel, &lstm_recurrent, &lstm_peephole, &lstm_bias, NULL, NULL, NULL, NULL, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &lstm_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  lstm_layer, 0,
  LSTM_TYPE, 0x0, NULL,
  lstm, forward_lstm,
  &lstm_chain,
  NULL, &lstm_1_layer, AI_STATIC, 
  .n_units = 60, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = false, 
  .reverse_seq = false, 
  .return_state = false, 
  .out_nl = nl_func_tanh_array_f32, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .cell_clip = 3e+38, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 226804, 1, 1),
    226804, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 11640, 1, 1),
    11640, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &dense_2_output),
  &lstm_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 226804, 1, 1),
      226804, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 11640, 1, 1),
      11640, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &dense_2_output),
  &lstm_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1080);
    input_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1080);
    
    lstm_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2760);
    lstm_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2760);
    
    lstm_output0_array.data = AI_PTR(g_network_activations_map[0] + 4440);
    lstm_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 4440);
    
    lstm_1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 3600);
    lstm_1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 3600);
    
    lstm_1_output0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    lstm_1_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 3600);
    dense_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3600);
    
    dense_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_1_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 120);
    dense_1_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 120);
    
    dense_1_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    dense_2_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 60);
    dense_2_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 60);
    
    dense_2_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    dense_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    dense_1_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 0);
    dense_1_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    
    dense_2_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_2_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 60);
    dense_2_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 60);
    
    dense_2_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_2_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 120);
    dense_2_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 120);
    
    lstm_kernel_array.format |= AI_FMT_FLAG_CONST;
    lstm_kernel_array.data = AI_PTR(g_network_weights_map[0] + 124);
    lstm_kernel_array.data_start = AI_PTR(g_network_weights_map[0] + 124);
    
    lstm_recurrent_array.format |= AI_FMT_FLAG_CONST;
    lstm_recurrent_array.data = AI_PTR(g_network_weights_map[0] + 13564);
    lstm_recurrent_array.data_start = AI_PTR(g_network_weights_map[0] + 13564);
    
    lstm_peephole_array.format |= AI_FMT_FLAG_CONST;
    lstm_peephole_array.data = AI_PTR(g_network_weights_map[0] + 71164);
    lstm_peephole_array.data_start = AI_PTR(g_network_weights_map[0] + 71164);
    
    lstm_bias_array.format |= AI_FMT_FLAG_CONST;
    lstm_bias_array.data = AI_PTR(g_network_weights_map[0] + 71884);
    lstm_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 71884);
    
    lstm_1_kernel_array.format |= AI_FMT_FLAG_CONST;
    lstm_1_kernel_array.data = AI_PTR(g_network_weights_map[0] + 72844);
    lstm_1_kernel_array.data_start = AI_PTR(g_network_weights_map[0] + 72844);
    
    lstm_1_recurrent_array.format |= AI_FMT_FLAG_CONST;
    lstm_1_recurrent_array.data = AI_PTR(g_network_weights_map[0] + 101644);
    lstm_1_recurrent_array.data_start = AI_PTR(g_network_weights_map[0] + 101644);
    
    lstm_1_peephole_array.format |= AI_FMT_FLAG_CONST;
    lstm_1_peephole_array.data = AI_PTR(g_network_weights_map[0] + 116044);
    lstm_1_peephole_array.data_start = AI_PTR(g_network_weights_map[0] + 116044);
    
    lstm_1_bias_array.format |= AI_FMT_FLAG_CONST;
    lstm_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 116404);
    lstm_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 116404);
    
    dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 116884);
    dense_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 116884);
    
    dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 224884);
    dense_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 224884);
    
    dense_1_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 225004);
    dense_1_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 225004);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 897857,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 897857,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

