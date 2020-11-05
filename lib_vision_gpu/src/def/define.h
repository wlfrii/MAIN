#ifndef GPU_ALGORITHM_DEFINE_H
#define GPU_ALGORITHM_DEFINE_H
// INCLUDE NECESSARY HEADERS
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/// THE SWITCH TO DEBUG CV::CUDA::GPUMAT
#define CU_DEBUG 0

/**************************************************************************/

#define GPU_ALGO_BEGIN namespace gpu{
#define GPU_ALGO_END }

GPU_ALGO_BEGIN
/* \brief This Enumeration is used to specify the algo node type of
 * different algo node object inherted from the base class.
 */
enum AlgoNodeType
{
    BASIC_NODE          = 0,
    RECTIFY_NODE        = 1,
    USM_NODE            = 2,
    GUIDED_FILTER_NODE  = 3,
    GAMMA_NODE          = 4,
    IMAGE_ADJUST_NODE   = 5,
	UNEVEN_Y_NODE       = 6,

    BASIC_SMOOTH_NODE   = 7,
    BASIC_SHARP_NODE    = 8,
};

/* Below is the definition of all the property objects.
 * Property is the base class of all the other property, including a public member 
 * storing the AlgoNodeType.
 * The other derived Properties are corresponding to the derived AlgoNodeBase 
 * class, and storing the algorithm data of the derived Classes.
 */

struct Property
{
protected:
    Property(AlgoNodeType type = BASIC_NODE) : algo_node_type(type) {}
public:
    virtual ~Property() {}
    AlgoNodeType algo_node_type;
};


enum TreeType
{
    LEFT_EYE  = 0,
    RIGHT_EYE = 1
};

enum ImageType
{
	//GRAY_U = CV_8UC1,
	//BGR_U = CV_8UC3,
	//BGRA_U = CV_8UC4,
	//GRAY_F = CV_32FC1,	
	//BGR_F = CV_32FC3,	
	//BGRA_F = CV_32FC4,
	GRAY = 1,
	BGR  = 3,
	BGRA = 4
};


/* Bellow is some usefull type definition.
 * U8C3 corresponds to RGB
 * U8C4 corresponds to RGBA
 */

using U8 = unsigned char;
struct U8C1
{
	__device__ __host__ U8C1()
		: x(0) {}
	unsigned char x;
};
struct U8C3
{
    __device__ __host__ U8C3()
        : x(0), y(0), z(0)
    {}
    unsigned char x;
    unsigned char y;
    unsigned char z;
};
struct U8C4
{
    __device__ __host__ U8C4()
        : x(0), y(0), z(0), w(255)
    {}
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
};
GPU_ALGO_END

#endif // GPU_ALGORITHM_DEFINE_H
