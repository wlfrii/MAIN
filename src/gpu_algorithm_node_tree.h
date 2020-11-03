#ifndef GPU_ALGORITHM_NODE_TREE_H
#define GPU_ALGORITHM_NODE_TREE_H
#include "./def/define.h"
/*---The header files below including all the algo node algorithm---*/
#include "./node/algo_node_base.h"
#include "./node/algo_node_rectify.h"
#include "./node/algo_node_image_adjust.h"
#include "./node/algo_node_guidedfilter.h"
#include "./node/algo_node_usm.h"
#include "./node/algo_node_gamma.h"
#include "./node/algo_node_uneven_y.h"
/*------------------------------------------------------------------*/
#include <vector>


GPU_ALGO_BEGIN
class AlgoNodeTree
{
public:
    AlgoNodeTree();
    ~AlgoNodeTree();

    void insert(AlgoNodeBase* algo);
	
	void clear();

    void process(cv::cuda::GpuMat & src, const cudaStream_t &stream);

    void setProperty(std::shared_ptr<Property> prop);
private:
	void release();

private:
    // Use map to store the inserted AlgoNode and its type
    // using AlgoPair = std::pair<AlgoNodeType, AlgoNodeBase*>;
    std::vector<AlgoNodeBase*> node_tree;
};
GPU_ALGO_END

#endif // GPU_ALGORITHM_NODE_TREE_H
