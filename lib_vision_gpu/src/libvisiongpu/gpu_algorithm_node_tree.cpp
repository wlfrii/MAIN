#include "gpu_algorithm_node_tree.h"
#include <stack>

GPU_ALGO_BEGIN
AlgoNodeTree::AlgoNodeTree()
{
}


AlgoNodeTree::~AlgoNodeTree()
{
	release();
}

void AlgoNodeTree::insert(AlgoNodeBase *algo)
{
    node_tree.push_back(algo);
}


void AlgoNodeTree::clear()
{
	release();
	node_tree.clear();
}


void AlgoNodeTree::process(cv::cuda::GpuMat &src, const cudaStream_t &stream)
{
    for(auto & node : node_tree)
    {
        if(node->getProcessFlag()){
            node->process(src, stream);
        }
    }
}


void AlgoNodeTree::setProperty(std::shared_ptr<Property> prop)
{
    for(auto & node : node_tree)
    {
        if(node->getNodeType() == prop->algo_node_type)
            node->setProperty(prop);
    }
}


void AlgoNodeTree::release()
{
	for (auto & node : node_tree)
	{
		delete node;
		node = nullptr;
	}
}

GPU_ALGO_END
