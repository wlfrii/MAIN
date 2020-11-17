#include "algo_node_base.h"

GPU_ALGO_BEGIN
AlgoNodeBase::AlgoNodeBase()
    : process_flag(true)
{

}

AlgoNodeBase::~AlgoNodeBase()
{

}

void AlgoNodeBase::setProperty(std::shared_ptr<Property> prop)
{
    property = prop;
}

AlgoNodeType AlgoNodeBase::getNodeType() const
{
    return property->algo_node_type;
}


void AlgoNodeBase::setProcessFlag(bool flag)
{
    process_flag = flag;
}

bool AlgoNodeBase::getProcessFlag() const
{
    return process_flag;
}

GPU_ALGO_END
