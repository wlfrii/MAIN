#ifndef ALGONODE_BASE_H
#define ALGONODE_BASE_H
#include "../def/define.h"
#include <memory>

GPU_ALGO_BEGIN


class AlgoNodeBase
{
public:
    AlgoNodeBase();
    virtual ~AlgoNodeBase();

    virtual void process(cv::cuda::GpuMat & src, cudaStream_t stream = 0) = 0;

    virtual void setProperty(std::shared_ptr<Property> prop);

    AlgoNodeType getNodeType() const;

    void setProcessFlag(bool flag);
    bool getProcessFlag() const;

protected:
    std::shared_ptr<Property> property;

    bool process_flag;  //!< flag: whether needs to process
};

GPU_ALGO_END

#endif // ALGONODE_BASE_H
