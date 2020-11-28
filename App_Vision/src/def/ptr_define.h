#ifndef PTR_DEFINE_H
#define PTR_DEFINE_H
#include <vector>

template<typename Tp>
void delete_ptr(Tp* ptr)
{
	if (ptr) {
		delete ptr;
		ptr = nullptr;
	}
}

template<typename Tp>
void delete_ptr(std::vector<Tp*> &ptrs)
{
	for (auto &ptr : ptrs) {
		delete_ptr<Tp>(ptr);
	}
}


/* Delete a pointer */
#define DELETE_PIONTER(ptr)	\
    if(ptr != nullptr) {	\
        delete ptr;			\
        ptr = nullptr;		\
    }
/* Delete an array of pointer */
#define DELETE_ARRAY_PIONTER(ptr)	\
    if(ptr != nullptr) {            \
        delete[] ptr;               \
        ptr = nullptr;              \
    }


#endif // PTR_DEFINE_H
