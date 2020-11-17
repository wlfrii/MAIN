#ifndef USB_DEVICE_DESC_H
#define USB_DEVICE_DESC_H

namespace
{
    // used to avoid ugly p->release() after using the pointer
    template<typename T>
    struct Handle
    {
        Handle(T* p) : p(p) {}
        ~Handle() { p->Release(); }

        T* operator->() const { return p; }

    private:
        T* p;
    };
};


// store the pid and uid of usb device
class USBDeviceID
{
public:
    USBDeviceID() : pid(-1), vid(-1) {}
    USBDeviceID(int pid, int vid)
        : pid(pid)
        , vid(vid)
    {}

    int getPID() const { return pid; }
    int getVID() const { return vid; }

private:
    int pid;
    int vid;
};


// store the description (in this case, just friendly-name) of usb device
class USBDeviceDesc
{
public:
    USBDeviceDesc();

    bool readFrom(::Handle<IMoniker>& moniker);

private:
    bool doReadFrom(::Handle<IPropertyBag>& prop_bag);
    void copyFrom(VARIANT& var_num);

private:
    enum{ MAX_LEN_OF_DESC = 255 };
    char desc[MAX_LEN_OF_DESC];
};

#endif // USB_DEVICE_DESC_H
