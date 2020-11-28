#ifndef CONTROL_PANAL_H
#define CONTROL_PANAL_H
#include <QWidget>
#include <vector>

class QBoxLayout;

class ControlPanel : public QWidget
{
    Q_OBJECT

public:
    ControlPanel(QWidget *parent = nullptr);
    ~ControlPanel();

private:
    void setupUI();
};


#endif // CONTROL_PANAL_H
