#ifndef CONTROLPANAL_H
#define CONTROLPANAL_H
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


#endif // CONTROLPANAL_H
