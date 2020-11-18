#ifndef CONTROLPANAL_H
#define CONTROLPANAL_H
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QLineEdit>
#include <QGroupBox>
#include <vector>

class ControlPanel : public QWidget
{
    Q_OBJECT

public:
    ControlPanel(QWidget *parent = nullptr);
    ~ControlPanel();
private:
    void setupUI();

private slots:
    void onPushBtnCaptureClicked();
    void onChkBoxFPSSelected();
    void onGroupBoxEnhanceSelected();

private:
    // The widgets just used for display could be stored in a vector
    // and release together when the app exit
    std::vector<QLabel*> labels;
    std::vector<QBoxLayout*> layouts;

    // Take photos
    QPushButton     *pBtn_capture;
    QSpinBox        *spBox_capture_num;
    QLineEdit       *lEdit_capture_path;

    /* Control */
    QGroupBox       *gpBox_control;
    QCheckBox       *chkBox_show_fps;

    /* Start enhance */
    QGroupBox       *gpBox_enhance;
};

#endif // CONTROLPANAL_H
