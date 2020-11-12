#ifndef CONTROLPANAL_H
#define CONTROLPANAL_H
#include "../def/micro_define.h"
#if LINUX && WITH_QT
#include <QWidget>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QTextBrowser>

#include <QString>
#include <QLabel>


class ControlPanel : public QWidget
{
    Q_OBJECT

public:
    ControlPanel(QWidget *parent = nullptr);
    ~ControlPanel();

private:
    void setupUI();
    void log(const QString &qstr);

private slots:
    void onChkBoxFPSSelected();
    void onChkBoxEnhanceSelected();

private:
    QVBoxLayout *vlayout;
    QTextBrowser *txt_browser;

    QCheckBox *ckb_show_fps;
    QCheckBox *ckb_enhance;

    QLabel *lb_show_fps;


};

#endif
#endif // CONTROLPANAL_H
