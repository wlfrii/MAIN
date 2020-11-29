#include "ui_capture.h"
#include "ui_logger.h"
#include <QPushButton>
#include <QLineEdit>
#include <QSpinBox>
#include <QLabel>
#include "cmd.h"

UICapture::UICapture()
{

}

UICapture::~UICapture()
{

}

UICapture* UICapture::getInstance()
{
	static UICapture ui_capture;
	return &ui_capture;
}

QBoxLayout* UICapture::create()
{
	if (!vlayout)
	{
		vlayout = new QVBoxLayout();

		pBtn_capture = new QPushButton();
		pBtn_capture->setText(tr("Capture Image"));
		connect(pBtn_capture, &QPushButton::clicked, this, &UICapture::onPushBtnCaptureClicked);

		QLabel* lb_capture_num = new QLabel();
		lb_capture_num->setText(tr("Capture number:"));
		spBox_capture_num = new QSpinBox();
		spBox_capture_num->setRange(1, 12);
		spBox_capture_num->setValue(1);
		spBox_capture_num->setSingleStep(1);
		spBox_capture_num->setMaximumWidth(50);

		QLabel* lb_capture_name = new QLabel();
		lb_capture_name->setText(tr("File name:"));
		lEdit_capture_name = new QLineEdit();
		lEdit_capture_name->setMaximumWidth(80);
		
		QHBoxLayout* hlayout_capture_num = new QHBoxLayout();
		hlayout_capture_num->addWidget(lb_capture_num);
		hlayout_capture_num->addWidget(spBox_capture_num);
		hlayout_capture_num->addWidget(lb_capture_name);
		hlayout_capture_num->addWidget(lEdit_capture_name);
		hlayout_capture_num->addWidget(pBtn_capture);

		QLabel* lb_capture_path = new QLabel();
		lb_capture_path->setText(tr("Save path:"));
		lEdit_capture_path = new QLineEdit();
		lEdit_capture_path->setText("./capture/");

		QBoxLayout* hlayout_capture_path = new QHBoxLayout();
		hlayout_capture_path->addWidget(lb_capture_path);
		hlayout_capture_path->addWidget(lEdit_capture_path);

		QBoxLayout* vlayout_capture = new QVBoxLayout();
		vlayout_capture->addLayout(hlayout_capture_num);
		vlayout_capture->addLayout(hlayout_capture_path);
		vlayout->addLayout(vlayout_capture);
	}
	return vlayout;
}

void UICapture::onPushBtnCaptureClicked()
{
	if (CMD::capture.finished) {
		CMD::capture.is_take_photo = true;
		CMD::capture.save_num = spBox_capture_num->value();
		CMD::capture.save_name = lEdit_capture_name->text().toStdString();
		CMD::capture.save_path = lEdit_capture_path->text().toStdString();
		CMD::capture.finished = false;

		UILogger::getInstance()->log(QString("Capturing %1 image...").arg(CMD::capture.save_num));
	}
	else
		UILogger::getInstance()->log(QString("Please wait for current capturing."));
}