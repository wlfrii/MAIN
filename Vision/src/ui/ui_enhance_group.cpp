#include "../def/micro_define.h"
#include <QString>
#include "ui_logger.h"
#include "ui_enhance_group.h"
#include "ui_enhance_property.h"
#include "ui_enhance_guidedfilter.h"

UIEnhanceGroup::UIEnhanceGroup()
{

}

UIEnhanceGroup *UIEnhanceGroup::getInstance()
{
    static UIEnhanceGroup enhance_group;
    return &enhance_group;
}

QGroupBox *UIEnhanceGroup::create()
{
    QBoxLayout* vlayout_enhance = new QVBoxLayout();
    layouts.push_back(vlayout_enhance);

    gpBox_enhance = new QGroupBox();
    gpBox_enhance->setTitle(tr("Enhance mode"));
    gpBox_enhance->setCheckable(true);
    gpBox_enhance->setLayout(vlayout_enhance);

    // Enhance Group: Rectification
    chkBox_rectify = new QCheckBox();
    chkBox_rectify->setText(tr("Rectification"));
    chkBox_rectify->setChecked(true);
    QObject::connect(chkBox_rectify, &QCheckBox::stateChanged, this, &UIEnhanceGroup::onChkBoxRectifySelected);
    vlayout_enhance->addWidget(chkBox_rectify);

	// Enhance Group: Image property
    vlayout_enhance->addLayout(UIEnhanceSaturation::getInstance()->create());
    vlayout_enhance->addLayout(UIEnhanceContrast::getInstance()->create());
    vlayout_enhance->addLayout(UIEnhanceBrightness::getInstance()->create());

	// Enhance Group: Guided filter
	vlayout_enhance->addLayout(UIEnhanceGuidedFilter::getInstance()->create());

    return gpBox_enhance;
}


void UIEnhanceGroup::onChkBoxRectifySelected()
{
    UILogger::getInstance()->log(QString("Open rectification"));
}


