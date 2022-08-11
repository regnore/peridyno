#include "PAnimationWidget.h"

#include "PSimulationThread.h"
#include "Platform.h"

#include <QString>
#include <QGridLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QIntValidator>
#include <QDebug>
namespace dyno
{
	PAnimationWidget::PAnimationWidget(QWidget *parent) : 
		QWidget(parent),
		m_startSim(nullptr),
		m_resetSim(nullptr),
		StartIcon(nullptr),
		PauseIcon(nullptr),
		ResetIcon(nullptr)
	{
		mTotalFrame = 800;

		QHBoxLayout* layout = new QHBoxLayout();
		setLayout(layout);

		QGridLayout* frameLayout	= new QGridLayout();

		mTotalFrameSpinbox = new QSpinBox();
		mTotalFrameSpinbox->setFixedSize(60, 29);
		mTotalFrameSpinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

		mTotalFrameSpinbox->setMaximum(1000000);
		mTotalFrameSpinbox->setValue(mTotalFrame);

		QGridLayout* GLayout = new QGridLayout;
		
		mFrameSlider = new PAnimationQSlider(0, mTotalFrame, this);
		mFrameSlider->setObjectName("AnimationSlider");
		mFrameSlider->setStyleSheet("border-top-right-radius: 0px; border-bottom-right-radius: 0px;");
		mFrameSlider->setFixedHeight(29);

		frameLayout->addWidget(mFrameSlider, 0, 0, 0 , (labelSize - 1) * 2);

		QHBoxLayout* operationLayout = new QHBoxLayout();

		m_startSim = new QPushButton();
		m_resetSim = new QPushButton();
		m_startSim->setStyleSheet("padding: 6px;");
		m_resetSim->setStyleSheet("padding: 6px;");

		StartIcon = new QIcon(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Start.png"));
		PauseIcon = new QIcon(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Pause.png"));
		ResetIcon = new QIcon(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/AnimationSlider/Reset.png"));

		m_startSim->setIcon(*StartIcon);
		m_resetSim->setIcon(*ResetIcon);
		m_resetSim->setCheckable(false);

		operationLayout->addWidget(mTotalFrameSpinbox, 0);
		operationLayout->addWidget(m_startSim, 0);
		operationLayout->addWidget(m_resetSim, 0);
		operationLayout->setSpacing(0);

		m_startSim->setCheckable(true);

		layout->addLayout(frameLayout, 10);
		layout->addStretch();
		layout->addLayout(operationLayout, 1);
		layout->setSpacing(0);
		
		connect(m_startSim, SIGNAL(released()), this, SLOT(toggleSimulation()));
		connect(m_resetSim, SIGNAL(released()), this, SLOT(resetSimulation()));
		connect(PSimulationThread::instance(), SIGNAL(simulationFinished()), this, SLOT(simulationFinished()));

		connect(PSimulationThread::instance(), SIGNAL(oneFrameFinished()), this, SLOT(updateSlider()));

		// 动态改变 Slider
		connect(mTotalFrameSpinbox, SIGNAL(valueChanged(int)), mFrameSlider, SLOT(maximumChanged(int)));

		PSimulationThread::instance()->start();
	}

	PAnimationWidget::~PAnimationWidget()
	{
		PSimulationThread::instance()->stop();
		PSimulationThread::instance()->deleteLater();
		PSimulationThread::instance()->wait();  //必须等待线程结束
	}
	
	void PAnimationWidget::toggleSimulation()
	{
		if (m_startSim->isChecked())
		{
			PSimulationThread::instance()->resume();
			m_startSim->setText("");
			m_startSim->setIcon(*PauseIcon);

			m_resetSim->setDisabled(true);
			mTotalFrameSpinbox->setEnabled(false);
			mFrameSlider->setEnabled(false);
		}
		else
		{
			PSimulationThread::instance()->pause();
			m_startSim->setText("");
			m_resetSim->setDisabled(false);
			m_startSim->setIcon(*StartIcon);


			mTotalFrameSpinbox->setEnabled(true);
			mFrameSlider->setEnabled(true);
		}
	}

	void PAnimationWidget::resetSimulation()
	{
		PSimulationThread::instance()->reset(mTotalFrameSpinbox->value());

		m_startSim->setText("");
		m_startSim->setEnabled(true);
		m_startSim->setChecked(false);
		m_startSim->setIcon(*StartIcon);

		mTotalFrameSpinbox->setEnabled(true);
		mFrameSlider->setEnabled(true);
		mFrameSlider->setValue(0);
	}

	void PAnimationWidget::simulationFinished()
	{
		m_startSim->setText("Finished");
		m_startSim->setDisabled(true);
		m_startSim->setChecked(false);

		m_resetSim->setDisabled(false);

		mTotalFrameSpinbox->setEnabled(true);
	}
	
	void PAnimationWidget::updateSlider()
	{
		mFrameSlider->setValue(PSimulationThread::instance()->getCurrentFrameNum());
	}
}
