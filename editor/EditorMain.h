/***************************************************************
 * Name:      EditorMain.h
 * Purpose:   Defines Application Frame
 * Author:    Ammar Qammaz (ammarkov+rgbd@gmail.com)
 * Created:   2013-10-22
 * Copyright: Ammar Qammaz (http://ammar.gr)
 * License:
 **************************************************************/

#ifndef EDITORMAIN_H
#define EDITORMAIN_H

//(*Headers(EditorFrame)
#include <wx/checkbox.h>
#include <wx/listctrl.h>
#include <wx/button.h>
#include <wx/menu.h>
#include <wx/slider.h>
#include <wx/statusbr.h>
#include <wx/statbox.h>
#include <wx/statline.h>
#include <wx/frame.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/timer.h>
//*)

#include <wx/dc.h>
class EditorFrame: public wxFrame
{
    public:

        EditorFrame(wxWindow* parent,wxWindowID id = -1);
        virtual ~EditorFrame();

        int initializeOverlay(char * pathForSceneFile);
        int stopOverlay();
        int DrawFeaturesAtFeed(wxDC & dc , unsigned int x , unsigned int y, wxListCtrl* whereFrom);
        int DrawAFPoints(wxDC & dc , unsigned int x , unsigned int y );
        void DoBlobTracking();

        void onIdle(wxIdleEvent& evt);
        int doGlobalSeek(long jumpTo);
        void guiSnapFrames(int doSnap);
        int  removeOldSegmentedFrames();
        int  refreshAllOverlays();

        unsigned char * rgbFrame;
        unsigned short * depthFrame;

        int feed_0_x,feed_0_y,feed_1_x,feed_1_y,feed_2_x,feed_2_y,feed_3_x,feed_3_y;
        int mouse_x,mouse_y;
        int add_new_track_point;

        int recording;
        int recordedFrames;
        int compressRecordingOutput;

        int framesDrawn;
        int framesSnapped;

    private:

        void OnOpenModule(wxCommandEvent& event);
        void OnSavePair(wxCommandEvent& event);
        void OnSavePCD(wxCommandEvent& event);
        void OnSaveDepth(wxCommandEvent& event);
        void OpenOverlayEditor(wxCommandEvent& event);

        void OnButtonGetExtrinsics(wxCommandEvent& event);

        //(*Handlers(EditorFrame)
        void OnQuit(wxCommandEvent& event);
        void OnAbout(wxCommandEvent& event);
        void OnTimerTrigger(wxTimerEvent& event);
        void OnbuttonPlayClick(wxCommandEvent& event);
        void OnbuttonStopClick(wxCommandEvent& event);
        void OnbuttonPreviousFrameClick(wxCommandEvent& event);
        void OnbuttonNextFrameClick(wxCommandEvent& event);
        void OncurrentFrameTextCtrlText(wxCommandEvent& event);
        void OnFrameSliderCmdScroll(wxScrollEvent& event);
        void OnButtonSegmentationClick(wxCommandEvent& event);
        void OnButtonScanHuman(wxCommandEvent& event);
        void OnButtonCalibrationClick(wxCommandEvent& event);
        void OnbuttonRecordClick(wxCommandEvent& event);
        void OnButtonAcquisitionGraphClick(wxCommandEvent& event);
        void OnButtonAddClick(wxCommandEvent& event);
        void OnButtonRemoveClick(wxCommandEvent& event);
        void OnButtonExecuteClick(wxCommandEvent& event);
        void OnButtonSendDirectCommandClick(wxCommandEvent& event);
        void OnButtonAFClick(wxCommandEvent& event);
        void OnOverlaySliderCmdScroll(wxScrollEvent& event);
        void OnButtonPlusXPosClick(wxCommandEvent& event);
        void OnButtonMinusPosXClick(wxCommandEvent& event);
        void OnButtonPlusPosYClick(wxCommandEvent& event);
        void OnButtonMinusPosYClick(wxCommandEvent& event);
        void OnButtonPlusPosZClick(wxCommandEvent& event);
        void OnButtonMinusPosZClick(wxCommandEvent& event);
        void OnButtonPlusRotXClick(wxCommandEvent& event);
        void OnButtonMinusRotXClick(wxCommandEvent& event);
        void OnButtonPlusRotYClick(wxCommandEvent& event);
        void OnButtonMinusRotYClick(wxCommandEvent& event);
        void OnButtonPlusRotZClick(wxCommandEvent& event);
        void OnButtonMinusRotZClick(wxCommandEvent& event);
        void OnButtonPrev3DObjClick(wxCommandEvent& event);
        void OnButtonNext3DObjClick(wxCommandEvent& event);
        //*)

        //(*Identifiers(EditorFrame)
        static const long ID_SLIDER1;
        static const long ID_STATICBOX1;
        static const long ID_STATICBOX2;
        static const long ID_BUTTON1;
        static const long ID_BUTTON2;
        static const long ID_BUTTON3;
        static const long ID_BUTTON4;
        static const long ID_STATICTEXT1;
        static const long ID_TEXTCTRL1;
        static const long ID_STATICTEXT2;
        static const long ID_STATICTEXT3;
        static const long ID_BUTTON5;
        static const long ID_BUTTON6;
        static const long ID_BUTTON7;
        static const long ID_BUTTON8;
        static const long ID_LISTCTRL1;
        static const long ID_BUTTON9;
        static const long ID_BUTTON10;
        static const long ID_BUTTON11;
        static const long ID_CHECKBOX1;
        static const long ID_TEXTCTRL2;
        static const long ID_BUTTON13;
        static const long ID_CHECKBOX2;
        static const long ID_CHECKBOX3;
        static const long ID_BUTTON14;
        static const long ID_SLIDER2;
        static const long ID_BUTTON12;
        static const long ID_STATICTEXT4;
        static const long ID_STATICTEXT5;
        static const long ID_STATICTEXT6;
        static const long ID_STATICTEXT7;
        static const long ID_STATICTEXT8;
        static const long ID_STATICTEXT9;
        static const long ID_BUTTON15;
        static const long ID_BUTTON16;
        static const long ID_BUTTON17;
        static const long ID_BUTTON18;
        static const long ID_BUTTON19;
        static const long ID_STATICLINE1;
        static const long ID_BUTTON20;
        static const long ID_STATICTEXT10;
        static const long ID_BUTTON21;
        static const long ID_BUTTON22;
        static const long ID_BUTTON23;
        static const long ID_BUTTON24;
        static const long ID_BUTTON25;
        static const long ID_BUTTON26;
        static const long ID_BUTTON27;
        static const long ID_MENUOPENMODULE;
        static const long ID_MENUSAVEPAIR;
        static const long ID_MENUSAVEDEPTH;
        static const long ID_MENUSAVEPCD;
        static const long ID_MENUSCANHUMAN;
        static const long idMenuQuit;
        static const long ID_MENUSEGMENTATION;
        static const long ID_MENUGETEXTRINSICS;
        static const long ID_MENUDETECTFEATURES;
        static const long ID_MENUOVERLAYEDITOR;
        static const long idMenuAbout;
        static const long ID_STATUSBAR1;
        static const long ID_TIMER1;
        //*)

        //(*Declarations(EditorFrame)
        wxButton* ButtonExecute;
        wxButton* buttonNextFrame;
        wxStatusBar* Status;
        wxStaticText* totalFramesLabel;
        wxButton* ButtonPlusRotZ;
        wxButton* ButtonPlusRotY;
        wxSlider* OverlaySlider;
        wxButton* buttonRecord;
        wxButton* ButtonMinusPosX;
        wxStaticText* StaticText1;
        wxButton* ButtonSendDirectCommand;
        wxButton* ButtonAdd;
        wxButton* buttonStop;
        wxStaticBox* StaticBox2;
        wxTextCtrl* TextCtrlDirectCommand;
        wxButton* ButtonMinusPosY;
        wxButton* ButtonPrev3DObj;
        wxStaticText* StaticText3;
        wxTextCtrl* currentFrameTextCtrl;
        wxStaticText* StaticTextJumpTo;
        wxStaticText* dashForFramesRemainingLabel;
        wxMenuItem* MenuItem3;
        wxStaticLine* StaticLine1;
        wxButton* buttonPreviousFrame;
        wxButton* ButtonMinusPosZ;
        wxSlider* FrameSlider;
        wxMenuItem* MenuItem9;
        wxButton* ButtonMinusRotX;
        wxMenu* Menu4;
        wxStaticBox* StaticBoxVideoFeed;
        wxButton* buttonPlay;
        wxMenuItem* MenuItem11;
        wxStaticText* StaticText7;
        wxButton* ButtonAcquisitionGraph;
        wxButton* ButtonAF;
        wxMenuItem* MenuItem5;
        wxCheckBox* CheckBoxPluginProc;
        wxTimer Timer;
        wxStaticText* StaticText4;
        wxListCtrl* ListCtrlPoints;
        wxStaticText* StaticText5;
        wxMenuItem* MenuItem10;
        wxStaticText* StaticText2;
        wxButton* ButtonCalibration;
        wxButton* ButtonNext3DObj;
        wxButton* ButtonMinusRotY;
        wxButton* ButtonPlusPosZ;
        wxButton* ButtonRemove;
        wxMenuItem* MenuItem7;
        wxMenuItem* MenuItem4;
        wxMenuItem* MenuItem6;
        wxStaticText* StaticText6;
        wxCheckBox* CheckBoxOverlayDepth;
        wxButton* ButtonPlusXPos;
        wxButton* ButtonPlusPosY;
        wxButton* ButtonMinusRotZ;
        wxCheckBox* CheckBoxOverlay;
        wxButton* ButtonPlusRotX;
        wxButton* ButtonSegmentation;
        wxMenuItem* MenuItem8;
        //*)

        void render(wxDC& dc);
        void OnPaint(wxPaintEvent& evt);
        void paintNow();
        void OnMotion(wxMouseEvent& event);

        DECLARE_EVENT_TABLE()
};

#endif // EDITORMAIN_H
