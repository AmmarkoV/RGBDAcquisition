/***************************************************************
 * Name:      EditorMain.cpp
 * Purpose:   Code for Application Frame
 * Author:    Ammar Qammaz (ammarkov+rgbd@gmail.com)
 * Created:   2013-10-22
 * Copyright: Ammar Qammaz (http://ammar.gr)
 * License:
 **************************************************************/

#include "EditorMain.h"
#include "FeedScreenMemory.h"
#include "Tools.h"
#include <wx/msgdlg.h>

#include "../acquisitionSegment/AcquisitionSegment.h"
#include "../acquisition_mux/AcquisitionMux.h"
#include "../acquisition/Acquisition.h"
#include "SelectCalibration.h"
#include "SelectAcquisitionGraph.h"
#include "SelectModule.h"
#include "SelectTarget.h"
#include "SelectSegmentation.h"
#include "GetExtrinsics.h"
#include "AddNewElement.h"

#include "../processors/ViewpointChange/ViewpointChange.h"

#define USE_BIRDVIEW_LOGIC 1


#define OVERLAY_EDITOR_SCENE_FILE "Scenes/editor.conf"

ModuleIdentifier moduleID = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//
unsigned int devID=0;

unsigned int width , height , fps ;
char openDevice[512];
char openDeviceOGLOverlay[1024];

unsigned int addingPoint=0;
unsigned int alreadyInitialized=0;
unsigned int play=0;
unsigned int lastFrameDrawn=12312312;
unsigned int totalFramesOfDevice=12312312;

unsigned int combinationMode=DONT_COMBINE;
struct SegmentationFeaturesRGB segConfRGB={0};
struct SegmentationFeaturesDepth segConfDepth={0};
struct calibration calib;

unsigned char * fallenBody =0;
unsigned int bleeps=0;

unsigned int overlayModule=OPENGL_ACQUISITION_MODULE;
unsigned int overlayDevice=6;
unsigned int overlayFramesExist=0;
unsigned char * overlayRGB=0;
unsigned short * overlayDepth=0;

unsigned int ignoreColor=0;
unsigned int ignoreDepth=0;


unsigned int segmentedFramesExist=0;
unsigned char * segmentedRGB=0;
unsigned short * segmentedDepth=0;
unsigned char trR=255,trG=255,trB=255;
unsigned int shiftX=0,shiftY=0;


struct AF_Rectangle
{
  unsigned int x1,y1,width,height,R,G,B;

};


unsigned int afPointsActive=0;
struct AF_Rectangle afPoints[10]={0};



unsigned char * copyRGB(unsigned char * source , unsigned int width , unsigned int height)
{
  unsigned char * output = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
  if (output==0) { return 0; }
  memcpy(output , source , width*height*3*sizeof(unsigned char));
  return output;
}

unsigned short * copyDepth(unsigned short * source , unsigned int width , unsigned int height)
{
  unsigned short * output = (unsigned short*) malloc(width*height*sizeof(unsigned short));
  if (output==0) { return 0; }
  memcpy(output , source , width*height*sizeof(unsigned short));
  return output;
}



//(*InternalHeaders(EditorFrame)
#include <wx/string.h>
#include <wx/intl.h>
//*)



//helper functions
enum wxbuildinfoformat {
    short_f, long_f };

wxString wxbuildinfo(wxbuildinfoformat format)
{
    wxString wxbuild(wxVERSION_STRING);

    if (format == long_f )
    {
#if defined(__WXMSW__)
        wxbuild << _T("-Windows");
#elif defined(__UNIX__)
        wxbuild << _T("-Linux");
#endif

#if wxUSE_UNICODE
        wxbuild << _T("-Unicode build");
#else
        wxbuild << _T("-ANSI build");
#endif // wxUSE_UNICODE
    }

    return wxbuild;
}

//(*IdInit(EditorFrame)
const long EditorFrame::ID_SLIDER1 = wxNewId();
const long EditorFrame::ID_STATICBOX1 = wxNewId();
const long EditorFrame::ID_STATICBOX2 = wxNewId();
const long EditorFrame::ID_BUTTON1 = wxNewId();
const long EditorFrame::ID_BUTTON2 = wxNewId();
const long EditorFrame::ID_BUTTON3 = wxNewId();
const long EditorFrame::ID_BUTTON4 = wxNewId();
const long EditorFrame::ID_STATICTEXT1 = wxNewId();
const long EditorFrame::ID_TEXTCTRL1 = wxNewId();
const long EditorFrame::ID_STATICTEXT2 = wxNewId();
const long EditorFrame::ID_STATICTEXT3 = wxNewId();
const long EditorFrame::ID_BUTTON5 = wxNewId();
const long EditorFrame::ID_BUTTON6 = wxNewId();
const long EditorFrame::ID_BUTTON7 = wxNewId();
const long EditorFrame::ID_BUTTON8 = wxNewId();
const long EditorFrame::ID_LISTCTRL1 = wxNewId();
const long EditorFrame::ID_BUTTON9 = wxNewId();
const long EditorFrame::ID_BUTTON10 = wxNewId();
const long EditorFrame::ID_BUTTON11 = wxNewId();
const long EditorFrame::ID_LISTCTRL2 = wxNewId();
const long EditorFrame::ID_BUTTON12 = wxNewId();
const long EditorFrame::ID_CHECKBOX1 = wxNewId();
const long EditorFrame::ID_TEXTCTRL2 = wxNewId();
const long EditorFrame::ID_BUTTON13 = wxNewId();
const long EditorFrame::ID_CHECKBOX2 = wxNewId();
const long EditorFrame::ID_CHECKBOX3 = wxNewId();
const long EditorFrame::ID_BUTTON14 = wxNewId();
const long EditorFrame::ID_MENUOPENMODULE = wxNewId();
const long EditorFrame::ID_MENUSAVEPAIR = wxNewId();
const long EditorFrame::ID_MENUSAVEDEPTH = wxNewId();
const long EditorFrame::ID_MENUSAVEPCD = wxNewId();
const long EditorFrame::idMenuQuit = wxNewId();
const long EditorFrame::ID_MENUSEGMENTATION = wxNewId();
const long EditorFrame::ID_MENUGETEXTRINSICS = wxNewId();
const long EditorFrame::ID_MENUDETECTFEATURES = wxNewId();
const long EditorFrame::ID_MENUOVERLAYEDITOR = wxNewId();
const long EditorFrame::idMenuAbout = wxNewId();
const long EditorFrame::ID_STATUSBAR1 = wxNewId();
const long EditorFrame::ID_TIMER1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(EditorFrame,wxFrame)
    //(*EventTable(EditorFrame)
    //*)

    //EVT_TIMER(-1,EditorFrame::OnTimerEvent)
    EVT_PAINT(EditorFrame::OnPaint)
    EVT_MOTION(EditorFrame::OnMotion)
END_EVENT_TABLE()

EditorFrame::EditorFrame(wxWindow* parent,wxWindowID id)
{
    //(*Initialize(EditorFrame)
    wxMenuItem* MenuItem2;
    wxMenuItem* MenuItem1;
    wxMenu* Menu1;
    wxMenuBar* MenuBar1;
    wxMenu* Menu2;

    Create(parent, id, _("RGBDAcquisition Editor "), wxDefaultPosition, wxDefaultSize, wxDEFAULT_FRAME_STYLE, _T("id"));
    SetClientSize(wxSize(1487,608));
    FrameSlider = new wxSlider(this, ID_SLIDER1, 0, 0, 10000, wxPoint(8,504), wxSize(1296,22), 0, wxDefaultValidator, _T("ID_SLIDER1"));
    StaticBoxVideoFeed = new wxStaticBox(this, ID_STATICBOX1, _("Video Feed"), wxPoint(8,0), wxSize(1304,504), 0, _T("ID_STATICBOX1"));
    StaticBox2 = new wxStaticBox(this, ID_STATICBOX2, _("Elements"), wxPoint(1312,0), wxSize(172,504), 0, _T("ID_STATICBOX2"));
    buttonPreviousFrame = new wxButton(this, ID_BUTTON1, _("<"), wxPoint(8,524), wxSize(56,27), 0, wxDefaultValidator, _T("ID_BUTTON1"));
    buttonPlay = new wxButton(this, ID_BUTTON2, _("Play"), wxPoint(64,524), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
    buttonStop = new wxButton(this, ID_BUTTON3, _("Stop"), wxPoint(150,524), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON3"));
    buttonNextFrame = new wxButton(this, ID_BUTTON4, _(">"), wxPoint(236,524), wxSize(56,27), 0, wxDefaultValidator, _T("ID_BUTTON4"));
    StaticTextJumpTo = new wxStaticText(this, ID_STATICTEXT1, _("Jump To : "), wxPoint(408,528), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
    currentFrameTextCtrl = new wxTextCtrl(this, ID_TEXTCTRL1, _("0"), wxPoint(472,524), wxDefaultSize, wxTE_PROCESS_ENTER|wxNO_FULL_REPAINT_ON_RESIZE, wxDefaultValidator, _T("ID_TEXTCTRL1"));
    dashForFramesRemainingLabel = new wxStaticText(this, ID_STATICTEXT2, _("/ "), wxPoint(560,528), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
    totalFramesLabel = new wxStaticText(this, ID_STATICTEXT3, _("\?"), wxPoint(576,528), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
    ButtonSegmentation = new wxButton(this, ID_BUTTON5, _("Segmentation"), wxPoint(856,524), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON5"));
    ButtonCalibration = new wxButton(this, ID_BUTTON6, _("Calibration"), wxPoint(608,524), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON6"));
    buttonRecord = new wxButton(this, ID_BUTTON7, _("Record"), wxPoint(304,524), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON7"));
    ButtonAcquisitionGraph = new wxButton(this, ID_BUTTON8, _("Stream Connections"), wxPoint(696,524), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON8"));
    ListCtrlPoints = new wxListCtrl(this, ID_LISTCTRL1, wxPoint(1320,24), wxSize(152,168), wxLC_REPORT|wxLC_SINGLE_SEL|wxRAISED_BORDER|wxVSCROLL, wxDefaultValidator, _T("ID_LISTCTRL1"));
    ButtonAdd = new wxButton(this, ID_BUTTON9, _("+"), wxPoint(1320,192), wxSize(40,29), 0, wxDefaultValidator, _T("ID_BUTTON9"));
    ButtonRemove = new wxButton(this, ID_BUTTON10, _("-"), wxPoint(1360,192), wxSize(40,29), 0, wxDefaultValidator, _T("ID_BUTTON10"));
    ButtonExecute = new wxButton(this, ID_BUTTON11, _("="), wxPoint(1408,192), wxSize(64,29), 0, wxDefaultValidator, _T("ID_BUTTON11"));
    ListCtrl1 = new wxListCtrl(this, ID_LISTCTRL2, wxPoint(1320,264), wxSize(152,208), wxLC_REPORT|wxLC_SINGLE_SEL|wxVSCROLL, wxDefaultValidator, _T("ID_LISTCTRL2"));
    Button1 = new wxButton(this, ID_BUTTON12, _("Remove"), wxPoint(1320,472), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON12"));
    CheckBoxOverlay = new wxCheckBox(this, ID_CHECKBOX1, _("Overlay Active"), wxPoint(976,528), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX1"));
    CheckBoxOverlay->SetValue(false);
    TextCtrlDirectCommand = new wxTextCtrl(this, ID_TEXTCTRL2, wxEmptyString, wxPoint(1320,228), wxSize(120,27), wxTE_PROCESS_ENTER, wxDefaultValidator, _T("ID_TEXTCTRL2"));
    ButtonSendDirectCommand = new wxButton(this, ID_BUTTON13, _(">"), wxPoint(1440,228), wxSize(29,29), 0, wxDefaultValidator, _T("ID_BUTTON13"));
    CheckBoxOverlayDepth = new wxCheckBox(this, ID_CHECKBOX2, _("Overlay Respect Depth"), wxPoint(1104,528), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX2"));
    CheckBoxOverlayDepth->SetValue(false);
    CheckBoxPluginProc = new wxCheckBox(this, ID_CHECKBOX3, _("PlugIn Proc"), wxPoint(1304,528), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX3"));
    CheckBoxPluginProc->SetValue(false);
    ButtonAF = new wxButton(this, ID_BUTTON14, _("AF"), wxPoint(1416,524), wxSize(37,29), 0, wxDefaultValidator, _T("ID_BUTTON14"));
    MenuBar1 = new wxMenuBar();
    Menu1 = new wxMenu();
    MenuItem6 = new wxMenuItem(Menu1, ID_MENUOPENMODULE, _("Open Module"), wxEmptyString, wxITEM_NORMAL);
    Menu1->Append(MenuItem6);
    MenuItem9 = new wxMenuItem(Menu1, ID_MENUSAVEPAIR, _("Save Pair"), wxEmptyString, wxITEM_NORMAL);
    Menu1->Append(MenuItem9);
    MenuItem5 = new wxMenuItem(Menu1, ID_MENUSAVEDEPTH, _("Save Depth Frame"), wxEmptyString, wxITEM_NORMAL);
    Menu1->Append(MenuItem5);
    MenuItem5->Enable(false);
    MenuItem4 = new wxMenuItem(Menu1, ID_MENUSAVEPCD, _("Save Frame as PCD"), wxEmptyString, wxITEM_NORMAL);
    Menu1->Append(MenuItem4);
    MenuItem1 = new wxMenuItem(Menu1, idMenuQuit, _("Quit\tAlt-F4"), _("Quit the application"), wxITEM_NORMAL);
    Menu1->Append(MenuItem1);
    MenuBar1->Append(Menu1, _("&File"));
    Menu4 = new wxMenu();
    MenuItem3 = new wxMenuItem(Menu4, ID_MENUSEGMENTATION, _("Segmentation"), wxEmptyString, wxITEM_NORMAL);
    Menu4->Append(MenuItem3);
    MenuItem7 = new wxMenuItem(Menu4, ID_MENUGETEXTRINSICS, _("Get Extrinsics"), wxEmptyString, wxITEM_NORMAL);
    Menu4->Append(MenuItem7);
    MenuItem8 = new wxMenuItem(Menu4, ID_MENUDETECTFEATURES, _("Detect Features"), wxEmptyString, wxITEM_NORMAL);
    Menu4->Append(MenuItem8);
    MenuItem10 = new wxMenuItem(Menu4, ID_MENUOVERLAYEDITOR, _("Overlay Editor"), wxEmptyString, wxITEM_NORMAL);
    Menu4->Append(MenuItem10);
    MenuBar1->Append(Menu4, _("Module"));
    Menu2 = new wxMenu();
    MenuItem2 = new wxMenuItem(Menu2, idMenuAbout, _("About\tF1"), _("Show info about this application"), wxITEM_NORMAL);
    Menu2->Append(MenuItem2);
    MenuBar1->Append(Menu2, _("Help"));
    SetMenuBar(MenuBar1);
    Status = new wxStatusBar(this, ID_STATUSBAR1, 0, _T("ID_STATUSBAR1"));
    int __wxStatusBarWidths_1[1] = { -1 };
    int __wxStatusBarStyles_1[1] = { wxSB_NORMAL };
    Status->SetFieldsCount(1,__wxStatusBarWidths_1);
    Status->SetStatusStyles(1,__wxStatusBarStyles_1);
    SetStatusBar(Status);
    Timer.SetOwner(this, ID_TIMER1);
    Timer.Start(33, false);

    Connect(ID_SLIDER1,wxEVT_SCROLL_TOP|wxEVT_SCROLL_BOTTOM|wxEVT_SCROLL_LINEUP|wxEVT_SCROLL_LINEDOWN|wxEVT_SCROLL_PAGEUP|wxEVT_SCROLL_PAGEDOWN|wxEVT_SCROLL_THUMBTRACK|wxEVT_SCROLL_THUMBRELEASE|wxEVT_SCROLL_CHANGED,(wxObjectEventFunction)&EditorFrame::OnFrameSliderCmdScroll);
    Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonPreviousFrameClick);
    Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonPlayClick);
    Connect(ID_BUTTON3,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonStopClick);
    Connect(ID_BUTTON4,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonNextFrameClick);
    Connect(ID_TEXTCTRL1,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&EditorFrame::OncurrentFrameTextCtrlText);
    Connect(ID_BUTTON5,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonSegmentationClick);
    Connect(ID_BUTTON6,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonCalibrationClick);
    Connect(ID_BUTTON7,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonRecordClick);
    Connect(ID_BUTTON8,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonAcquisitionGraphClick);
    Connect(ID_BUTTON9,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonAddClick);
    Connect(ID_BUTTON10,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonRemoveClick);
    Connect(ID_BUTTON11,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonExecuteClick);
    Connect(ID_BUTTON13,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonSendDirectCommandClick);
    Connect(ID_BUTTON14,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonAFClick);
    Connect(idMenuQuit,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnQuit);
    Connect(idMenuAbout,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnAbout);
    Connect(ID_TIMER1,wxEVT_TIMER,(wxObjectEventFunction)&EditorFrame::OnTimerTrigger);
    //*)


    //Connect menu stuff
    Connect(ID_MENUSAVEPAIR,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnSavePair);
    Connect(ID_MENUSAVEPCD,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnSavePCD);
    Connect(ID_MENUSAVEDEPTH,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnSaveDepth);
    Connect(ID_MENUOPENMODULE,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnOpenModule);
    Connect(ID_MENUSEGMENTATION,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnButtonSegmentationClick);
    Connect(ID_MENUGETEXTRINSICS,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnButtonGetExtrinsics);

    Connect(ID_MENUOVERLAYEDITOR,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OpenOverlayEditor);

    Connect(ID_TEXTCTRL2,wxEVT_COMMAND_TEXT_ENTER,(wxObjectEventFunction)&EditorFrame::OnButtonSendDirectCommandClick);

    rgbFrame=0;
    depthFrame=0;

    play=0;
    recording=0;
    recordedFrames=0;

    framesSnapped=0;
    framesDrawn=0;

    initFeeds();


    feed_0_x=10;
    feed_0_y=20;

    feed_1_x=feed_0_x+default_feed->GetWidth()+10;
    feed_1_y=feed_0_y;

    feed_2_x=feed_0_x;
    feed_2_y=feed_0_y+default_feed->GetHeight()+10;

    feed_3_x=feed_1_x;
    feed_3_y=feed_2_y;


    wxCommandEvent  event;

    OnOpenModule(event);


    wxListItem col0;
     col0.SetId(0);
     col0.SetText( wxT(" X ") );
     col0.SetWidth(70);
     ListCtrlPoints->InsertColumn(0, col0);

    wxListItem col1;
     col1.SetId(0);
     col1.SetText( wxT(" Y ") );
     col1.SetWidth(70);
     ListCtrlPoints->InsertColumn(1, col1);

    #if USE_BIRDVIEW_LOGIC
     unsigned int width , height , channels, bitsperpixel;
     fallenBody = viewPointChange_ReadPPM((char*) "emergency.pnm",&width,&height,&channels,&bitsperpixel,0);
    #endif // USE_BIRDVIEW_LOGIC

     //Todo -> acquisitionOpenDevice(OPENGL_ACQUISITION_MODULE,9,"Scenes/dragon.conf",width,height,30);
    //Connect( wxID_ANY, wxEVT_IDLE, wxIdleEventHandler(EditorFrame::onIdle) );
}

EditorFrame::~EditorFrame()
{
    //(*Destroy(EditorFrame)
    //*)
}


inline wxString _U(const char String[] = "")
{
  return wxString(String, wxConvUTF8);
}


void EditorFrame::OpenOverlayEditor(wxCommandEvent& event)
{
  char outStr[512];
  sprintf(outStr,"gedit %s",openDeviceOGLOverlay);
  wxExecute(_U(outStr));
}


int EditorFrame::initializeOverlay(char * pathForSceneFile)
{
   if ( acquisitionStartModule(overlayModule,16 /*maxDevices*/ , 0 ) )
   {

     if ( ignoreColor ) { acquisitionDisableStream(overlayModule,overlayDevice,0); }
     if ( ignoreDepth ) { acquisitionDisableStream(overlayModule,overlayDevice,1); }

     if ( acquisitionOpenDevice(overlayModule,overlayDevice,pathForSceneFile,width,height,fps) )
     {
        overlayFramesExist=1;
        return 1;
     }
   }
   return 0;
}

int EditorFrame::stopOverlay()
{
  if (overlayFramesExist)
  {
      acquisitionCloseDevice(overlayModule,overlayDevice);
      acquisitionStopModule(overlayModule);
      return 1;
  }
  return 0;
}


void EditorFrame::OnOpenModule(wxCommandEvent& event)
{
   if (alreadyInitialized)
   {
    removeOldSegmentedFrames();
    stopOverlay();

    acquisitionCloseDevice(moduleID,devID);
    acquisitionStopModule(moduleID);
   }


    SelectModule  * moduleSelectorFrame = new SelectModule(this, wxID_ANY);
    moduleSelectorFrame->ShowModal();

    fprintf(stderr,"Module %u Device %u  -> %u x %u : %u \n",
                     moduleSelectorFrame->moduleSelected ,
                     moduleSelectorFrame->deviceSelected ,
                     moduleSelectorFrame->widthSelected,
                     moduleSelectorFrame->heightSelected,
                     moduleSelectorFrame->fpsSelected
                     );

   ignoreColor = moduleSelectorFrame->ignoreColor;
   ignoreDepth = moduleSelectorFrame->ignoreDepth;

   moduleID = moduleSelectorFrame->moduleSelected;
   devID = moduleSelectorFrame->deviceSelected;
   width = moduleSelectorFrame->widthSelected;
   height = moduleSelectorFrame->heightSelected;
   fps = moduleSelectorFrame->fpsSelected;
   strcpy( openDevice , moduleSelectorFrame->deviceNameSelected );

   fprintf(stderr,"Device %s \n", moduleSelectorFrame->deviceNameSelected);

   delete moduleSelectorFrame;


   if (!acquisitionIsModuleAvailiable(moduleID))
    {
        fprintf(stderr,"The module you are trying to use is not linked in this build of the Acquisition library..\n");
        //return 1;
    }


   //We need to initialize our module before calling any related calls to the specific module..
   if (!acquisitionStartModule(moduleID,16 /*maxDevices*/ , 0 ))
   {
       fprintf(stderr,"Could not start module %s ..\n",getModuleNameFromModuleID(moduleID));
       //return 1;
    }

  if ( ignoreColor ) { acquisitionDisableStream(moduleID,devID,0); }
  if ( ignoreDepth ) { acquisitionDisableStream(moduleID,devID,1); }

   if (fps!=0) { Timer.Stop(); Timer.Start((unsigned int) 1000/fps, false); }

   if ( ! acquisitionOpenDevice(moduleID,devID,openDevice,width,height,fps) )
   {
       wxMessageBox(wxT("Error while Opening device"),wxT("RGBDAcquisition Editor"));
       fprintf(stderr,"Could not open device \n");
       return;
   }

   if ( !acquisitionGetColorCalibration(moduleID,devID,&calib) )
   {
       wxMessageBox(wxT("Error while Getting Calibration!"),wxT("RGBDAcquisition Editor"));
   }

   if (! acquisitionMapDepthToRGB(moduleID,devID) )
   {
       fprintf(stderr,"Could not map depth to rgb , well no one really cares I guess.. \n");
   }


   unsigned int totalFrames =  acquisitionGetTotalFrameNumber(moduleID,devID);

   play=0;
   recording=0; recordedFrames=0;
   framesSnapped=0;

   if  (
         (totalFrames==0) ||
         (moduleID==V4L2_ACQUISITION_MODULE) ||
         (moduleID==V4L2STEREO_ACQUISITION_MODULE)
       )

                       {
                         totalFramesLabel->SetLabel(wxT("Live Stream"));
                         play=1;
                         FrameSlider->SetMin(0);
                         FrameSlider->SetMax(1);
                         FrameSlider->Disable();
                         currentFrameTextCtrl->Disable();
                         buttonPreviousFrame->Disable();
                         buttonNextFrame->Disable();

                       } else // This means a live stream
                       {
                         wxString msg; msg.Printf(wxT("%u"),totalFrames);
                         totalFramesLabel->SetLabel(msg);
                         FrameSlider->SetMin(0);
                         FrameSlider->SetMax(totalFrames);
                         FrameSlider->Enable();
                         currentFrameTextCtrl->Enable();
                         buttonPreviousFrame->Enable();
                         buttonNextFrame->Enable();
                       }

   if (moduleID==TEMPLATE_ACQUISITION_MODULE)
   {
    sprintf(openDeviceOGLOverlay,"frames/%s/dataset.scene",openDevice);
    if (!acquisitionFileExists(openDeviceOGLOverlay))
     {
      fprintf(stderr,"\n\nCouldn't find a dataset specific dataset.scene file (%s) , falling back on editor default\n\n",openDeviceOGLOverlay);
      strcpy(openDeviceOGLOverlay,OVERLAY_EDITOR_SCENE_FILE);
     } else
     {
      fprintf(stderr,"\n\nDataset contains OGL overlay file %s which will  be used \n\n",openDeviceOGLOverlay);
     }
   } else
   {
     //Non template input gets a forced editor overlay
     strcpy(openDeviceOGLOverlay,OVERLAY_EDITOR_SCENE_FILE);
   }


   initializeOverlay(openDeviceOGLOverlay);

   initializeRGBSegmentationConfiguration(&segConfRGB,width,height);
   initializeDepthSegmentationConfiguration(&segConfDepth,width,height);

   guiSnapFrames(1);

   alreadyInitialized=1;
   //This hangs the window -> guiSnapFrames();
   wxYield();
   wxThread::Sleep(0.4);
   wxYield();

}


void EditorFrame::OnQuit(wxCommandEvent& event)
{
    play=0;
    wxSleep(0.1);
    Refresh();
    if (recording)
    {
        fprintf(stderr,"TODO : handle stopping recording properly \n");
    }
    wxSleep(0.1);
    Close();

    closeFeeds();
    stopOverlay();

    removeOldSegmentedFrames();

    #if USE_BIRDVIEW_LOGIC
     if (fallenBody!=0) { free(fallenBody); fallenBody=0; }
    #endif // USE_BIRDVIEW_LOGIC


    acquisitionCloseDevice(moduleID,devID);
    acquisitionStopModule(moduleID);
    fprintf(stderr,"Gracefully stopped modules \n");

}

void EditorFrame::OnAbout(wxCommandEvent& event)
{
    //wxString msg = wxbuildinfo(long_f);
    wxMessageBox(wxT("Thank you for using RGBDAcquisition , a GPL project\nWritten by AmmarkoV\nHosted at https://github.com/AmmarkoV/RGBDAcquisition"), wxT("RGBDAcquisition Editor"));
}


void EditorFrame::OnSavePair(wxCommandEvent& event)
 {
    char filename[512];
    sprintf(filename,"color%05u",lastFrameDrawn);
    acquisitionSaveColorFrame(moduleID,devID,filename);
    sprintf(filename,"depth%05u",lastFrameDrawn);
    acquisitionSaveDepthFrame(moduleID,devID,filename);
 }


void EditorFrame::OnSavePCD(wxCommandEvent& event)
 {
    char filename[512];
    sprintf(filename,"pointCloud%05u.pcd",lastFrameDrawn);
    acquisitionSavePCDPointCoud(moduleID,devID,filename);
 }
void EditorFrame::OnSaveDepth(wxCommandEvent& event)
{
  dumpExtDepths(moduleID , devID ,(char*) "extDepths.txt");
}


void EditorFrame::OnPaint(wxPaintEvent& evt)
{
    wxPaintDC dc(this);
    render(dc);
}

void EditorFrame::paintNow()
{
    wxClientDC dc(this);
    render(dc);
}


int EditorFrame::DrawAFPoints(wxDC & dc , unsigned int x , unsigned int y )
{
 //fprintf(stderr,"DrawAFPoints for %u points",afPointsActive);
 int i;
 dc.SetBrush(*wxTRANSPARENT_BRUSH);
        for ( i=0; i<afPointsActive; i++ )
         {
            wxPen tmp_marker(wxColour(afPoints[i].R,afPoints[i].G,afPoints[i].B),1,wxSOLID);
            dc.SetPen(tmp_marker);
            dc.DrawRectangle(x+afPoints[i].x1,y+afPoints[i].y1,afPoints[i].width,afPoints[i].height);
         }
  return 1;
}




int EditorFrame::DrawFeaturesAtFeed(wxDC & dc , unsigned int x , unsigned int y, wxListCtrl* whereFrom)
{
if (whereFrom==0) { return 0;}
if  ( whereFrom->GetItemCount() > 0 )
      { fprintf(stderr,"Drawing %i features \n",whereFrom->GetItemCount());
        wxPen red_marker(wxColour(255,0,0),1,wxSOLID);
         dc.SetPen(red_marker);

        int i;
        for ( i=0; i<whereFrom->GetItemCount(); i++ )
         {
            unsigned int ptX,ptY;
            ptX = getwxListInteger(whereFrom,0,i);
            ptY = getwxListInteger(whereFrom,1,i);

            dc.DrawRectangle(x+ptX,y+ptY,3,3);
         }
     }
  return 1;
}



void EditorFrame::render(wxDC& dc)
{
  if ( (rgbFrame!=0) && (live_feeds[0].bmp!=0) )
   { dc.DrawBitmap(*live_feeds[0].bmp,feed_0_x,feed_0_y,0); } //FEED 1

  if ( (depthFrame!=0) && (live_feeds[1].bmp!=0) )
   { dc.DrawBitmap(*live_feeds[1].bmp,feed_1_x,feed_1_y,0); } //FEED 2



   if (recording)
   { //DRAW RECORDING DECAL ON LEFT FEED
     wxPen red(wxColour(255,0,0),1,wxSOLID);
     dc.SetPen(red);
     dc.SetBrush(*wxRED_BRUSH); //*wxTRANSPARENT_BRUSH
     dc.DrawCircle(50,50,10); //Recording Mark ON!
   }

   DrawAFPoints(dc ,feed_0_x,feed_0_y);
   DrawFeaturesAtFeed(dc,feed_0_x,feed_0_y,ListCtrlPoints);

 wxSleep(0.01);
 wxYieldIfNeeded();
}



void activateBlobSelector(unsigned int x,unsigned int y)
{

}


int convertCenterCoordinatesToUpperLeft(unsigned int * sX , unsigned int *sY , unsigned int centerX,unsigned int centerY , unsigned int *width , unsigned int *height , unsigned int maxWidth , unsigned int maxHeight)
{
   fprintf(stderr,"Converting Center Coord( %u , %u ) with a patch size ( %u , %u ) ",centerX,centerY,*width,*height);
   unsigned int halfWidth  = (unsigned int) *width / 2;
   unsigned int halfHeight = (unsigned int) *height / 2;

   if ( halfWidth < centerX )   { *sX = centerX-halfWidth; }  else { *sX=0; }
   if ( halfHeight < centerY )  { *sY = centerY-halfHeight; } else { *sY=0; }

   if (*sX + *width >=maxWidth) { *width=maxWidth-*sX-1; }
   if (*sY + *height >=maxHeight) { *height=maxHeight-*sY-1; }
   fprintf(stderr,"to Coord( %u , %u ) with a patch size ( %u , %u ) ",*sX,*sY,*width,*height);

  return 1;
}

void EditorFrame::OnMotion(wxMouseEvent& event)
{
  int x=event.GetX();
  int y=event.GetY();

  int fd_rx1,fd_rx2,fd_ry1,fd_ry2;
  fd_rx1=10;
  fd_ry1=15+default_feed->GetHeight()+10;
  fd_rx2=fd_rx1 + default_feed->GetWidth();
  fd_ry2=fd_ry1 + default_feed->GetHeight();

  unsigned int unscaledImageWidth , unscaledImageHeight , unscaledImageChannels , unscaledImageBitsPerPixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&unscaledImageWidth ,&unscaledImageHeight ,&unscaledImageChannels ,&unscaledImageBitsPerPixel);

  float upscaleRatioWidth  = (float) unscaledImageWidth / default_feed->GetWidth();
  float upscaleRatioHeight = (float) unscaledImageHeight / default_feed->GetHeight();

  if ( XYOverRect(x,y,feed_0_x,feed_0_y,feed_0_x+default_feed->GetWidth(),feed_0_y+default_feed->GetHeight()) )
       {

         mouse_x=(unsigned int) ((x-feed_0_x) * upscaleRatioWidth );
         mouse_y=(unsigned int) ((y-feed_0_y) * upscaleRatioHeight );

         if ( event.LeftIsDown()==1 )
           {
              if (addingPoint)
              {
               ListCtrlPoints->Hide();
               wxString txt; txt<<wxT("2d");
               long tmp = ListCtrlPoints->InsertItem(0,txt);
               fprintf(stderr,"Inserting item %lu \n",tmp);
                //ListCtrlPoints->SetItemData(tmp, i);
                //ListCtrlPoints->SetItemData(tmp, i);
               txt.clear(); txt<<x;
                ListCtrlPoints->SetItem(tmp, 0, txt);
               txt.clear(); txt<<y;
                ListCtrlPoints->SetItem(tmp, 1, txt);

               ListCtrlPoints->Show();
               addingPoint=0;
              }

              unsigned char r=0,g=0,b=0;
              acquisitionGetColorRGBAtXY(moduleID,devID,mouse_x,mouse_y,&r,&g,&b);

              float x,y,z;
              if ( acquisitionGetDepth3DPointAtXY(moduleID,devID,mouse_x,mouse_y,&x,&y,&z) )
              {
                wxString msg;

                fprintf(stderr,"Depth(%u,%u)=%u - 3D(%0.5f,%0.5f,%0.5f) - RGB(%u,%u,%u)  \n",mouse_x,mouse_y,acquisitionGetDepthValueAtXY(moduleID,devID,mouse_x,mouse_y),x,y,z,r,g,b);
                if (calib.extrinsicParametersSet) { msg.Printf( wxT("Using Extrinsic Calibration : Depth(%0.5f,%0.5f,%0.5f) - RGB(%u,%u,%u) ") ,x,y,z , r,g,b  ); } else
                                                  { msg.Printf( wxT("Using Camera Space : Depth(%0.5f,%0.5f,%0.5f) - RGB(%u,%u,%u) ") ,x,y,z , r,g,b ); }

                Status->SetStatusText(msg);
              } else
              {
                if (!calib.intrinsicParametersSet)
                {
                   wxString msg;
                   msg.Printf( wxT("Cannot get 3D point from input source , please check your calibration data - RGB(%u,%u,%u) ")  , r,g,b  );
                   Status->SetStatusText(msg);
                }
                 else
                {
                   wxString msg;
                   msg.Printf( wxT("No 3D Depth at this point..! - RGB(%u,%u,%u) ")  , r,g,b  );
                   Status->SetStatusText(msg);
                }
              }
           }
       }

  if ( XYOverRect(x,y,feed_1_x,feed_1_y,feed_1_x+default_feed->GetWidth(),feed_1_y+default_feed->GetHeight()) )
       {
         mouse_x=(unsigned int) ((x-feed_1_x) * upscaleRatioWidth );
         mouse_y=(unsigned int) ((y-feed_1_y) * upscaleRatioHeight );

         if ( event.LeftIsDown()==1 )
           {
             unsigned int checkWidth=40 , checkHeight=40;
             unsigned int sX=mouse_x,sY=mouse_y;

             float centerX , centerY , centerZ;
             float dimX , dimY , dimZ;
             unsigned int width , height , channels , bitsperpixel;
             acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
             segmentGetDepthBlobAverage(depthFrame,width,height,
                                        sX,sY,checkWidth,checkHeight,
                                        &centerX,&centerY,&centerZ);

             //This was just used for a test : mallocSelectVolume(depthFrame,width,height,sX,sY,1.0);
             unsigned int nonCenterX,nonCenterY,wantedWidth=300,wantedHeight=200;
             convertCenterCoordinatesToUpperLeft(&nonCenterX,&nonCenterY,sX,sY,&wantedWidth,&wantedHeight,width,height);
             segmentGetDepthBlobDimensions(depthFrame,width,height,nonCenterX,nonCenterY,wantedWidth,wantedHeight,&dimX,&dimY,&dimZ);
             afPointsActive=1;
             afPoints[0].R=255; afPoints[0].G=0; afPoints[0].B=0;
             afPoints[0].x1=nonCenterX; afPoints[0].y1=nonCenterY; afPoints[0].width=wantedWidth; afPoints[0].height=wantedHeight;



             fprintf(stderr,"getDepthBlobAverage starting @ %u,%u dims %u,%u\n",sX,sY,checkWidth,checkHeight);

             float mouseX , mouseY , mouseZ;
             transform2DProjectedPointTo3DPoint(&calib, sX , sY , (unsigned short) centerZ , &mouseX , &mouseY , &mouseZ);

             fprintf(stderr,"transform2DProjectedPointTo3DPoint got %f,%f,%f\n", mouseX , mouseY , mouseZ);

             transform3DPointUsingCalibration(&calib ,&mouseX , &mouseY , &mouseZ);

             fprintf(stderr,"transform3DPointUsingCalibration got %f,%f,%f\n", mouseX , mouseY , mouseZ);

             wxString msg;

             if (calib.extrinsicParametersSet) { msg.Printf(wxT("3D Blob Center Using Extrinsic Calibration :  %0.5f   %0.5f   %0.5f "),mouseX,mouseY,mouseZ); } else
                                               { msg.Printf(wxT("3D Blob Center Using Camera Space :  %0.5f   %0.5f   %0.5f "),mouseX,mouseY,mouseZ); }

             Status->SetStatusText(msg);


           }
       }

  wxSleep(0.01);
  wxYield();
}


int EditorFrame::removeOldSegmentedFrames()
{
    if (!segmentedFramesExist ) { return 0; }
    if ( segmentedRGB!=0 ) { free(segmentedRGB); segmentedRGB=0; }
    if ( segmentedDepth!=0 ) { free(segmentedDepth); segmentedDepth=0; }
    return 1;
}


int  EditorFrame::refreshAllOverlays()
{
    unsigned int retres=0;
    if (segmentedFramesExist)
    {
     removeOldSegmentedFrames();
     segmentedRGB = copyRGB(rgbFrame, width , height);
     segmentedDepth = copyDepth(depthFrame , width , height);


     segmentRGBAndDepthFrame (
                              segmentedRGB ,
                              segmentedDepth ,
                              width , height ,
                              &segConfRGB ,
                              &segConfDepth ,
                              &calib ,
                              combinationMode
                             );

     unsigned int newColorByteSize = width * height * 3 * sizeof(unsigned char);
     acquisitionOverrideColorFrame(moduleID,devID,segmentedRGB,newColorByteSize,width,height,3,24);

     unsigned int newDepthByteSize = width * height * 1 * sizeof(unsigned short);
     acquisitionOverrideDepthFrame(moduleID,devID,segmentedDepth,newDepthByteSize,width,height,1,16);

     if ( segmentedRGB!=0 ) { free(segmentedRGB); segmentedRGB=0; }
     if ( segmentedDepth!=0 ) { free(segmentedDepth); segmentedDepth=0; }

     retres=1;
    }


   if ( (overlayFramesExist) && ( CheckBoxOverlay->GetValue() ) )
    {
        unsigned char * rgbOut = (unsigned char * ) malloc(width * height * 3 * sizeof(unsigned char) );
        unsigned short * depthOut = (unsigned short * )  malloc(width * height * 1 * sizeof(unsigned short) );

        if ( (rgbOut!=0) && (depthOut!=0) )
        {
         int muxMode = COLOR_MUXING;
         if (CheckBoxOverlayDepth->IsChecked())   { muxMode=DEPTH_MUXING; }

         overlayRGB = acquisitionGetColorFrame(overlayModule,overlayDevice);
         if (muxMode==DEPTH_MUXING) { overlayDepth = acquisitionGetDepthFrame(overlayModule,overlayDevice); } else
                                    { overlayDepth = 0; } //COLOR MUXING IGNORES DEPTH OVERLAY , so we can get away passing a null there

         mux2RGBAndDepthFrames(
                                rgbFrame    , overlayRGB , rgbOut ,
                                depthFrame  , overlayDepth , depthOut ,
                                trR,trG,trB,
                                shiftX,shiftY,
                                width , height , 0 ,
                                muxMode
                              );

         unsigned int newColorByteSize = width * height * 3 * sizeof(unsigned char);
         acquisitionOverrideColorFrame(moduleID,devID,rgbOut,newColorByteSize,width,height,3,24);

         unsigned int newDepthByteSize = width * height * 1 * sizeof(unsigned short);
         acquisitionOverrideDepthFrame(moduleID,devID,depthOut,newDepthByteSize,width,height,1,16);
        }

       if ( rgbOut!=0 ) { free(rgbOut); rgbOut=0; }
       if ( depthOut!=0 ) { free(depthOut); depthOut=0; }
    }

    #if USE_BIRDVIEW_LOGIC
      if(CheckBoxPluginProc->GetValue())
      {

      unsigned int depthAvg = viewPointChange_countDepths(acquisitionGetDepthFrame(moduleID,devID) , width , height,
                                147 , 169 , 300 , 200 ,
                                1000 );
      fprintf(stderr,"RECT Score is %u \n",depthAvg);
      if (
           ( depthAvg > 1000) &&
           ( depthAvg < 2000)
         )
                       {
                        fprintf(stderr,"\n\n OBSTACLE \n\n");
                        ++bleeps;
                        if (bleeps%10==0) { int i=system("paplay bleep.wav&");
                                            if (i!=0) { fprintf(stderr,"Could not emit beep\n"); }
                                          }
                       }

/*
       unsigned char * bev = viewPointChange_mallocTransformToBirdEyeView
                                 (
                                  acquisitionGetColorFrame(moduleID,devID) ,
                                  acquisitionGetDepthFrame(moduleID,devID) ,
                                  width , height , 10000
                                 );
       if (bev!=0) {
                       //acquisitionSaveRawImageToFile( "bev.pnm",bev   , width , height , 3, 8 );
                       unsigned int newColorByteSize = width * height * 3 * sizeof(unsigned char);
                       acquisitionOverrideColorFrame(moduleID,devID,bev,newColorByteSize);

                       unsigned int fitScore = viewPointChange_fitImageInMask(bev,fallenBody, width , height );

                       fprintf(stderr,"Score is %u \n",fitScore);
                       if (fitScore < 4000)
                       {
                        fprintf(stderr,"\n\n OBSTACLE \n\n",fitScore);
                        ++bleeps;
                        if (bleeps%10==0) { system("paplay bleep.wav&"); }
                       }

                       free(bev);
                   }*/
      }
    #endif // USE_BIRDVIEW_LOGIC
 return retres;
}


void EditorFrame::guiSnapFrames(int doSnap)
{
  this->Freeze();                 // Freeze the window to prevent scrollbar jumping

  ++framesSnapped;
  //fprintf(stderr,"guiSnapFrames Called %u ! \n",framesSnapped);
  if (doSnap)
          {
            acquisitionSnapFrames(moduleID,devID);

            if ( (overlayFramesExist) && ( CheckBoxOverlay->GetValue() ) )
              {
                acquisitionSnapFrames(overlayModule,overlayDevice);
              }
          }

  rgbFrame = acquisitionGetColorFrame(moduleID,devID);
  depthFrame = acquisitionGetDepthFrame(moduleID,devID);

  refreshAllOverlays();

  //These should now contain the segmented frame!
  rgbFrame = acquisitionGetColorFrame(moduleID,devID);
  depthFrame = acquisitionGetDepthFrame(moduleID,devID);


  unsigned int devID=0;
  unsigned int width , height , channels , bitsperpixel;
  unsigned int currentFrameDrawn=acquisitionGetCurrentFrameNumber(moduleID,devID);


  if (
        ( (lastFrameDrawn!=currentFrameDrawn)  )
        // || ( (acquisitionGetTotalFrameNumber(moduleID,devID)==0) && (play) )
     )
  {
       //DRAW RGB FRAME -------------------------------------------------------------------------------------
       acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
       passVideoRegisterToFeed(0,rgbFrame,width,height,bitsperpixel,channels);

       //DRAW DEPTH FRAME -------------------------------------------------------------------------------------
       acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);


       unsigned char * rgbDepth = convertShortDepthTo3CharDepth(depthFrame,width,height,0,2048);
       //char * rgbDepth = convertShortDepthToRGBDepth(depthFrame,width,height);
       if (rgbDepth!=0)
       {
        passVideoRegisterToFeed(1,rgbDepth,width,height,8,3);
        free(rgbDepth);
       }


   lastFrameDrawn=currentFrameDrawn;
   totalFramesOfDevice=acquisitionGetTotalFrameNumber(moduleID,devID);


   wxString currentFrame;
   if (totalFramesOfDevice!=0)
   {
      #warning "This call is problematic , it might lead to the window becoming non-responsive"
      //fprintf(stderr,"This call is problematic , it might lead to the window becoming non-responsive");
      currentFrame.clear();
      currentFrame<<lastFrameDrawn;
      if (!currentFrame.IsSameAs(currentFrameTextCtrl->GetValue()))
               { currentFrameTextCtrl->SetValue(currentFrame); }
   }

   currentFrame.clear();
   currentFrame<<totalFramesOfDevice;
   totalFramesLabel->SetLabel(currentFrame);


   if (lastFrameDrawn!=FrameSlider->GetValue())
        { FrameSlider->SetValue(lastFrameDrawn); }

   if (FrameSlider->GetMax()!=totalFramesOfDevice)
        {
          if (totalFramesOfDevice==0) { FrameSlider->SetMax(1); } else
                                      { FrameSlider->SetMax(totalFramesOfDevice); }
        }

  } else
  {
      fprintf(stderr,"Will not refresh bitmaps (acquisitionGetCurrentFrameNumber does not indicate a new frame availiable)..\n" );
  }


  this->Thaw();                 // Freeze the window to prevent scrollbar jumping
  wxMilliSleep(10);
}


void EditorFrame::onIdle(wxIdleEvent& evt)
{
   fprintf(stderr,"OnIdle\n");
   this->paintNow();
   //evt.RequestMore(); // render continuously, not only once on idle

   //Refresh();
   //wxThread::Sleep(0.4);
}



void EditorFrame::OnTimerTrigger(wxTimerEvent& event)
{
  if (!alreadyInitialized) { return ; }
  //fprintf(stderr,"OnTimerTrigger Called\n");
  if (play)
    {
     guiSnapFrames(1); //Get New Frames

     if (recording)
     {
         acquisitionPassFramesToTarget(moduleID,devID,recordedFrames);
         ++recordedFrames;

         if (recordedFrames % 3 == 0 ) { Refresh(); /*Throttle window refreshes when recording*/}
     } else
     {
       Refresh();
       //if (framesDrawn%2 == 0 ) { Refresh();  /*Throttle window refreshes when viewing*/ }
     }
    }

  wxYield();
  //wxThread::Sleep(0.4);
}

void EditorFrame::OnbuttonPlayClick(wxCommandEvent& event)
{
    fprintf(stderr,"Play Button Clicked ( current frame = %u ) \n", acquisitionGetCurrentFrameNumber(moduleID,devID));
    play=1;
}

void EditorFrame::OnbuttonStopClick(wxCommandEvent& event)
{
    play=0;
}

void EditorFrame::OnbuttonPreviousFrameClick(wxCommandEvent& event)
{
    acquisitionSeekRelativeFrame(moduleID,devID,(signed int) -2);
    guiSnapFrames(1); //Get New Frames

    Refresh(); // <- This draws the window!
}

void EditorFrame::OnbuttonNextFrameClick(wxCommandEvent& event)
{
    //acquisitionSeekRelativeFrame(moduleID,devID,(signed int) +1);
    guiSnapFrames(1); //Get New Frames

    Refresh(); // <- This draws the window!
}


int EditorFrame::doGlobalSeek(long jumpTo)
{
  acquisitionSeekFrame(moduleID,devID,jumpTo);

  if (overlayFramesExist)
  {
    acquisitionSeekFrame(overlayModule,overlayDevice,jumpTo);
  }
  guiSnapFrames(1); //Get New Frames
  return 1;
}


void EditorFrame::OncurrentFrameTextCtrlText(wxCommandEvent& event)
{
    long jumpTo=0;

    if(currentFrameTextCtrl->GetValue().ToLong(&jumpTo))
        {
          if (jumpTo>0) { --jumpTo; }
          doGlobalSeek(jumpTo);
          Refresh(); // <- This draws the window!
        }
}

void EditorFrame::OnFrameSliderCmdScroll(wxScrollEvent& event)
{
    long jumpTo = FrameSlider->GetValue();

    if (jumpTo>0) { --jumpTo; }
    doGlobalSeek(jumpTo);
    Refresh(); // <- This draws the window!
}

void EditorFrame::OnButtonSegmentationClick(wxCommandEvent& event)
{
    SelectSegmentation  * segmentationSelector = new SelectSegmentation(this, wxID_ANY);

    segmentationSelector->selectedCombinationMode=combinationMode;
    copyRGBSegmentation(&segmentationSelector->selectedRGBConf, &segConfRGB);
    copyDepthSegmentation(&segmentationSelector->selectedDepthConf, &segConfDepth);

    printDepthSegmentationData((char*) "Initial Depth Configuration",&segConfDepth);

    segmentationSelector->reloadSegmentationFormFromValues();
    segmentationSelector->ShowModal();

    printDepthSegmentationData((char*) "What Depth Configuration the form filled in ",&segmentationSelector->selectedDepthConf);


    segmentedFramesExist=1;

    combinationMode=segmentationSelector->selectedCombinationMode;
    copyRGBSegmentation(&segConfRGB , &segmentationSelector->selectedRGBConf);
    copyDepthSegmentation(&segConfDepth , &segmentationSelector->selectedDepthConf);


    printDepthSegmentationData((char*) "New Depth Configuration",&segConfDepth);

    delete  segmentationSelector;

    //refreshSegmentedFrame();
    //refreshOverlay(segmentedFramesExist,segmentedRGB,segmentedDepth);
    refreshAllOverlays();
    lastFrameDrawn+=1000;
    guiSnapFrames(0); //Get New Frames
    Refresh();
}

void EditorFrame::OnButtonCalibrationClick(wxCommandEvent& event)
{
   SelectCalibration  * calibrationSelector = new SelectCalibration(this, wxID_ANY);

   if ( !acquisitionGetColorCalibration(moduleID,devID,&calibrationSelector->calib) )
   {
       wxMessageBox(wxT("Error while Getting Calibration!"),wxT("RGBDAcquisition Editor"));
   } else
   {
      calibrationSelector->reloadCalibrationFormFromValues();
      calibrationSelector->ShowModal();
      if ( calibrationSelector->userLikesTheNewCalibration )
          {
            acquisitionSetColorCalibration(moduleID,devID,&calibrationSelector->calib);
          }
   }

  delete  calibrationSelector;
}

void EditorFrame::OnbuttonRecordClick(wxCommandEvent& event)
{
  if (recording)
  {
      recording=0;
      acquisitionStopTargetForFrames(moduleID,devID);
      return;
  }
  SelectTarget * targetSelector = new SelectTarget(this, wxID_ANY);

  targetSelector->moduleID = moduleID;
  targetSelector->devID = devID;

  targetSelector->ShowModal();
  if ( targetSelector->recording ) {
                                     recordedFrames=0;
                                     recording=1;
                                     play=1;
                                     fprintf(stderr,"Recording started\n");
                                     Refresh();
                                   }


  delete  targetSelector;
}

void EditorFrame::OnButtonAcquisitionGraphClick(wxCommandEvent& event)
{
   SelectAcquisitionGraph  * inputConnector = new SelectAcquisitionGraph(this, wxID_ANY);
   inputConnector->ShowModal();
   delete  inputConnector;
}



void EditorFrame::OnButtonGetExtrinsics(wxCommandEvent& event)
{
   GetExtrinsics * extrinsicsSelector = new GetExtrinsics(this, wxID_ANY);


  extrinsicsSelector->moduleID = moduleID;
  extrinsicsSelector->devID = devID;

  extrinsicsSelector->ShowModal();

   delete  extrinsicsSelector;
}

void EditorFrame::OnButtonAddClick(wxCommandEvent& event)
{
  addingPoint=1;

    guiSnapFrames(0); //Get New Frames
    Refresh();
}




long getItemIndex(wxListCtrl * lstctrl)
{
  unsigned int itemsNumber =0;
  long itemIndex = -1;

  for (;;)
   {
      itemIndex = lstctrl->GetNextItem(itemIndex, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
      if (itemIndex == -1) break;
      // Got the selected item index

      wxLogDebug(lstctrl->GetItemText(itemIndex));
      ++itemsNumber;

      if (itemsNumber>512) { return 0; }
  }

  return itemIndex;
}


void EditorFrame::OnButtonRemoveClick(wxCommandEvent& event)
{
  long i = getItemIndex(ListCtrlPoints); // ListCtrlPoints->GetSelectedItemCount();



  fprintf(stderr,"List Active %lu , active %i \n",i,ListCtrlPoints->GetItemCount());

  guiSnapFrames(0); //Get New Frames
  Refresh();
}

void EditorFrame::OnButtonExecuteClick(wxCommandEvent& event)
{
    AddNewElement * ane  = new AddNewElement(this, wxID_ANY);
    ane->segDepth = &segConfDepth;
    ane->segRGB   = &segConfRGB;
    ane->ListCtrlPoints = ListCtrlPoints;
    ane->moduleID = moduleID;
    ane->devID = devID;

    ane->ShowModal();

    segmentedFramesExist=1;
    //refreshSegmentedFrame();
    refreshAllOverlays();
    lastFrameDrawn+=1000;
    guiSnapFrames(0); //Get New Frames
    Refresh();


    delete  ane;
}

void EditorFrame::OnButtonSendDirectCommandClick(wxCommandEvent& event)
{
  //wxMessageBox(wxT("Test"),wxT("Test"));

}

void EditorFrame::OnButtonAFClick(wxCommandEvent& event)
{
  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);


 int i=0;
 for (i=0; i<10; i++) { afPoints[i].R = 123;   afPoints[i].G = 123;   afPoints[i].B = 123; afPoints[i].width = 80; afPoints[i].height= 80; }
 i=0;
 afPoints[i].x1 = 50;   afPoints[i].y1 = width/2; ++i;
 afPoints[i].x1 = 100;   afPoints[i].y1 = width/2; ++i;
 afPoints[i].x1 = 150;   afPoints[i].y1 = width/2; ++i;
 afPoints[i].x1 = 200;   afPoints[i].y1 = width/2; ++i;
 afPoints[i].x1 = 250;   afPoints[i].y1 = width/2; ++i;
 afPoints[i].x1 = 300;   afPoints[i].y1 = width/2; ++i;
 afPoints[i].x1 = 350;   afPoints[i].y1 = width/2; ++i;
 afPoints[i].x1 = 400;   afPoints[i].y1 = width/2; ++i;

  float dimX,dimY,dimZ;


  afPointsActive=0;
  for (i=0; i<10; i++)
  {
   segmentGetDepthBlobDimensions(depthFrame,width,height,afPoints[0].x1,afPoints[0].y1,afPoints[0].width,afPoints[0].height,&dimX,&dimY,&dimZ);
   ++afPointsActive=1;
  }
}
