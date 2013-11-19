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
#include <wx/msgdlg.h>

#include "../acquisitionSegment/AcquisitionSegment.h"
#include "../acquisition/Acquisition.h"
#include "SelectModule.h"
#include "SelectSegmentation.h"

ModuleIdentifier moduleID = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//
unsigned int devID=0;

unsigned int width , height , fps ;
char openDevice[512];

int play=0;
int lastFrameDrawn=12312312;

int combinationMode=DONT_COMBINE;
struct SegmentationFeaturesRGB segConfRGB={0};
struct SegmentationFeaturesDepth segConfDepth={0};
struct calibration calib;


unsigned int segmentedFramesExist=0;
unsigned char * segmentedRGB=0;
unsigned short * segmentedDepth=0;


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
const long EditorFrame::ID_MENUITEM1 = wxNewId();
const long EditorFrame::idMenuQuit = wxNewId();
const long EditorFrame::ID_MENUSEGMENTATION = wxNewId();
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
    StaticTextJumpTo = new wxStaticText(this, ID_STATICTEXT1, _("Jump To : "), wxPoint(322,528), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
    currentFrameTextCtrl = new wxTextCtrl(this, ID_TEXTCTRL1, _("0"), wxPoint(392,524), wxDefaultSize, wxTE_PROCESS_ENTER, wxDefaultValidator, _T("ID_TEXTCTRL1"));
    dashForFramesRemainingLabel = new wxStaticText(this, ID_STATICTEXT2, _("/ "), wxPoint(474,528), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
    totalFramesLabel = new wxStaticText(this, ID_STATICTEXT3, _("\?"), wxPoint(484,528), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
    ButtonSegmentation = new wxButton(this, ID_BUTTON5, _("Segmentation"), wxPoint(680,524), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON5"));
    MenuBar1 = new wxMenuBar();
    Menu1 = new wxMenu();
    Menu3 = new wxMenuItem(Menu1, ID_MENUITEM1, _("Open Device"), wxEmptyString, wxITEM_NORMAL);
    Menu1->Append(Menu3);
    MenuItem1 = new wxMenuItem(Menu1, idMenuQuit, _("Quit\tAlt-F4"), _("Quit the application"), wxITEM_NORMAL);
    Menu1->Append(MenuItem1);
    MenuBar1->Append(Menu1, _("&File"));
    Menu4 = new wxMenu();
    MenuItem3 = new wxMenuItem(Menu4, ID_MENUSEGMENTATION, _("Segmentation"), wxEmptyString, wxITEM_NORMAL);
    Menu4->Append(MenuItem3);
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
    Timer.Start(100, false);

    Connect(ID_SLIDER1,wxEVT_SCROLL_TOP|wxEVT_SCROLL_BOTTOM|wxEVT_SCROLL_LINEUP|wxEVT_SCROLL_LINEDOWN|wxEVT_SCROLL_PAGEUP|wxEVT_SCROLL_PAGEDOWN|wxEVT_SCROLL_THUMBTRACK|wxEVT_SCROLL_THUMBRELEASE|wxEVT_SCROLL_CHANGED,(wxObjectEventFunction)&EditorFrame::OnFrameSliderCmdScroll);
    Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonPreviousFrameClick);
    Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonPlayClick);
    Connect(ID_BUTTON3,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonStopClick);
    Connect(ID_BUTTON4,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnbuttonNextFrameClick);
    Connect(ID_TEXTCTRL1,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&EditorFrame::OncurrentFrameTextCtrlText);
    Connect(ID_BUTTON5,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&EditorFrame::OnButtonSegmentationClick);
    Connect(idMenuQuit,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnQuit);
    Connect(idMenuAbout,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&EditorFrame::OnAbout);
    Connect(ID_TIMER1,wxEVT_TIMER,(wxObjectEventFunction)&EditorFrame::OnTimerTrigger);
    //*)

//
    initFeeds();


    feed_0_x=10;
    feed_0_y=20;

    feed_1_x=feed_0_x+default_feed->GetWidth()+10;
    feed_1_y=feed_0_y;

    feed_2_x=feed_0_x;
    feed_2_y=feed_0_y+default_feed->GetHeight()+10;

    feed_3_x=feed_1_x;
    feed_3_y=feed_2_y;



    SelectModule  * moduleSelectorFrame = new SelectModule(this, wxID_ANY);
    moduleSelectorFrame->ShowModal();

    fprintf(stderr,"Module %u Device %u  -> %u x %u : %u \n",
                     moduleSelectorFrame->moduleSelected ,
                     moduleSelectorFrame->deviceSelected ,
                     moduleSelectorFrame->widthSelected,
                     moduleSelectorFrame->heightSelected,
                     moduleSelectorFrame->fpsSelected
                     );

   moduleID = moduleSelectorFrame->moduleSelected;
   devID = moduleSelectorFrame->deviceSelected;
   width = moduleSelectorFrame->widthSelected;
   height = moduleSelectorFrame->heightSelected;
   fps = moduleSelectorFrame->fpsSelected;
   strcpy( openDevice , moduleSelectorFrame->deviceNameSelected );

   fprintf(stderr,"Device %s \n", moduleSelectorFrame->deviceNameSelected);

   delete moduleSelectorFrame;


   if (!acquisitionIsModuleLinked(moduleID))
    {
        fprintf(stderr,"The module you are trying to use is not linked in this build of the Acquisition library..\n");
        //return 1;
    }

   //We need to initialize our module before calling any related calls to the specific module..
   if (!acquisitionStartModule(moduleID,16 /*maxDevices*/ , 0 ))
   {
       fprintf(stderr,"Could not start module %s ..\n",getModuleStringName(moduleID));
       //return 1;
    }

   //We want to initialize all possible devices in this example..
   unsigned int devID=0,maxDevID=acquisitionGetModuleDevices(moduleID);
   if (maxDevID==0)
   {
       wxMessageBox(wxT("error while Openning device"),wxT("RGBDAcquisition Editor"));
      fprintf(stderr,"No devices found for Module used \n");
     // return 1;
   }

   if ( ! acquisitionOpenDevice(moduleID,devID,openDevice,width,height,fps) )
   {
       wxMessageBox(wxT("Error while Opening device"),wxT("RGBDAcquisition Editor"));
       fprintf(stderr,"Could not open device \n");
       return;
   }

   unsigned int totalFrames =  acquisitionGetTotalFrameNumber(moduleID,devID);

   if (totalFrames==0) {
                         totalFramesLabel->SetLabel(wxT("Live Stream"));
                         play=1;
                       } else // This means a live stream
                       {
                         wxString msg; msg.Printf(wxT("%u"),totalFrames);
                         totalFramesLabel->SetLabel(msg);
                       }


   initializeRGBSegmentationConfiguration(&segConfRGB,width,height);
   initializeDepthSegmentationConfiguration(&segConfDepth,width,height);

}

EditorFrame::~EditorFrame()
{
    //(*Destroy(EditorFrame)
    //*)
}


int removeOldSegmentedFrames()
{

    if (!segmentedFramesExist ) { return 0; }
    if ( segmentedRGB!=0 ) { free(segmentedRGB); segmentedRGB=0; }
    if ( segmentedDepth!=0 ) { free(segmentedDepth); segmentedDepth=0; }
}


int refreshSegmentedFrame()
{
    removeOldSegmentedFrames();
    segmentedRGB = copyRGB(acquisitionGetColorFrame(moduleID,devID) , width , height);
    segmentedDepth = copyDepth(acquisitionGetDepthFrame(moduleID,devID) , width , height);

    segmentRGBAndDepthFrame (
                              segmentedRGB ,
                              segmentedDepth ,
                              width , height ,
                              &segConfRGB ,
                              &segConfDepth ,
                              &calib ,
                              combinationMode
                             );
    return 1;
}



void EditorFrame::OnQuit(wxCommandEvent& event)
{
    play=0;
    wxSleep(0.1);
    closeFeeds();

    removeOldSegmentedFrames();

    acquisitionCloseDevice(moduleID,devID);
    acquisitionStopModule(moduleID);

    Close();
}

void EditorFrame::OnAbout(wxCommandEvent& event)
{
    wxString msg = wxbuildinfo(long_f);
    wxMessageBox(msg, _("Welcome to..."));
}




void EditorFrame::OnPaint(wxPaintEvent& event)
{
  wxPaintDC dc(this); // OnPaint events should always create a wxPaintDC

  unsigned int devID=0;
  unsigned int width , height , channels , bitsperpixel;

  if (
        (lastFrameDrawn!=acquisitionGetCurrentFrameNumber(moduleID,devID)) ||
        ( (acquisitionGetTotalFrameNumber(moduleID,devID)==0) && (play) )
     )
  {

     if (segmentedFramesExist)
     {
        //DRAW RGB FRAME -------------------------------------------------------------------------------------
       acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
       passVideoRegisterToFeed(0,segmentedRGB,width,height,bitsperpixel,channels);

       //DRAW DEPTH FRAME -------------------------------------------------------------------------------------
       acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);


       unsigned char * rgb = convertShortDepthTo3CharDepth(segmentedDepth,width,height,0,2048);
       //char * rgb = convertShortDepthToRGBDepth(acquisitionGetDepthFrame(moduleID,devID),width,height);
       passVideoRegisterToFeed(1,rgb,width,height,8,3);
       free(rgb);
     } else
     {
       //DRAW RGB FRAME -------------------------------------------------------------------------------------
       acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
       passVideoRegisterToFeed(0,acquisitionGetColorFrame(moduleID,devID),width,height,bitsperpixel,channels);

       //DRAW DEPTH FRAME -------------------------------------------------------------------------------------
       acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);


       unsigned char * rgb = convertShortDepthTo3CharDepth(acquisitionGetDepthFrame(moduleID,devID),width,height,0,2048);
       //char * rgb = convertShortDepthToRGBDepth(acquisitionGetDepthFrame(moduleID,devID),width,height);
       passVideoRegisterToFeed(1,rgb,width,height,8,3);
       free(rgb);
     }

   lastFrameDrawn=acquisitionGetCurrentFrameNumber(moduleID,devID);


   wxString currentFrame;
   currentFrame.clear();
   currentFrame<<lastFrameDrawn;
   currentFrameTextCtrl->SetValue(currentFrame);

   currentFrame.clear();
   currentFrame<<acquisitionGetTotalFrameNumber(moduleID,devID);
   totalFramesLabel->SetLabel(currentFrame);
  }

  dc.DrawBitmap(*live_feeds[0].bmp,feed_0_x,feed_0_y,0); //FEED 1
  dc.DrawBitmap(*live_feeds[1].bmp,feed_1_x,feed_1_y,0); //FEED 2
    //dc.DrawBitmap(*live_feeds[2].bmp,feed_2_x,feed_2_y,0); //FEED 3
    //dc.DrawBitmap(*live_feeds[3].bmp,feed_3_x,feed_3_y,0); //FEED 4

  wxSleep(0.01);

}


inline int XYOverRect(int x , int y , int rectx1,int recty1,int rectx2,int recty2)
{
  if ( (x>=rectx1) && (x<=rectx2) )
    {
      if ( (y>=recty1) && (y<=recty2) )
        {
          return 1;
        }
    }
  return 0;
}


void EditorFrame::OnMotion(wxMouseEvent& event)
{
  wxSleep(0.01);
  int x=event.GetX();
  int y=event.GetY();
  //fprintf(stderr,"Mouse %u,%u\n",x,y);


  int fd_rx1,fd_rx2,fd_ry1,fd_ry2;
  fd_rx1=10;
  fd_ry1=15+default_feed->GetHeight()+10;
  fd_rx2=fd_rx1 + default_feed->GetWidth();
  fd_ry2=fd_ry1 + default_feed->GetHeight();

  if ( XYOverRect(x,y,feed_0_x,feed_0_y,feed_0_x+default_feed->GetWidth(),feed_0_y+default_feed->GetHeight())==1 )
       {
         mouse_x=x;
         mouse_y=y;

         if ( event.LeftIsDown()==1 )
           {
              float x,y,z;
              if ( acquisitionGetDepth3DPointAtXY(moduleID,devID,mouse_x,mouse_y,&x,&y,&z) )
              {

                wxString msg;
                msg.Printf( wxT("Depth at point is  %0.5f,%0.5f,%0.5f") ,x,y,z  );
                Status->SetStatusText(msg);

                fprintf(stderr,"Depth at point is  %0.5f,%0.5f,%0.5f\n",x,y,z);
              } else
              {
                //Status->SetStatusText(wxT("No depth at point"));
              }

           }
       }

}


void EditorFrame::OnTimerTrigger(wxTimerEvent& event)
{
  wxSleep(0.01);
  if (play)
    {
     acquisitionSnapFrames(moduleID,devID);
     refreshSegmentedFrame();
     Refresh(); // <- This draws the window!
    }


}

void EditorFrame::OnbuttonPlayClick(wxCommandEvent& event)
{
    play=1;
}

void EditorFrame::OnbuttonStopClick(wxCommandEvent& event)
{
    play=0;
}

void EditorFrame::OnbuttonPreviousFrameClick(wxCommandEvent& event)
{
    acquisitionSeekRelativeFrame(moduleID,devID,(signed int) -2);
    acquisitionSnapFrames(moduleID,devID);
    refreshSegmentedFrame();
    Refresh(); // <- This draws the window!
}

void EditorFrame::OnbuttonNextFrameClick(wxCommandEvent& event)
{
    //acquisitionSeekRelativeFrame(moduleID,devID,(signed int) +1);
    acquisitionSnapFrames(moduleID,devID);
    refreshSegmentedFrame();
    Refresh(); // <- This draws the window!
}

void EditorFrame::OncurrentFrameTextCtrlText(wxCommandEvent& event)
{
    long jumpTo=0;

    if(currentFrameTextCtrl->GetValue().ToLong(&jumpTo))
        {
          if (jumpTo>0) { --jumpTo; }
          acquisitionSeekFrame(moduleID,devID,jumpTo);
          acquisitionSnapFrames(moduleID,devID);
          refreshSegmentedFrame();
          Refresh(); // <- This draws the window!
        }

}

void EditorFrame::OnFrameSliderCmdScroll(wxScrollEvent& event)
{

    float percentageOfFrames = (float) FrameSlider->GetValue()  /  FrameSlider->GetMax() ;
    fprintf(stderr,"Go to : %u ( %0.2f ) min %0.2f max %0.2f \n",FrameSlider->GetValue(),percentageOfFrames , FrameSlider->GetMin() , FrameSlider->GetMax());
    fprintf(stderr,"Max Frames : %u \n",acquisitionGetTotalFrameNumber(moduleID,devID));


    long jumpTo =  (long) (percentageOfFrames * acquisitionGetTotalFrameNumber(moduleID,devID));


    if (jumpTo>0) { --jumpTo; }
          acquisitionSeekFrame(moduleID,devID,jumpTo);
          acquisitionSnapFrames(moduleID,devID);
          refreshSegmentedFrame();
          Refresh(); // <- This draws the window!

}

void EditorFrame::OnButtonSegmentationClick(wxCommandEvent& event)
{
    SelectSegmentation  * segmentationSelector = new SelectSegmentation(this, wxID_ANY);

    segmentationSelector->selectedCombinationMode=combinationMode;
    segmentationSelector->selectedRGBConf=segConfRGB;
    segmentationSelector->selectedDepthConf=segConfDepth;

    segmentationSelector->ShowModal();

    acquisitionGetColorCalibration(moduleID,devID,&calib);

    segmentedFramesExist=1;

    combinationMode=segmentationSelector->selectedCombinationMode;
    segConfRGB=segmentationSelector->selectedRGBConf;
    segConfDepth=segmentationSelector->selectedDepthConf;



    delete  segmentationSelector;

    refreshSegmentedFrame();

    Refresh();
}
