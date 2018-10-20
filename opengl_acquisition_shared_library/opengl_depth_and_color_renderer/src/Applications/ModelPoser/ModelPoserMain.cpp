/***************************************************************
 * Name:      ModelPoserMain.cpp
 * Purpose:   Code for Application Frame
 * Author:    Ammar Qammaz ()
 * Created:   2018-10-20
 * Copyright: Ammar Qammaz (http://ammar.gr)
 * License:
 **************************************************************/

#include "ModelPoserMain.h"
#include <wx/msgdlg.h>


#include "../../Library/OGLRendererSandbox.h"

//(*InternalHeaders(ModelPoserFrame)
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

//(*IdInit(ModelPoserFrame)
const long ModelPoserFrame::ID_STATICBOX1 = wxNewId();
const long ModelPoserFrame::ID_STATICBOX2 = wxNewId();
const long ModelPoserFrame::ID_SLIDER1 = wxNewId();
const long ModelPoserFrame::ID_SPINCTRL1 = wxNewId();
const long ModelPoserFrame::ID_STATICTEXT1 = wxNewId();
const long ModelPoserFrame::ID_CHOICE1 = wxNewId();
const long ModelPoserFrame::ID_STATICTEXT2 = wxNewId();
const long ModelPoserFrame::ID_SPINCTRL2 = wxNewId();
const long ModelPoserFrame::ID_STATICTEXT3 = wxNewId();
const long ModelPoserFrame::ID_SPINCTRL3 = wxNewId();
const long ModelPoserFrame::ID_HYPERLINKCTRL1 = wxNewId();
const long ModelPoserFrame::ID_BUTTON1 = wxNewId();
const long ModelPoserFrame::ID_BUTTON2 = wxNewId();
const long ModelPoserFrame::idMenuQuit = wxNewId();
const long ModelPoserFrame::idMenuAbout = wxNewId();
const long ModelPoserFrame::ID_STATUSBAR1 = wxNewId();
const long ModelPoserFrame::ID_TIMER1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(ModelPoserFrame,wxFrame)
    //(*EventTable(ModelPoserFrame)
    //*)
END_EVENT_TABLE()


int initialize()
{
  int started=startOGLRendererSandbox(0,0,640,480,1 /*View OpenGL Window*/,0);


  return started;
}



void ModelPoserFrame::onIdle(wxIdleEvent& evt)
{
   fprintf(stderr,"OnIdle\n");
   this->paintNow();
}

void ModelPoserFrame::onPaint(wxPaintEvent& evt)
{
    wxPaintDC dc(this);
    render(dc);
}

void ModelPoserFrame::paintNow()
{
    wxClientDC dc(this);
    render(dc);
}


void ModelPoserFrame::render(wxDC& dc)
{
    /*
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
*/


     wxPen red(wxColour(255,0,0),1,wxSOLID);
     dc.SetPen(red);
     dc.SetBrush(*wxRED_BRUSH); //*wxTRANSPARENT_BRUSH
     dc.DrawCircle(50,50,10); //Recording Mark ON!

 wxSleep(0.01);
 wxYieldIfNeeded();
}








ModelPoserFrame::ModelPoserFrame(wxWindow* parent,wxWindowID id)
{
    //(*Initialize(ModelPoserFrame)
    wxMenuItem* MenuItem2;
    wxMenuItem* MenuItem1;
    wxMenu* Menu1;
    wxMenuBar* MenuBar1;
    wxMenu* Menu2;

    Create(parent, id, _("ModelPoser"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_FRAME_STYLE, _T("id"));
    SetClientSize(wxSize(845,555));
    StaticBox1 = new wxStaticBox(this, ID_STATICBOX1, _("Model View"), wxPoint(8,8), wxSize(576,488), 0, _T("ID_STATICBOX1"));
    StaticBox2 = new wxStaticBox(this, ID_STATICBOX2, _("Control"), wxPoint(592,8), wxSize(240,488), 0, _T("ID_STATICBOX2"));
    SliderPlayWithJoint = new wxSlider(this, ID_SLIDER1, 0, 0, 100, wxPoint(608,80), wxSize(208,27), 0, wxDefaultValidator, _T("ID_SLIDER1"));
    SpinCtrlMin = new wxSpinCtrl(this, ID_SPINCTRL1, _T("0"), wxPoint(704,176), wxSize(120,27), 0, 0, 100, 0, _T("ID_SPINCTRL1"));
    SpinCtrlMin->SetValue(_T("0"));
    StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Minimum"), wxPoint(608,184), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
    Choice1 = new wxChoice(this, ID_CHOICE1, wxPoint(608,32), wxSize(202,31), 0, 0, 0, wxDefaultValidator, _T("ID_CHOICE1"));
    StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("Maximum"), wxPoint(608,152), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
    SpinCtrlMax = new wxSpinCtrl(this, ID_SPINCTRL2, _T("0"), wxPoint(704,144), wxSize(120,27), 0, 0, 100, 0, _T("ID_SPINCTRL2"));
    SpinCtrlMax->SetValue(_T("0"));
    StaticText3 = new wxStaticText(this, ID_STATICTEXT3, _("Initialization"), wxPoint(608,120), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
    SpinCtrlInit = new wxSpinCtrl(this, ID_SPINCTRL3, _T("0"), wxPoint(704,112), wxSize(120,27), 0, 0, 100, 0, _T("ID_SPINCTRL3"));
    SpinCtrlInit->SetValue(_T("0"));
    HyperlinkCtrl1 = new wxHyperlinkCtrl(this, ID_HYPERLINKCTRL1, _("https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer"), wxEmptyString, wxPoint(512,504), wxDefaultSize, wxHL_CONTEXTMENU|wxHL_ALIGN_CENTRE|wxNO_BORDER, _T("ID_HYPERLINKCTRL1"));
    ButtonPrev = new wxButton(this, ID_BUTTON1, _("<"), wxPoint(16,504), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));
    ButtonNext = new wxButton(this, ID_BUTTON2, _(">"), wxPoint(160,504), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
    MenuBar1 = new wxMenuBar();
    Menu1 = new wxMenu();
    MenuItem1 = new wxMenuItem(Menu1, idMenuQuit, _("Quit\tAlt-F4"), _("Quit the application"), wxITEM_NORMAL);
    Menu1->Append(MenuItem1);
    MenuBar1->Append(Menu1, _("&File"));
    Menu2 = new wxMenu();
    MenuItem2 = new wxMenuItem(Menu2, idMenuAbout, _("About\tF1"), _("Show info about this application"), wxITEM_NORMAL);
    Menu2->Append(MenuItem2);
    MenuBar1->Append(Menu2, _("Help"));
    SetMenuBar(MenuBar1);
    StatusBar1 = new wxStatusBar(this, ID_STATUSBAR1, 0, _T("ID_STATUSBAR1"));
    int __wxStatusBarWidths_1[1] = { -1 };
    int __wxStatusBarStyles_1[1] = { wxSB_NORMAL };
    StatusBar1->SetFieldsCount(1,__wxStatusBarWidths_1);
    StatusBar1->SetStatusStyles(1,__wxStatusBarStyles_1);
    SetStatusBar(StatusBar1);
    Timer1.SetOwner(this, ID_TIMER1);
    Timer1.Start(10, false);

    Connect(idMenuQuit,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&ModelPoserFrame::OnQuit);
    Connect(idMenuAbout,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&ModelPoserFrame::OnAbout);
    Connect(ID_TIMER1,wxEVT_TIMER,(wxObjectEventFunction)&ModelPoserFrame::OnTimer1Trigger);
    //*)
}

ModelPoserFrame::~ModelPoserFrame()
{
    //(*Destroy(ModelPoserFrame)
    //*)
}

void ModelPoserFrame::OnQuit(wxCommandEvent& event)
{
    Close();
}

void ModelPoserFrame::OnAbout(wxCommandEvent& event)
{
    wxString msg = wxbuildinfo(long_f);
    wxMessageBox(msg, _("Welcome to..."));
}

void ModelPoserFrame::OnTimer1Trigger(wxTimerEvent& event)
{
       Refresh();
       wxYieldIfNeeded();
}
